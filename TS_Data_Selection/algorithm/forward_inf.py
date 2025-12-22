import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture # [新增] 用于自适应聚类

class ForwardINFSelector:
    def __init__(self, model, device, criterion=None, pred_len=96):
        self.model = model
        self.device = device
        self.criterion = criterion if criterion else nn.MSELoss(reduction='none')
        self.pred_len = pred_len

    def select_data(self, train_loader, val_loader, lr_adapt=0.001, adaptation_steps=1, 
                    selection_mode='gmm', keep_ratio=0.7):
        """
        Args:
            selection_mode (str): 筛选策略
                - 'positive': 自适应，保留所有影响力分数 > 0 的样本 (推荐)
                - 'gmm': 自适应，使用高斯混合模型自动聚类，保留高分簇
                - 'top_k': 固定比例，使用 keep_ratio 参数
            keep_ratio (float): 仅在 selection_mode='top_k' 时生效
        """
        print(">>> 1. Training Proxy Model on Full Source Set...")
        self._train_proxy(train_loader)
        
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        print(">>> 2. Performing Target Adaptation (Mirrored Influence)...")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr_adapt)
        self.model.train()
        
        for _ in range(adaptation_steps):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.pred_len:, :] # 切片

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y).mean()
                loss.backward()
                optimizer.step()
        
        adapted_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        print(">>> 3. Calculating Influence Scores for Candidates...")
        scores = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.pred_len:, :]
                
                # Load Original
                self.model.load_state_dict(original_state)
                pred_orig = self.model(batch_x)
                loss_orig = self.criterion(pred_orig, batch_y).mean(dim=(1,2))
                
                # Load Adapted
                self.model.load_state_dict(adapted_state)
                pred_adapted = self.model(batch_x)
                loss_adapted = self.criterion(pred_adapted, batch_y).mean(dim=(1,2))
                
                # Score = Loss_Old - Loss_New (Positive means helpful)
                batch_scores = (loss_orig - loss_adapted).cpu().numpy()
                scores.append(batch_scores)
        
        scores = np.concatenate(scores)
        
        # ==========================================
        # [核心修改] 自适应筛选逻辑
        # ==========================================
        indices = np.arange(len(scores))
        
        if selection_mode == 'positive':
            # 策略1: 正向阈值 (保留所有有益样本)
            # 设定一个小阈值 1e-6 防止浮点误差
            keep_mask = scores > 1e-6
            keep_indices = indices[keep_mask]
            print(f"[Adaptive-Positive] Cutoff at score > 0")
            
        elif selection_mode == 'gmm':
            # 策略2: GMM 聚类 (自动寻找好坏分界线)
            print("[Adaptive-GMM] Fitting Gaussian Mixture Model...")
            reshaped_scores = scores.reshape(-1, 1)
            try:
                # 拟合两个高斯分布：一个代表噪声/无关，一个代表有效数据
                gmm = GaussianMixture(n_components=2, random_state=42)
                labels = gmm.fit_predict(reshaped_scores)
                means = gmm.means_.flatten()
                
                # 找出均值较大的那个簇（代表 Score 高，即更有益）
                good_cluster_label = np.argmax(means)
                keep_indices = indices[labels == good_cluster_label]
                print(f"[Adaptive-GMM] Selected cluster with mean score: {means[good_cluster_label]:.4f}")
            except:
                print("[Adaptive-GMM] GMM failed (maybe data too uniform), falling back to positive mode.")
                keep_indices = indices[scores > 0]

        else: 
            # 策略3: 固定比例 (Top-K)
            num_keep = int(len(scores) * keep_ratio)
            sorted_indices = np.argsort(scores)[::-1]
            keep_indices = sorted_indices[:num_keep]
            print(f"[Fixed-Ratio] Keeping top {keep_ratio*100}%")

        # 打印统计信息
        original_count = len(scores)
        kept_count = len(keep_indices)
        ratio = kept_count / original_count * 100
        print(f">>> Selection Result: {kept_count}/{original_count} samples kept ({ratio:.2f}%)")
        
        return keep_indices, scores

    def _train_proxy(self, loader, epochs=1):
        # ... (保持不变) ...
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(epochs):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y[:, -self.pred_len:, :] # 切片修正

                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()