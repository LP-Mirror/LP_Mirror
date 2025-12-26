import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import grad

class RandomSelector:
    """
    Baseline Method 1: Random Selection
    基线方法 1: Random Selection
    """
    def __init__(self, pred_len=96):
        self.pred_len = pred_len

    def select_data(self, train_loader, val_loader, keep_ratio=0.7, **kwargs):
        print(f">>> [Random] Selecting {keep_ratio*100}% samples randomly...")
        total_samples = len(train_loader.dataset)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        num_keep = int(total_samples * keep_ratio)
        return indices[:num_keep], np.random.rand(total_samples)

class TracInSelector:
    """
    Baseline Method 2: TracIn (Gradient-based)
    基线方法 2: TracIn (Gradient-based)
    Score = ∇L_val · ∇L_train
    """
    def __init__(self, model, device, pred_len=96):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.pred_len = pred_len

    def get_flat_grads(self, inputs, targets):
        """Calculate flattened gradients for a single batch / 计算单个批次的平铺梯度"""
        self.model.zero_grad()
        preds = self.model(inputs)
        loss = self.criterion(preds, targets)
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = grad(loss, params, create_graph=False) # create_graph=False saves memory / create_graph=False 节省显存
        return torch.cat([g.reshape(-1) for g in grads])

    def get_val_grads(self, val_loader):
        """Calculate average gradients for validation set / 计算验证集的平均梯度"""
        self.model.eval()
        total_loss = 0
        cnt = 0
        for batch_x, batch_y, _, _ in val_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)[:, -self.pred_len:, :]
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            total_loss += loss * batch_x.shape[0]
            cnt += batch_x.shape[0]
        
        avg_loss = total_loss / cnt
        self.model.zero_grad()
        params = [p for p in self.model.parameters() if p.requires_grad]
        grads = grad(avg_loss, params)
        return torch.cat([g.reshape(-1) for g in grads])

    def select_data(self, train_loader, val_loader, keep_ratio=0.7, **kwargs):
        print(">>> [TracIn] 1. Training Proxy Model...")
        self._train_proxy(train_loader)
        
        print(">>> [TracIn] 2. Computing Validation Gradients...")
        val_grads = self.get_val_grads(val_loader)
        
        print(">>> [TracIn] 3. Computing Scores...")
        scores = []
        self.model.eval()
        for batch_x, batch_y, _, _ in tqdm(train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)[:, -self.pred_len:, :]
            
            # For efficiency, assume average gradient within Batch represents the direction of that Batch
            # Rigorous approach should set batch_size=1, but it is extremely slow in Python
            # 为了效率，这里假设 Batch 内平均梯度代表该 Batch 的方向
            # 严谨做法应设 batch_size=1，但这在 Python 中极慢
            batch_grads = self.get_flat_grads(batch_x, batch_y)
            score = torch.dot(val_grads, batch_grads).item()
            scores.extend([score] * batch_x.shape[0])
            
        scores = np.array(scores)
        num_keep = int(len(scores) * keep_ratio)
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices[:num_keep], scores

    def _train_proxy(self, loader, epochs=1):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y, _, _ in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)[:, -self.pred_len:, :]
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

class InfluenceFunctionSelector(TracInSelector):
    """
    Baseline Method 3: Influence Functions (IF) - LiSSA Implementation
    基线方法 3: Influence Functions (IF) - LiSSA 实现
    Score = - ∇L_val · H^-1 · ∇L_train
    Use LiSSA algorithm to estimate H^-1 · ∇L_val
    使用 LiSSA 算法估算 H^-1 · ∇L_val
    """
    def get_inverse_hvp_lissa(self, v, train_loader, damping=0.01, scale=25.0, recursion_depth=100):
        """
        Use LiSSA algorithm to estimate inverse Hessian-Vector Product: H^-1 v
        This is the most time-consuming and core part of IF calculation.
        使用 LiSSA 算法估算 inverse Hessian-Vector Product: H^-1 v
        这是 IF 计算中最耗时、最核心的部分。
        """
        cur_estimate = v.clone()
        
        # Iterative estimation / 迭代估算
        iterator = iter(train_loader)
        for _ in tqdm(range(recursion_depth), desc="[IF] Estimating HVP (LiSSA)"):
            try:
                batch_x, batch_y, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch_x, batch_y, _, _ = next(iterator)

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)[:, -self.pred_len:, :]
            
            # 1. Calculate Loss for current Batch / 计算当前 Batch 的 Loss
            self.model.zero_grad()
            preds = self.model(batch_x)
            loss = self.criterion(preds, batch_y)
            
            params = [p for p in self.model.parameters() if p.requires_grad]
            
            # 2. Calculate Hessian-Vector Product (HVP) / 计算 Hessian-Vector Product (HVP)
            # HVP = ∇(∇L · v)
            # First calculate gradient ∇L / 首先计算梯度 ∇L
            grads = grad(loss, params, create_graph=True) # create_graph=True is needed for second derivative / create_graph=True 才能求二阶导
            flat_grads = torch.cat([g.reshape(-1) for g in grads])
            
            # Then calculate dot product of ∇L and current estimate vector cur_estimate
            # Need to detach cur_estimate here, otherwise memory will explode
            # 然后计算 ∇L 与 当前估计向量 cur_estimate 的点积
            # 这里需要 detach cur_estimate，否则显存会爆炸
            grad_dot_est = torch.dot(flat_grads, cur_estimate.detach())
            
            # Finally differentiate the dot product to get H · cur_estimate
            # 最后对点积求导，得到 H · cur_estimate
            hvp = grad(grad_dot_est, params)
            flat_hvp = torch.cat([g.reshape(-1) for g in hvp])
            
            # 3. LiSSA update step: v_new = v + (I - damping*H) v_old
            # Formula: est = v + (1 - damping) * est - 1/scale * HVP
            # Note: scaling and damping here need to be fine-tuned based on model Loss magnitude
            # For simplicity, use standard Neumann series form: est = v + (est - H * est)
            # Add damping to prevent eigenvalues > 1 causing divergence
            # 3. LiSSA 更新步: v_new = v + (I - damping*H) v_old
            # 公式: est = v + (1 - damping) * est - 1/scale * HVP
            # 注意: 这里的 scaling 和 damping 需要根据模型 Loss 的量级细调
            # 为简化，使用标准 Neumann 级数形式: est = v + (est - H * est)
            # 加入 damping 防止特征值 > 1 导致发散
            cur_estimate = v + (1 - damping) * cur_estimate - (1/scale) * flat_hvp
            
            # Release memory / 释放显存
            del hvp, flat_hvp, grads, flat_grads, grad_dot_est
            torch.cuda.empty_cache()
            
        return cur_estimate

    def select_data(self, train_loader, val_loader, keep_ratio=0.7, **kwargs):
        print(">>> [IF-LiSSA] 1. Training Proxy Model...")
        self._train_proxy(train_loader)
        
        print(">>> [IF-LiSSA] 2. Computing Validation Gradients (v)...")
        val_grads = self.get_val_grads(val_loader)
        
        print(">>> [IF-LiSSA] 3. Estimating Inverse Hessian-Vector Product (s_test = H^-1 v)...")
        # Warning: LiSSA is very slow and sensitive to hyperparameters.
        # damping: Damping coefficient to prevent non-positive definite Hessian
        # scale: Scaling factor to prevent H norm being too large causing series divergence (usually approx max_eigenvalue)
        # depth: Iteration depth, deeper is more accurate but slower
        # 警告：LiSSA 非常慢且对超参数敏感。
        # damping: 阻尼系数，防止Hessian非正定
        # scale: 缩放因子，防止 H 的模过大导致级数发散 (通常取 max_eigenvalue 的近似)
        # depth: 迭代深度，越深越准但越慢
        s_test = self.get_inverse_hvp_lissa(
            val_grads, train_loader, 
            damping=0.01, scale=100.0, recursion_depth=50 # Example params, adjust as needed / 示例参数，可根据实际情况调整
        )
        
        print(">>> [IF-LiSSA] 4. Computing Influence Scores...")
        scores = []
        self.model.eval()
        
        # Influence = - s_test · ∇L_train
        for batch_x, batch_y, _, _ in tqdm(train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)[:, -self.pred_len:, :]
            
            # Calculate training sample gradients / 计算训练样本梯度
            batch_grads = self.get_flat_grads(batch_x, batch_y)
            
            # Calculate IF score
            # Original formula has negative sign, indicating increase in Loss if point is removed (i.e., contribution of point to reducing Loss)
            # Larger Score means the point is more important (beneficial for reducing Val Loss)
            # 计算 IF 分数
            # 原始公式为负号，表示移除该点对Loss的增加量（即该点对降低Loss的贡献）
            # Score 越大，表示该点越重要（对降低 Val Loss 有益）
            score = -torch.dot(s_test, batch_grads).item()
            scores.extend([score] * batch_x.shape[0])
            
        scores = np.array(scores)
        
        # Selection / 筛选
        num_keep = int(len(scores) * keep_ratio)
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices[:num_keep], scores