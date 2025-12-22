import argparse
import torch
import numpy as np
import os
import sys

# 将当前目录加入路径，确保能导入项目模块
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader, Subset
from models.linear_proxy import LinearProxy
from algorithm.forward_inf import ForwardINFSelector

# 导入所有 Dataset 类
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

def get_dataset_class(data_name):
    """根据名称返回对应的 Dataset 类"""
    if data_name == 'custom': 
        return Dataset_Custom
    elif data_name in ['ETTh1', 'ETTh2']: 
        return Dataset_ETT_hour
    elif data_name in ['ETTm1', 'ETTm2']: 
        return Dataset_ETT_minute
    else:
        return Dataset_Custom

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =======================================================
    # 1. 加载 Source Dataset (被筛选的大规模数据, 候选池)
    # =======================================================
    Source_Data = get_dataset_class(args.source_data_type)
    print(f"Loading SOURCE dataset: {args.source_data_path} ({Source_Data.__name__})")
    
    source_dataset = Source_Data(
        root_path=args.root_path,
        data_path=args.source_data_path,
        flag='train', # 加载全部训练集作为候选
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )
    # shuffle=False 以保持索引对应
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)
    
    # =======================================================
    # 2. 加载 Target Dataset (筛选标准/指南针)
    # =======================================================
    Target_Data = get_dataset_class(args.target_data_type)
    print(f"Loading TARGET dataset: {args.target_data_path} ({Target_Data.__name__})")

    target_dataset_full = Target_Data(
        root_path=args.root_path,
        data_path=args.target_data_path,
        flag='train', # 同样使用训练集模式
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )
    
    # 截取 Target 数据的一小部分 (由 target_ratio 控制，默认 10%)
    subset_len = int(len(target_dataset_full) * args.target_ratio)
    # 确保至少有几个样本
    subset_len = max(subset_len, 10) 
    indices = list(range(subset_len))
    
    target_subset = Subset(target_dataset_full, indices)
    
    # shuffle=True 用于随机梯度下降
    target_loader = DataLoader(target_subset, batch_size=args.batch_size, shuffle=True)

    print(f"Source Data Size (Candidates): {len(source_dataset)}")
    print(f"Target Data Size (Guide):      {len(target_subset)} (Top {args.target_ratio*100}% of Target)")

    # =======================================================
    # 3. 初始化代理模型 (Linear Proxy)
    # =======================================================
    # 尝试自动推断输入通道数
    try:
        sample_x, _, _, _ = source_dataset[0]
        enc_in = sample_x.shape[1]
    except:
        enc_in = 7 # 默认值，如果推断失败
        print(f"Warning: Could not infer input channels, using default: {enc_in}")
        
    model = LinearProxy(args.seq_len, args.pred_len, enc_in).to(device)

    # =======================================================
    # 4. 运行 Forward-INF 跨域筛选
    # =======================================================
    # 初始化筛选器，务必传入 pred_len
    selector = ForwardINFSelector(model, device, pred_len=args.pred_len)
    
    # 执行筛选
    keep_indices, scores = selector.select_data(
        source_loader,   # 候选数据
        target_loader,   # 指导数据
        lr_adapt=args.lr_adapt, 
        adaptation_steps=args.adapt_steps,
        selection_mode=args.selection_mode, # 自适应模式: positive, gmm, top_k
        keep_ratio=args.keep_ratio          # 仅在 top_k 模式下生效
    )

    # =======================================================
    # 5. 保存结果
    # =======================================================
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # 转换为 int64，防止 PyTorch DataLoader 读取时报错
    keep_indices = keep_indices.astype(np.int64)
    
    np.save(os.path.join(args.save_path, 'selected_indices.npy'), keep_indices)
    np.save(os.path.join(args.save_path, 'influence_scores.npy'), scores)
    
    print(f"\n[Success] Selected {len(keep_indices)} samples from {args.source_data_path}")
    print(f"          Saved to: {os.path.join(args.save_path, 'selected_indices.npy')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-Domain Data Selection via Forward-INF')
    
    # 基础路径
    parser.add_argument('--root_path', type=str, default='/root/dataset/ETT-small/')
    
    # Source Dataset (被筛选的源域数据)
    parser.add_argument('--source_data_path', type=str, required=True, help='e.g., ETTh2.csv')
    parser.add_argument('--source_data_type', type=str, default='custom', help='custom, ETTh1, ETTh2, etc.')
    
    # Target Dataset (用于指导的目标域数据)
    parser.add_argument('--target_data_path', type=str, required=True, help='e.g., ETTh1.csv')
    parser.add_argument('--target_data_type', type=str, default='custom', help='custom, ETTh1, ETTh2, etc.')
    parser.add_argument('--target_ratio', type=float, default=0.1, help='Ratio of target data used for guidance (default: 10%)')
    
    # 序列参数
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=128)
    
    # 算法参数
    parser.add_argument('--lr_adapt', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=10)
    
    # 自适应筛选参数
    parser.add_argument('--selection_mode', type=str, default='positive', 
                        choices=['positive', 'gmm', 'top_k'],
                        help='Strategy: positive (score>0), gmm (clustering), or top_k (fixed ratio)')
    parser.add_argument('--keep_ratio', type=float, default=0.5, 
                        help='Only used when selection_mode is top_k')
    
    # 输出路径
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    main(args)