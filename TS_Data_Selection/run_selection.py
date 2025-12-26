import argparse
import torch
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from models.linear_proxy import LinearProxy
# Import all selectors / 导入所有选择器
from algorithm.forward_inf import ForwardINFSelector
from algorithm.baseline import RandomSelector, TracInSelector, InfluenceFunctionSelector
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Running Experiment Method: [{args.method}]")

    # 1. Dataset Loading / 数据集加载
    if args.data == 'custom':
        Data = Dataset_Custom
    elif args.data in ['ETTh1', 'ETTh2']:
        Data = Dataset_ETT_hour
    elif args.data in ['ETTm1', 'ETTm2']:
        Data = Dataset_ETT_minute
    else:
        Data = Dataset_Custom

    # Note: For TracIn/IF, to calculate gradients precisely, it is recommended to set a smaller batch_size or handle it in an internal loop
    # Here we keep it consistent and use args.batch_size
    # 注意：对于 TracIn/IF，为了精确计算梯度，建议 batch_size 设小一点，或者在内部循环处理
    # 这里保持一致，使用 args.batch_size
    train_dataset = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )
    val_dataset = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Initialize Model (Random does not need a model) / 初始化模型 (Random 不需要模型)
    if args.method != 'random':
        try:
            sample_x, _, _, _ = train_dataset[0]
            enc_in = sample_x.shape[1]
        except:
            enc_in = 21
        model = LinearProxy(args.seq_len, args.pred_len, enc_in).to(device)
    else:
        model = None

    # 3. Select Selection Strategy / 选择筛选策略
    if args.method == 'forward_inf':
        # Method proposed in the paper / 论文提出的方法
        selector = ForwardINFSelector(model, device, pred_len=args.pred_len)
        selection_mode = args.selection_mode # 'positive', 'gmm', 'top_k'
    elif args.method == 'tracin':
        # Baseline: Gradient Tracing / 基线：梯度追踪
        selector = TracInSelector(model, device, pred_len=args.pred_len)
        selection_mode = 'top_k'
    elif args.method == 'if':
        # Baseline: Influence Function / 基线：影响函数
        selector = InfluenceFunctionSelector(model, device, pred_len=args.pred_len)
        selection_mode = 'top_k'
    elif args.method == 'random':
        # Baseline: Random / 基线：随机
        selector = RandomSelector(pred_len=args.pred_len)
        selection_mode = 'top_k'
    else:
        raise NotImplementedError(f"Method {args.method} not implemented")

    # 4. Execute Selection / 执行筛选
    # Note: Forward-INF uses lr_adapt and adaptation_steps
    # Other methods may ignore these parameters
    # 注意：Forward-INF 使用 lr_adapt 和 adaptation_steps
    # 其他方法可能忽略这些参数
    keep_indices, scores = selector.select_data(
        train_loader, val_loader, 
        lr_adapt=args.lr_adapt, 
        adaptation_steps=args.adapt_steps,
        selection_mode=selection_mode, # Pass mode / 传递模式
        keep_ratio=args.keep_ratio
    )

    # 5. Save Results / 保存结果
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    save_file = os.path.join(args.save_path, f'selected_indices_{args.method}.npy')
    keep_indices = keep_indices.astype(np.int64)
    np.save(save_file, keep_indices)
    
    print(f"Saved {len(keep_indices)} indices to {save_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    
    # Model args
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Selection args
    parser.add_argument('--method', type=str, default='forward_inf', 
                        help='Options: forward_inf, tracin, if, random')
    parser.add_argument('--selection_mode', type=str, default='positive',
                        help='For Forward-INF: positive, gmm, top_k')
    parser.add_argument('--lr_adapt', type=float, default=0.01)
    parser.add_argument('--adapt_steps', type=int, default=1)
    parser.add_argument('--keep_ratio', type=float, default=0.7)
    
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    main(args)