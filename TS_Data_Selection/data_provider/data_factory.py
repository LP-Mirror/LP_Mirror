from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
# === 新增导入 ===
from torch.utils.data import Subset 
import numpy as np
import os
# ================

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
    else:
        # train and val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 1. 创建原始数据集 (这一步保证了Scaler是在完整训练集上拟合的，非常重要)
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    
    # === 新增代码：应用数据筛选 ===
    # 仅在训练阶段 ('train') 且 用户提供了筛选文件路径时 执行
    if flag == 'train' and args.data_selection_path is not None:
        if os.path.exists(args.data_selection_path):
            print(f"\n>>>>>> Loading selected data indices from: {args.data_selection_path}")
            # 加载索引
            selected_indices = np.load(args.data_selection_path)
            
            # 记录原始大小用于对比
            original_len = len(data_set)
            
            # 核心操作：使用Subset只保留选中的样本
            data_set = Subset(data_set, selected_indices)
            
            print(f">>>>>> Data Selection Applied. Original Size: {original_len} -> New Size: {len(data_set)}")
            print(f">>>>>> Retention Ratio: {len(data_set)/original_len*100:.2f}%\n")
        else:
            print(f"\n>>>>>> Warning: Selection file not found at {args.data_selection_path}. Training on FULL dataset.\n")
    # ============================

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader