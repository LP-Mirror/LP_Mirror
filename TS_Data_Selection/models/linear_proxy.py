import torch
import torch.nn as nn

class LinearProxy(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(LinearProxy, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Simple linear mapping: Input(Seq) -> Output(Pred)
        # This implementation is Channel-Independent, meaning all channels share the same temporal linear layer
        # 简单的线性映射: Input(Seq) -> Output(Pred)
        # 这种实现方式是 Channel-Independent 的，即所有通道共享同一个时间线性层
        self.linear = nn.Linear(seq_len, pred_len)
        self.enc_in = enc_in
        
        # === [Critical Fix] Add output_features attribute / [关键修复] 添加 output_features 属性 ===
        self.output_features = 'M' 
        # ==========================================

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        # Linear operates on the last dimension, we need to operate on the time dimension
        # So transpose first: [Batch, Channels, Seq_Len]
        # Linear是在最后一个维度操作的，我们需要对时间维度操作
        # 所以先转置: [Batch, Channels, Seq_Len]
        x = x.permute(0, 2, 1) 
        x = self.linear(x)
        # Transpose back: [Batch, Channels, Pred_Len] -> [Batch, Pred_Len, Channels]
        # 转置回来: [Batch, Channels, Pred_Len] -> [Batch, Pred_Len, Channels]
        x = x.permute(0, 2, 1)
        return x