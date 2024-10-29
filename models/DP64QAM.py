import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import transformer

class CNNReduce(nn.Module):
    def __init__(self,args):
        super(CNNReduce, self).__init__()
        self.args = args
        # 第一层卷积，升维，将 12 -> 更高的维度
        self.conv1 = nn.Conv1d(in_channels=self.args.modulation, out_channels=self.args.CNN_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.args.CNN_dim)
        # self.gelu1 = nn.GeLU()
        # 第二层卷积，降低维度
        self.conv2 = nn.Conv1d(in_channels=self.args.CNN_dim, out_channels=self.args.reduce_ch, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(self.args.reduce_ch)
        # self.gelu2 = nn.GeLU()

    def Padding(self, x, modulation=12):
        batch_size, seq_len = x.shape  # 只考虑2维张量，确保输入形状为 [batch_size, seq_len]
        desired_len = math.ceil(seq_len / modulation) * modulation
        padding_needed = desired_len - seq_len
        if padding_needed > 0:
            # 在第二维度 (时间维度) 末尾填充 zeros
            x = F.pad(x, (0, padding_needed), "constant", 0)  # 只在 seq_len 维度后面填充
        # 重新调整形状，确保时间维度能被 modulation 整除
        x = x.view(batch_size, desired_len // modulation, modulation)  # [bs,seq_len/modulation,modulation]
        return x

    def forward(self, x):
        x = self.Padding(x, self.args.modulation)# 输入形状为 [bs, 1366,12]
        assert x.shape[1] == 1366 and x.shape[2]==12, f"x.shape is {x.shape}"
        x = x.permute(0, 2, 1)  # 变为 [bs, 12, 1366]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # 变为 [bs,1366,12,]
        return x

class Model(nn.Module):
    def __init__(self,args,cnn_dims=[64,128,192]):
        super().__init__()
        self.args = args

        self.Transformer_model_time = transformer.Model(args)
        self.Reduce_layers_time = nn.ModuleList([CNNReduce(args, dim) for dim in cnn_dims])

        self.Transformer_model_freq = transformer.Model(args)
        self.Reduce_layers_freq = nn.ModuleList([CNNReduce(args, dim) for dim in cnn_dims])
    def forward(self, x):
        # input  = bs,16384,2

        (batch_size, _, _) = x.shape
        padding_needed = math.ceil(self.args.seq_len / self.args.modulation) * self.args.modulation - self.args.seq_len

        time = x[:, :,0] # [bs,16384]
        time_pad = self.Reduce_layers_time(time)
        time_output = self.Transformer_model_time(time_pad) # [bs,1376,hidden_ch] #8,1366,16

        freq = x[:, :,1]# [bs,16384]
        freq_pad = self.Reduce_layers_freq(freq)
        freq_output = self.Transformer_model_freq(freq_pad) # [bs,1376,hidden_ch] #8,1366,16

        output = torch.cat((time_output.unsqueeze(3), freq_output.unsqueeze(3)), dim=3)
        return output # output  = bs,1366,12,2

