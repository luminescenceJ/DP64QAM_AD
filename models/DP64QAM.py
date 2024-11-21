import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import transformer

class MixerReduce(nn.Module):
    def __init__(self,args,kernel_sizes=[3, 5, 7]):
        super(MixerReduce, self).__init__()
        self.args = args
        self.kernel_sizes = kernel_sizes  # 不同卷积核大小列表

        # 使用ModuleList来存储不同卷积层
        self.conv_ups = nn.ModuleList()
        self.bn_ups = nn.ModuleList()

        # 为每个卷积核大小创建不同的卷积层
        for kernel_size in kernel_sizes:
            self.conv_ups.append(
                nn.Conv1d(in_channels=self.args.modulation, out_channels=self.args.d_model,
                          kernel_size=kernel_size, padding=(kernel_size // 2))
            )
            self.bn_ups.append(nn.BatchNorm1d(self.args.d_model))

    def Padding(self, x, modulation=12):
        batch_size, seq_len = x.shape  # 只考虑2维张量，确保输入形状为 [batch_size, seq_len]
        desired_len = math.ceil(seq_len / modulation) * modulation
        padding_needed = desired_len - seq_len
        if padding_needed > 0:
            # 在第二维度 (时间维度) 末尾填充 zeros
            x = F.pad(x, (0, padding_needed), "constant", 0)  # 只在 seq_len 维度后面填充
        x = x.view(batch_size, desired_len // modulation, modulation)  # [bs,seq_len/modulation,modulation]
        return x

    def forward(self, x):
        x = self.Padding(x, self.args.modulation)# 输入形状为 [bs, 1366,12]
        assert x.shape[1] == 1366 and x.shape[2]==12, f"x.shape is {x.shape}"
        x = x.permute(0, 2, 1)  # 变为 [bs, 12, 1366]
        out = 0
        # 遍历每个卷积核大小
        for i, kernel_size in enumerate(self.kernel_sizes):
            # 使用不同卷积核的卷积操作
            x_up = F.gelu(self.bn_ups[i](self.conv_ups[i](x)))  # 升维卷积
            out += x_up  # 累加不同卷积核的结果
        out = out.permute(0, 2, 1)  # 变为 [bs, 1366, d_model]
        return out
class Model(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        self.Transformer_model_time = transformer.Model(args)
        self.Reduce_layers_time = MixerReduce(args)

        self.Transformer_model_freq = transformer.Model(args)
        self.Reduce_layers_freq = MixerReduce(args)
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

