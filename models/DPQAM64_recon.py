import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers import transformer

class CNNReduce(nn.Module):
    def __init__(self,args,):
        super(CNNReduce, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(in_channels=self.args.modulation,out_channels=self.args.CNN_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.args.CNN_dim)
        self.conv2 = nn.Conv1d(in_channels=self.args.CNN_dim,out_channels=self.args.modulation, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(self.args.modulation)

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
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # 变为 [bs,1366,12,]
        return x
class CNNRestore(nn.Module):
    def __init__(self, args):
        super(CNNRestore, self).__init__()
        self.args = args
        # 逆卷积第一层, 恢复到 [bs, 1366, 64]
        self.deconv1 = nn.ConvTranspose1d(in_channels=self.args.reduce_ch, out_channels=self.args.CNN_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.args.CNN_dim)
        # self.relu1 = nn.ReLU()
        # 逆卷积第二层, 恢复到 [bs, 1366, modulation]，即原始的维度 12
        self.deconv2 = nn.ConvTranspose1d(in_channels=self.args.CNN_dim, out_channels=self.args.modulation, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.args.modulation)
        # self.relu2 = nn.ReLU()
    def forward(self, x):
        batch_size, seq_len,seq_ch = x.shape
        x = x.permute(0, 2, 1)  # 变为 [bs, 12, 1366]
        x = F.gelu(self.bn1(self.deconv1(x)))   # 逆卷积第一层, 从 [bs, 1366, latent_dim] -> [bs, 1366, 64]
        x = F.gelu(self.bn2(self.deconv2(x)))   # 逆卷积第二层, 从 [bs, 1366, 64] -> [bs, 1366, 12]
        x = x.permute(0, 2, 1)  # 变为 [bs, 12, 1366]
        x = x.reshape(batch_size,-1)
        return x[:,:self.args.out_len]
class Model(nn.Module):
    def __init__(self,args,cnn_dims=[32,64,128]):
        super().__init__()
        self.args = args

        self.Transformer_model_time = transformer.Model(args)
        self.Reduce_model_time = CNNReduce(args)
        self.Restore_model_time = CNNRestore(args)

        self.Transformer_model_freq = transformer.Model(args)
        self.Reduce_model_freq = CNNReduce(args)
        self.Restore_model_freq = CNNRestore(args)

    def forward(self, x):
        (batch_size , _ ,_) = x.shape
        # padding_needed = math.ceil(self.args.seq_len / self.args.modulation) * self.args.modulation - self.args.seq_len
        time = x[:, :,0] # [bs,16384]
        time_pad = sum([reduce(time) for reduce in self.Reduce_layers_time])
        time_output = self.Transformer_model_time(time_pad) # [bs,1376,hidden_ch] #8,1366,16
        time_output = self.Restore_model_time(time_output)

        freq = x[:, :,1]# [bs,16384]
        freq_pad = sum([reduce(freq) for reduce in self.Reduce_layers_freq])
        freq_output = self.Transformer_model_freq(freq_pad) # [bs,1376,hidden_ch] #8,1366,16
        freq_output = self.Restore_model_freq(freq_output)
        return torch.cat((time_output.unsqueeze(2), freq_output.unsqueeze(2)), dim=2)


