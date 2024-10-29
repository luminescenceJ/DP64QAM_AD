import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DP64QAM import Model as DP64QAM


class CNNRestore(nn.Module):
    def __init__(self, args, cnn_dim_list=[64, 128, 192]):
        super(CNNRestore, self).__init__()
        self.args = args
        self.cnn_dim_list = cnn_dim_list  # 控制升维和降维的目标维度列表

        # 使用ModuleList来存储不同升维的卷积层
        self.deconv1_ups = nn.ModuleList()
        self.bn_ups = nn.ModuleList()

        # 使用ModuleList来存储对应的降维卷积层
        self.deconv2_downs = nn.ModuleList()
        self.bn_downs = nn.ModuleList()

        # 遍历cnn_dim_list，为每个指定的维度创建升维和降维层
        for cnn_dim in cnn_dim_list:
            self.deconv1_ups.append(
                nn.ConvTranspose1d(in_channels=self.args.reduce_ch, out_channels=cnn_dim, kernel_size=3, padding=1)
            )
            self.bn_ups.append(nn.BatchNorm1d(cnn_dim))

            self.deconv2_downs.append(
                nn.ConvTranspose1d(in_channels=cnn_dim, out_channels=self.cnn_dim_list[0], kernel_size=3, padding=1)
            )
            self.bn_downs.append(nn.BatchNorm1d(self.cnn_dim_list[0]))

        # 最后一层，恢复到原始的 modulation 维度
        self.final_deconv = nn.ConvTranspose1d(in_channels=self.cnn_dim_list[0], out_channels=self.args.modulation,
                                               kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm1d(self.args.modulation)

    def forward(self, x):
        batch_size, seq_len, seq_ch = x.shape
        x = x.permute(0, 2, 1)  # 变为 [bs, reduce_ch, seq_len]

        # 使用for循环遍历每个升维和降维模块
        out = 0
        for i in range(len(self.cnn_dim_list)):
            # 升维卷积
            x_up = F.gelu(self.bn_ups[i](self.deconv1_ups[i](x)))
            # 降维卷积
            x_down = F.gelu(self.bn_downs[i](self.deconv2_downs[i](x_up)))
            out += x_down  # 累加不同卷积结果

        # 最后一层恢复到 modulation 维度
        x = F.gelu(self.bn_final(self.final_deconv(out)))
        x = x.permute(0, 2, 1)  # 变为 [bs, seq_len, modulation]

        # 将张量拉平成一维并裁剪到指定长度
        x = x.reshape(batch_size, -1)
        return x[:, :self.args.out_len]

class Model(nn.Module):
    def __init__(self, args,prevModel=None):
        super().__init__()
        self.args = args
        self.prevModel = prevModel
        if prevModel == None:
            self.prevModel = DP64QAM(args)
        self.Restore_model_time = CNNRestore(args)
        self.Restore_model_freq = CNNRestore(args)

    def forward(self, x):
        # input  = bs,16384,12,2
        x = self.prevModel(x)

        (batch_size, _, _ ,_) = x.shape

        time = x[:, :, :,0]  # [bs,1366,12]
        time_out = self.Restore_model_time(time)

        freq = x[:, :, :,0]  # [bs,1366,12]
        freq_out = self.Restore_model_freq(freq) #[bs,16384]

        output = torch.cat((time_out.unsqueeze(2), freq_out.unsqueeze(2)), dim=2)
        return output  # output  = bs,1366,12,2

