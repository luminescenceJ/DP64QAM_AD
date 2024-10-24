import torch.nn as nn
import math
import torch

class DPQAM64Classifier(nn.Module):
    def __init__(self, args, Transformer_model_time, Reduce_model_time, Transformer_model_freq, Reduce_model_freq):
        super(DPQAM64Classifier, self).__init__()
        self.args = args
        self.Transformer_model_time = Transformer_model_time
        self.Reduce_model_time = Reduce_model_time
        self.Transformer_model_freq = Transformer_model_freq
        self.Reduce_model_freq = Reduce_model_freq

        for param in self.Transformer_model_time.parameters():
            param.requires_grad = False  # 冻结 DPQAM64 模型的参数
        for param in self.Reduce_model_time.parameters():
            param.requires_grad = False  # 冻结 DPQAM64 模型的参数
        for param in self.Transformer_model_freq.parameters():
            param.requires_grad = False  # 冻结 DPQAM64 模型的参数
        for param in self.Reduce_model_freq.parameters():
            param.requires_grad = False  # 冻结 DPQAM64 模型的参数

        self.fc_time = nn.Linear(12, 1)
        self.fc_freq = nn.Linear(12, 1)
        self.fc = nn.Linear(1366 * 2, 256)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):  # [bs,16384,2]
        (batch_size, _, _) = x.shape
        padding_needed = math.ceil(self.args.seq_len / self.args.modulation) * self.args.modulation - self.args.seq_len
        with torch.no_grad():  # 确保 DPQAM64 模型在前向传播时不更新参数
            time = x[:, :, 0]  # [bs,16384]
            time_pad = self.Reduce_model_time(time)
            time_output = self.Transformer_model_time(time_pad)  # [bs,1376,hidden_ch] #8,1366,12
            freq = x[:, :, 1]  # [bs,16384]
            freq_pad = self.Reduce_model_freq(freq)
            freq_output = self.Transformer_model_freq(freq_pad)  # [bs,1376,hidden_ch] #8,1366,12
            x = torch.cat((time_output.unsqueeze(3), freq_output.unsqueeze(3)), dim=3)  # 8,1366,12,2
        time = self.fc_time(x[:, :, :, 0]).reshape(batch_size, -1)  # bs,1366
        freq = self.fc_freq(x[:, :, :, 1]).reshape(batch_size, -1)  # bs,1366
        x = torch.cat((time, freq), dim=-1)  # bs,128*12*2
        x = self.fc2(self.fc(x))
        return x