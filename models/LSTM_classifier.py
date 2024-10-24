import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Model(nn.Module):
    def __init__(self, args, input_dim=12, hidden_dim=128, num_layers=4, dropout=0.2, seq_len=1366, num_classes=9):
        super(Model, self).__init__()
        self.args = args

        # LSTM层
        self.lstm_time = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.lstm_freq = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # 全连接层
        self.fc1_time = nn.Linear(seq_len, hidden_dim)
        self.fc2_time = nn.Linear(hidden_dim, num_classes)

        self.fc1_freq = nn.Linear(seq_len, hidden_dim)
        self.fc2_freq = nn.Linear(hidden_dim, num_classes)

        # 激活函数
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        (batch_size, _, _) = x.shape
        padding_needed = math.ceil(self.args.seq_len / self.args.modulation) * self.args.modulation - self.args.seq_len
        time = x[:, :, 0]  # [bs, seq_len]
        freq = x[:, :, 1]  # [bs, seq_len]
        time_pad = F.pad(time, (0, padding_needed), "constant", 0).unsqueeze(2).reshape(self.args.batch_size, -1, self.args.reduce_ch) #BS,1366,12
        freq_pad = F.pad(freq, (0, padding_needed), "constant", 0).unsqueeze(2).reshape(self.args.batch_size, -1, self.args.reduce_ch)

        # LSTM 编码器
        encoded_time, (hn_time, cn_time) = self.lstm_time(time_pad)  # LSTM输出和隐状态
        encoded_freq, (hn_freq, cn_freq) = self.lstm_freq(freq_pad)  #BS,1366,256

        # 全连接层处理 - 频域
        out_freq = self.fc1_freq(encoded_freq.permute(0, 2, 1))
        out_freq = self.relu(out_freq)
        out_freq = self.fc2_freq(out_freq)  #BS,256,9

        # 全连接层处理 - 时域
        out_time = self.fc1_time(encoded_time.permute(0, 2, 1))
        out_time = self.relu(out_time)
        out_time = self.fc2_time(out_time)  # bs, 9

        # 组合时域和频域的输出
        out = torch.cat((out_time.permute(0, 2, 1), out_freq.permute(0, 2, 1)), dim=-1)  # bs,9,512
        return self.final(out).reshape(batch_size, -1)