import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, args, input_dim=1366, hidden_dim=256, num_layers=4, dropout=0.2, seq_len=12,pred_len=1):
        super(Model, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.pred_len = pred_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LSTM层
        self.lstm_time_encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.lstm_freq_encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        self.lstm_time_decoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.lstm_freq_decoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # 全连接层
        self.fc_time = nn.Linear(hidden_dim, self.input_dim)
        self.fc_freq = nn.Linear(hidden_dim, self.input_dim)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim] bs,16384,2 => bs,1366,12,2
        (batch_size, _, _) = x.shape
        padding_needed = math.ceil(self.args.seq_len / self.args.modulation) * self.args.modulation - self.args.seq_len
        time = x[:, :, 0]  # [bs, seq_len]
        freq = x[:, :, 1]  # [bs, seq_len]
        time_pad = F.pad(time, (0, padding_needed), "constant", 0).unsqueeze(2).reshape(self.args.batch_size, -1,
                                                                                        self.args.reduce_ch)  # BS,1366,12
        freq_pad = F.pad(freq, (0, padding_needed), "constant", 0).unsqueeze(2).reshape(self.args.batch_size, -1,
                                                                                        self.args.reduce_ch)


        # LSTM 编码器
        _, decoder_hidden_time = self.lstm_time_encoder(time_pad.permute(0,2,1))  # LSTM输出和隐状态
        _, decoder_hidden_freq = self.lstm_freq_encoder(freq_pad.permute(0,2,1))  # BS,1366,128

        decoder_input_time = torch.zeros(batch_size, self.pred_len, self.input_dim).to(self.device)
        outputs_time = torch.zeros(self.seq_len, batch_size, self.input_dim).to(self.device)

        decoder_input_freq = torch.zeros(batch_size, self.pred_len, self.input_dim).to(self.device)
        outputs_freq = torch.zeros(self.seq_len, batch_size, self.input_dim).to(self.device)

        for t in range(self.seq_len):
            decoder_output_time, decoder_hidden_time = self.lstm_time_encoder(decoder_input_time, decoder_hidden_time)
            decoder_output_time = self.relu(decoder_output_time)
            decoder_input_time = self.fc_time(decoder_output_time)
            outputs_time[t] = torch.squeeze(decoder_input_time, dim=-2)

            decoder_output_freq, decoder_hidden_time = self.lstm_time_encoder(decoder_input_freq, decoder_hidden_freq)
            decoder_output_freq = self.relu(decoder_output_freq)
            decoder_input_freq = self.fc_time(decoder_output_freq)
            outputs_freq[t] = torch.squeeze(decoder_input_freq, dim=-2)

        outputs_freq = outputs_freq.permute(1,0,2).reshape(batch_size, -1,1)[:, :16384,:]
        outputs_time = outputs_time.permute(1,0,2).reshape(batch_size, -1,1)[:, :16384,:]

        outputs = torch.concatenate((outputs_time,outputs_freq),dim=-1)
        return outputs




class args():
    pass
args.modulation = 12
args.reduce_ch = 12
args.seq_len = 16384
args.batch_size = 8
model = Model(args)

input_data = torch.randn(8, 16384, 2)
output_data = model(input_data)

from torchinfo import summary

summary(model,(8,16384,2))

# 输出结果形状
print("Input shape:", input_data.shape)
print("Output shape:", output_data.shape)