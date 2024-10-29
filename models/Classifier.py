import torch.nn as nn
import math
import torch
from models.DP64QAM import Model as DP64QAM

class Model(nn.Module):
    def __init__(self,args,prevModel=None):
        super(Model, self).__init__()
        self.prevModel = prevModel
        if prevModel == None:
            self.prevModel = DP64QAM(args)

        self.fc_time = nn.Linear(12, 1)
        self.fc_freq = nn.Linear(12, 1)
        self.fc = nn.Linear(1366 * 2, 256)
        self.fc2 = nn.Linear(256, 9)
    def forward(self, x):  # [bs,16384,2]
        x = self.prevModel(x)
        (batch_size, _, _,_) = x.shape #bs ,1366,12,2
        time = self.fc_time(x[:, :, :, 0]).reshape(batch_size, -1)  # bs,1366
        freq = self.fc_freq(x[:, :, :, 1]).reshape(batch_size, -1)  # bs,1366
        x = torch.cat((time, freq), dim=-1)  # bs,128*12*2
        x = self.fc2(self.fc(x))
        return x