import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
import os
import glob

class MakeDataset():
    def __init__(self,args,pattern='abnormal',scale=True,quickLoad=False,distanceNum=9,scaleMethod="std"):
        assert pattern in ['abnormal', 'normal'] ,"wrong pattern"
        if quickLoad:
            self.data = np.load(os.path.join(args.path,pattern + "-" + 'data.npy'))
            if len(self.data) == 0:
                raise Exception("no quickLoad file")
#             print(self.data.shape)
            return
        self.seq_len = args.seq_len
        self.iteration = args.iteration
        self.path = args.path
        self.pattern = pattern
        self.scale = scale
        self.distanceNum = distanceNum
        self.data = self.__read_data__(scaleMethod)
        self.__save_data__()
        print(pattern + " dataset has saved successfully")
    def get_data(self):
        return self.data
    def __save_data__(self):
        np.save(os.path.join(self.path,self.pattern + "-" + 'data.npy'), self.data)
    def __read_data__(self,scaleMethod="std"):
        if self.pattern == 'abnormal':
            tempPathSpectrum = "abnormal/spectrum"
            tempPathTime = "abnormal/time"
        elif self.pattern == 'normal':
            tempPathSpectrum = "normal/spectrum"
            tempPathTime = "normal/time"
        else:
            raise Exception("wrong pattern in MakeDataset")
        freq_data = self.__readSingleFolder__(os.path.join(self.path, tempPathSpectrum))    # [9,30*fileNum,16384]
        time_data = self.__readSingleFolder__(os.path.join(self.path, tempPathTime))        # [9,30*fileNum,16384]
        if self.scale:
            # 对 freq_data 和 time_data 进行归一化
            freq_data, freq_scalers = normalize_data(freq_data,scaleMethod)
            time_data, time_scalers = normalize_data(time_data,scaleMethod)
#             print("归一化后的数据形状 (freq): ", freq_data.shape)
#             print("归一化后的数据形状 (time): ", time_data.shape)
            self.scalerx_time = time_scalers
            self.scalerx_freq = freq_scalers
        data = np.concatenate((np.expand_dims(time_data, -1), np.expand_dims(freq_data, -1)),axis=-1)  # [9,30*fileNum,16384,2]
        data = winsorize_per_channel(data)
        return data
    def __readSingleFolder__(self,directory):
        txt_files = glob.glob(os.path.join(directory, '*.txt'))
        res = np.array([])
        for file in txt_files:         # 遍历并打开每个txt文件
            with open(file, 'r') as file:
                lines = file.readlines()
                amplitudes = []
                started_reading = False
                for line in lines:
                    if line.startswith('%Sweep Iteration'):
                        started_reading = True
                        continue
                    if started_reading and not line.startswith('%') and not line.startswith('\n'):
                        parts = line.strip().split('\t')
                        amplitudes.append(float(parts[1]))
                amplitudes = np.array(amplitudes).reshape(self.distanceNum,self.iteration//self.distanceNum, self.seq_len) # 270 16384 => 9,30

                if len(res) == 0:
                    res = amplitudes
                else:
                    res = np.append(res, amplitudes, axis=1)
        return res          # [9,30*fileNum,16384]
def normalize_data(data,scaler='std'):
    """
    对数据进行归一化，针对不同的前两维度组合使用不同的 StandardScaler。

    :param data: 输入数据，形状为 [9, 30*fileNum, 16384]
    :return: 归一化后的数据，形状不变
    """
    s = StandardScaler
    if scaler != 'std':
        s = MinMaxScaler
    # 获取数据的形状
    num_1, num_2, num_3 = data.shape

    # 创建用于存储不同Scaler的列表
    scalers = [[s() for _ in range(num_2)] for _ in range(num_1)]

    # 对每个 [i, j] 的组合使用不同的 Scaler 对象
    for i in range(num_1):
        for j in range(num_2):
            # 对第 16384 维度进行归一化
            data[i, j, :] = scalers[i][j].fit_transform(data[i, j, :].reshape(-1, 1)).flatten()

    return data, scalers  # 返回归一化后的数据和scalers
def winsorize_per_channel(data, lower_percentile=0.03, upper_percentile=0.97):
    """
    对 [num, 16384, 2] 的数据的每个通道分别进行 Winsorization。

    :param data: 输入数据，形状为 [num, 16384, 2]
    :param lower_percentile: 下百分位数，如 0.01 表示 1%
    :param upper_percentile: 上百分位数，如 0.99 表示 99%
    :return: 经过 Winsorization 的数据，形状不变
    """
    # 分别处理通道 0 和通道 1
    for i in range(data.shape[-1]):  # 遍历最后一维度的每个通道
        lower_limit = np.percentile(data[:, :, i], lower_percentile * 100)
        upper_limit = np.percentile(data[:, :, i], upper_percentile * 100)

        # 对每个通道进行裁剪
        data[:, :, i] = np.clip(data[:, :, i], lower_limit, upper_limit)

    return data

