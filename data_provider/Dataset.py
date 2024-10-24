import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils import shuffle
import os
from data_provider.data_loader import MakeDataset

class DP64QAM_Dataset():
    def __init__(self, args, flag='valid', quickLoad=True, ratio=0.8):
        assert flag in ['train', 'valid', 'test']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.args = args
        self.quickLoad = quickLoad
        self.ratio = ratio
        self.set_type = type_map[flag]
        self.__load__()
        self.__process__()
        self.__show__(flag, quickLoad=quickLoad, ratio=ratio)

    def __show__(self, flag, quickLoad, ratio):
        print(
            f"{flag} dataset prepared,data shape is {self.data.shape},label shape is {self.label.shape} , quickLoad : {self.quickLoad} , {self.ratio} of normal data used for train and valid dataset")

    def __load__(self):
        if os.path.exists(os.path.join(self.args.path, "abnormal-data.npy")):
            self.AbnormalDataset = MakeDataset(self.args, pattern='abnormal', quickLoad=self.quickLoad)
        else:
            print("abnormal-data.npy not found!")
            self.AbnormalDataset = MakeDataset(self.args, pattern='abnormal', quickLoad=False)
        if os.path.exists(os.path.join(self.args.path, "normal-data.npy")):
            self.NormalDataset = MakeDataset(self.args, pattern='normal', quickLoad=self.quickLoad)
        else:
            print("normal-data.npy not found!")
            self.NormalDataset = MakeDataset(self.args, pattern='normal', quickLoad=False)

    def __process__(self):
        yes_data = self.NormalDataset.get_data()
        no_data = self.AbnormalDataset.get_data()

        # [9,itr*fileNum,16384,2]   归一化
        num_class, itr, length, channel = yes_data.shape
        for i in range(num_class):
            time = yes_data[i, :, :, 0]
            s = MinMaxScaler()
            time = s.fit_transform(time)
            freq = yes_data[i, :, :, 1]
            s = MinMaxScaler()
            freq = s.fit_transform(freq)  # 480，16384
            yes_data[i] = np.append(time.reshape(itr, length, 1), freq.reshape(itr, length, 1), axis=2)

        num_class, itr, length, channel = no_data.shape
        for i in range(num_class):
            time = no_data[i, :, :, 0]
            s = MinMaxScaler()
            time = s.fit_transform(time)
            freq = no_data[i, :, :, 1]
            s = MinMaxScaler()
            freq = s.fit_transform(freq)  # 480，16384
            no_data[i] = np.append(time.reshape(itr, length, 1), freq.reshape(itr, length, 1), axis=2)
        # end

        yes_data = yes_data.reshape(-1, self.args.seq_len, self.args.seq_ch)  # normal
        no_data = no_data.reshape(-1, self.args.seq_len, self.args.seq_ch)  # abnormal

        normal_itr = len(yes_data)
        abnormal_itr = len(no_data)
        yes_label = torch.zeros(normal_itr, dtype=torch.float32)  # 异常为1 正常为0
        no_label = torch.ones(abnormal_itr, dtype=torch.float32)
        trainAndValid_data = yes_data[:int(normal_itr * self.ratio)]  # actually train and valid ,len = 2700*0.8 => 2160
        trainAndValid_label = yes_label[:int(normal_itr * self.ratio)]
        test_data = np.append(yes_data[int(normal_itr * self.ratio):], no_data,
                              axis=0)  # only test ,len = 2700-2160 + 1080 = 1620
        test_label = np.append(yes_label[int(normal_itr * self.ratio):], no_label, axis=0)
        trainAndValid_data, trainAndValid_label = shuffle(trainAndValid_data, trainAndValid_label)
        test_data, test_label = shuffle(test_data, test_label)
        data = np.append(trainAndValid_data, test_data, axis=0)
        label = np.append(trainAndValid_label, test_label, axis=0)
        data_iteration = len(data)
        border1s = [0, int(normal_itr * self.ratio * 0.8), int(normal_itr * self.ratio)]
        border2s = [int(normal_itr * self.ratio * 0.8), int(normal_itr * self.ratio), data_iteration]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = data[border1:border2]
        self.label = label[border1:border2]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

class Classifier_Dataset():
    def __init__(self, args, flag='valid',quickLoad=True,ratio=0.6):
        assert flag in ['train', 'valid', 'test']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.args = args
        self.quickLoad = quickLoad
        self.ratio = ratio
        self.set_type = type_map[flag]
        self.__load__()
        self.__process__()
    def __load__(self):
        if os.path.exists(os.path.join(self.args.path, "abnormal-data.npy")):
            self.AbnormalDataset = MakeDataset(self.args, pattern='abnormal', quickLoad=self.quickLoad)
        else:
            print("abnormal-data.npy not found!")
            self.AbnormalDataset = MakeDataset(self.args, pattern='abnormal', quickLoad=False)
    def __process__(self):
        data = self.AbnormalDataset.get_data()  # abnormal # [9,30*fileNum,16384,2] = [9,480,16384,2]
        num_classes,itr,length,channel = data.shape
        for i in range(num_classes):
            time = data[i,:,:,0]
            s = MinMaxScaler()
            time = s.fit_transform(time)
            freq = data[i,:,:,1]
            s = MinMaxScaler()
            freq = s.fit_transform(freq) #480，16384
            data[i] = np.append(time.reshape(itr, length, 1), freq.reshape(itr, length, 1), axis=2)
        label = np.array([i for i in range(num_classes) for _ in range(itr)]).reshape(-1)
        total = num_classes * itr
        data = data.reshape(-1, self.args.seq_len, self.args.seq_ch)
        data,label = shuffle(data,label)
        border1s = [0, int(total * self.ratio), int(total * (1+self.ratio) // 2)]
        border2s = [int(total * self.ratio), int(total * (1+self.ratio) // 2), total]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data = data[border1:border2]
        self.label = label[border1:border2]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.label)


