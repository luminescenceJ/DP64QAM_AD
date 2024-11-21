



# from data_provider.data_loader import MakeDataset
# import matplotlib.pyplot as plt
# import numpy as np
#
# class args():
#     def __init__(self):
#         self.batch_size = 8
#         self.seq_len = 16384
#         self.seq_ch = 2
#         self.iteration = 540
#         self.path = "D:\lumin\DATASET/540itr"
#
# args_ = args()
# b = MakeDataset(args_, pattern='abnormal', quickLoad=True)
# data = b.get_data()
# print(data.shape)
#
# data = data.reshape(-1,16384,2)
# data_freq = data[:,:,1]
# data_time = data[:,:,0]
