import numpy as np
import scipy.stats as stats
import pandas as pd
from data_provider.Dataset import DP64QAM_Dataset,MakeDataset,Classifier_Dataset
def data_spect(args, pattern='abnormal'):
    n = MakeDataset(args, pattern=pattern, scale=True, quickLoad=True)  # [9,30*fileNum,16384] => [9,480,16384,2]
    data = n.data
    num_batches = data.shape[1] // 30  # 480 // 30 = 16
    reshaped_data = data.reshape(9, num_batches, 30, 16384, 2)  # [9, 16, 30, 16384, 2]
    combined_data = reshaped_data.transpose(1, 0, 2, 3, 4)  # [16,9,30,16384,2]
    data = combined_data.reshape(num_batches, 9 * 30, 16384, 2)  # [16,270,16384,2]
    r = analyze_experiments(data)
    print(r)
    # r.values
def resaveData(args):
    n = MakeDataset(args, pattern='abnormal', scale=True, quickLoad=False,scaleMethod="std")  # [9,30*fileNum,16384] => [9,480,16384,2]
    m = MakeDataset(args, pattern='normal', scale=True, quickLoad=False,scaleMethod="std")  # [9,30*fileNum,16384] => [9,480,16384,2]
def analyze_experiments(data):
    """
    分析 16 次实验的数据，检查每次实验的正常性。

    参数:
    - data: numpy 数组，形状为 [16, 270, 16384, 2]

    返回:
    - results: DataFrame, 包含每次实验的统计分析结果
    """
    results = []

    for i in range(data.shape[0]):  # 遍历每次实验
        experiment_data = data[i]  # 获取当前实验数据

        # 分别提取时间通道和频谱通道
        time_data = experiment_data[:, :, 0]  # 时间通道
        spectrum_data = experiment_data[:, :, 1]  # 频谱通道

        # 统计分析
        time_mean = np.mean(time_data)
        time_std = np.std(time_data)
        spectrum_mean = np.mean(spectrum_data)
        spectrum_std = np.std(spectrum_data)

        time_skew = stats.skew(time_data.flatten())
        spectrum_skew = stats.skew(spectrum_data.flatten())
        time_kurt = stats.kurtosis(time_data.flatten())
        spectrum_kurt = stats.kurtosis(spectrum_data.flatten())

        # 将结果存入列表
        results.append({
            'experiment': i,
            'time_mean': time_mean,
            'time_std': time_std,
            'spectrum_mean': spectrum_mean,
            'spectrum_std': spectrum_std,
            'time_skew': time_skew,
            'spectrum_skew': spectrum_skew,
            'time_kurt': time_kurt,
            'spectrum_kurt': spectrum_kurt,
        })

        # # 绘制箱型图
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 2, 1)
        # plt.boxplot(time_data.flatten())
        # plt.title(f'Experiment {i} Time Data')
        #
        # plt.subplot(1, 2, 2)
        # plt.boxplot(spectrum_data.flatten())
        # plt.title(f'Experiment {i} Spectrum Data')
        #
        # plt.show()

    # 将结果转换为 pandas DataFrame
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    class args():
        pass
    args = args()
    args.path = "/home/lumin/code/pythonProject/dataset"
    args.iteration = 270
    args.seq_len = 16384
    args.seq_ch = 2
    dataset = DP64QAM_Dataset(args)
