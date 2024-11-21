import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def DrawWaveComparation(timeIndex, timeValue, freqIndex, freqValue, num=4000, save_path='D:\lumin\pythonProject\Graph\graph_result'):
    def tera_formatter(x, _):
        """格式化为科学计数法并加单位"""
        return f"{x / 1e12:.3f}"

    def second_formatter(x, _):
        """格式化为科学计数法并加单位"""
        return f"{x / 1e-9:.0f}"

    def power_formatter(x, _):
        """格式化为科学计数法并加单位"""
        return f"{x / 1e-6:.1f}"

    itr, length = timeIndex.shape
    fig, axs = plt.subplots(itr, 2, figsize=(12, 18))  # 2行2列的子图
    linewidth = 2.5
    count = 0
    for i in range(itr):
        value = timeValue[i][:num]
        index = timeIndex[i][:num]
        axs[i][0].plot(index, value, color='blue', linestyle='-', linewidth=linewidth)
        axs[i][0].set_title('(' + str(chr(ord('a') + count) + ')'), fontsize=26)
        count += 1
        axs[i][0].set_ylabel("Power(mW)", fontsize=26)
        axs[i][0].set_xlabel("Time(ns)", fontsize=26)
        axs[i][0].xaxis.set_major_formatter(FuncFormatter(second_formatter))
        axs[i][0].yaxis.set_major_formatter(FuncFormatter(power_formatter))

        index_ = np.linspace(0, len(freqIndex[0]) - 1, num * 4, dtype=int)
        index = freqIndex[i][index_]
        value = freqValue[i][index_]
        axs[i][1].plot(index, value, color='blue', linestyle='-', linewidth=linewidth)
        axs[i][1].set_title('(' + str(chr(ord('a') + count) + ')'), fontsize=26)
        count += 1
        axs[i][1].set_ylabel("Power(dBm)", fontsize=26)
        axs[i][1].set_xlabel("Frequence(THz)", fontsize=26)
        axs[i][1].xaxis.set_major_formatter(FuncFormatter(tera_formatter))

    for ax in axs.flatten():  # 遍历子图
        ax.tick_params(axis='both', labelsize=20)  # 更改刻度字体大小
        # for label in ax.get_xticklabels() + ax.get_yticklabels():
        #     label.set_fontweight('bold')  # 设置刻度字体加粗
        for spine in ax.spines.values():  # 遍历子图的边框（上下左右）
            spine.set_linewidth(2.75)  # 设置边框线宽为 2
        ax.yaxis.get_offset_text().set_fontsize(20)  # 修改科学计数法字体
        ax.xaxis.get_offset_text().set_fontsize(20)  # 修改科学计数法字体
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"/waveComparation.pdf",format="pdf", dpi=1200, bbox_inches='tight', transparent=True)
    plt.show()


def getData(path="D:\lumin\pythonProject\\Graph\\OriginalWaveComparation"):
    preix = ['normal_', 'pe_', 'shade_', 'turbulence_']
    timeIndex = []
    timeValue = []
    freqIndex = []
    freqValue = []
    for pattern in preix:
        time_path = os.path.join(path, pattern + 'time.txt')
        freq_path = os.path.join(path, pattern + 'freq.txt')
        time_index = []
        time_value = []
        with open(time_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line == '\n' or line == "":
                    continue
                parts = line.strip().split('\t')
                time_index.append(float(parts[0]))
                time_value.append(float(parts[1]))
        freq_index = []
        freq_value = []
        with open(freq_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line == '\n' or line == "":
                    continue
                parts = line.strip().split('\t')
                freq_index.append(float(parts[0]))
                freq_value.append(float(parts[1]))
        timeIndex.append(time_index)
        timeValue.append(time_value)
        freqIndex.append(freq_index)
        freqValue.append(freq_value)
    timeIndex = np.array(timeIndex)  # 3,16384
    timeValue = np.array(timeValue)
    freqIndex = np.array(freqIndex)
    freqValue = np.array(freqValue)
    return timeIndex, timeValue, freqIndex, freqValue


if __name__ == '__main__':
    timeIndex, timeValue, freqIndex, freqValue = getData()
    DrawWaveComparation(timeIndex, timeValue, freqIndex, freqValue)
