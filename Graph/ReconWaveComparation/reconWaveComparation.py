import matplotlib.pyplot as plt
import numpy as np


def visualize3D(time_ori, time_pred, freq_ori, freq_pred, label, num, itr):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(projection='3d')
    start = 400
    xs = np.linspace(0, num - 1, (num - start))
    # # 画出原始、预测和差异曲
    # ax.plot(xs, [-1] * (num-start), freq_ori[itr, start:num], color='orange', label='original')
    # ax.plot(xs, [-0.4] * (num-start), freq_pred[itr, start:num], color='blue', label='predict')
    # ax.plot(xs, [0] * (num-start), abs(freq_ori[itr, start:num] - freq_pred[itr, start:num]), color='red', label='difference')
    ax.plot(xs, [-1] * (num - start), time_ori[itr, start:num], color='orange', label='original')
    ax.plot(xs, [-0.45] * (num - start), time_pred[itr, start:num], color='blue', label='predict')
    ax.plot(xs, [0] * (num - start), abs(time_ori[itr, start:num] - time_pred[itr, start:num]), color='red',
            label='difference')
    ax.view_init(elev=25, azim=120)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴刻度的字体大小为14
    ax.tick_params(axis='z', labelsize=16)  # 设置x轴刻度的字体大小为14
    ax.set_xlim(np.min(xs), np.max(xs))  # 设置x轴范围
    ax.set_ylim(-1, 0)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 0.5])  # 设置坐标轴的比例
    ax.text(-50, -0.75, z=0, s='Original', ha='left', va='center', fontsize=17, rotation=90)
    ax.text(-50, -0.3, z=0, s='Predict', ha='left', va='center', fontsize=17, rotation=90)
    ax.text(-50, 0.1, z=0, s='Difference', ha='left', va='center', fontsize=17, rotation=90)
    time_mse = np.mean((time_ori - time_pred) ** 2)
    freq_mse = np.mean((freq_ori - freq_pred) ** 2)
    file_name = f"./reconWave_label{label[0]}time{time_mse}_freq{freq_mse}.pdf"  # 或者使用其他逻辑来确定文件名
    plt.tight_layout()
    plt.savefig(file_name, format="pdf", dpi=1200, transparent=True)
    plt.show()


def getData(filename='data_output_label.npz'):
    file = np.load(filename)
    output = file['output']
    data = file['data']
    label = file['label']
    time_ori = data[:, :, 0]
    freq_ori = data[:, :, 1]
    time_pred = output[:, :, 0]
    freq_pred = output[:, :, 1]
    return (time_ori, time_pred, freq_ori, freq_pred, label)


num = 600
itr = 2
(time_ori, time_pred, freq_ori, freq_pred, label) = getData("./test_data_output_label.npz")
visualize3D(time_ori, time_pred, freq_ori, freq_pred, label, num=num, itr=itr)
(time_ori, time_pred, freq_ori, freq_pred, label) = getData("./valid_data_output_label.npz")
visualize3D(time_ori, time_pred, freq_ori, freq_pred, label, num=num, itr=itr)

''' 并列图
def visualize3D(ax, time_ori, time_pred, freq_ori, freq_pred, label, num, itr):
    start = 400
    xs = np.linspace(0, num - 1, (num - start))

    # 画出原始、预测和差异曲线
    ax.plot(xs, [-1] * (num - start), time_ori[itr, start:num], color='orange', label='original')
    ax.plot(xs, [-0.45] * (num - start), time_pred[itr, start:num], color='blue', label='predict')
    ax.plot(xs, [0] * (num - start), abs(time_ori[itr, start:num] - time_pred[itr, start:num]), color='red',
            label='difference')

    ax.view_init(elev=25, azim=120)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=16)  # 设置x轴刻度的字体大小为14
    ax.tick_params(axis='z', labelsize=16)  # 设置x轴刻度的字体大小为14
    ax.set_xlim(np.min(xs), np.max(xs))  # 设置x轴范围
    ax.set_ylim(-1, 0)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 0.7])  # 设置坐标轴的比例

    ax.text(-50, -0.75, 0, s='Original', ha='left', va='center', fontsize=14, rotation=90)
    ax.text(-50, -0.3, 0, s='Predict', ha='left', va='center', fontsize=14, rotation=90)
    ax.text(-50, 0.1, 0, s='Difference', ha='left', va='center', fontsize=14, rotation=90)
def getData(filename='data_output_label.npz'):
    file = np.load(filename)
    output = file['output']
    data = file['data']
    label = file['label']
    time_ori = data[:, :, 0]
    freq_ori = data[:, :, 1]
    time_pred = output[:, :, 0]
    freq_pred = output[:, :, 1]
    return (time_ori, time_pred, freq_ori, freq_pred, label)

num = 600
itr = 2
fig = plt.figure(figsize=(10, 4))
(time_ori, time_pred, freq_ori, freq_pred, label) = getData("./test_data_output_label.npz")
ax1 = fig.add_subplot(121, projection='3d')
visualize3D(ax1, time_ori, time_pred, freq_ori, freq_pred, label, num=num, itr=itr)
(time_ori, time_pred, freq_ori, freq_pred, label) = getData("./valid_data_output_label.npz")
ax2 = fig.add_subplot(122, projection='3d')
visualize3D(ax2, time_ori, time_pred, freq_ori, freq_pred, label, num=num, itr=itr)
plt.tight_layout()
plt.savefig("./reconWave_combined.pdf", format="pdf", dpi=1200, transparent=True)
plt.show()
'''
