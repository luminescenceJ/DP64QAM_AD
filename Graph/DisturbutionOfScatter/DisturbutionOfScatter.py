import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import font_manager


def distribution_scatter_with_inset(energy, true, pred, threshold, s=20, alpha=0.9, figsize=(8, 6), save_path='../graph_result'):
    tp_shown = False
    tn_shown = False
    fp_shown = False
    fn_shown = False
    energy, true, pred = shuffle(energy, true, pred)
    print(f"Energy shape: {energy.shape}, True shape: {true.shape}, Pred shape: {pred.shape}")
    fig, ax = plt.subplots(figsize=figsize)

    # 主图数据绘制（18-22部分）
    for i, (e, label, prediction) in enumerate(zip(energy, true, pred)):
        if prediction == 1 and label == 1:
            if not tp_shown:
                ax.scatter(i, e, color='blue', s=s, alpha=alpha, label='True Positive (TP)', edgecolor='black')
                tp_shown = True
            else:
                ax.scatter(i, e, color='blue', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 0 and label == 0:
            if not tn_shown:
                ax.scatter(i, e, color='red', s=s, alpha=alpha, label='True Negative (TN)', edgecolor='black')
                tn_shown = True
            else:
                ax.scatter(i, e, color='red', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 1 and label == 0:
            if not fp_shown:
                ax.scatter(i, e, color='green', s=s, alpha=alpha, label='False Positive (FP)', edgecolor='black')
                fp_shown = True
            else:
                ax.scatter(i, e, color='green', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 0 and label == 1:
            if not fn_shown:
                ax.scatter(i, e, color='orange', s=s, alpha=alpha, label='False Negative (FN)', edgecolor='black')
                fn_shown = True
            else:
                ax.scatter(i, e, color='orange', s=s, alpha=alpha, edgecolor='black')
    ax.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.6f}', alpha=0.9)
    ax.set_ylim([17, 22])
    # ax.set_title('Distribution Map of Reconstructed Data', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Sample Index", fontsize=16)
    ax.set_ylabel("Energy Value", fontsize=16)
    # 添加右侧密度图
    divider = make_axes_locatable(ax)
    ax_density = divider.append_axes("right", size="15%", pad=0.1)
    sns.kdeplot(y=energy, ax=ax_density, fill=True, color='purple')
    # ax_density.get_yaxis().set_visible(False)
    # ax_density.tick_params(axis='y', labelsize=14, labelright=True, right=True, pad=10)  # 调整刻度位置
    # ax_density.spines['right'].set_position(('axes', 1.2))  # 右移密度图的坐标轴
    # 添加子图
    inset_ax = ax.inset_axes([0.1, 0.1, 0.35, 0.2])  # 子图位置 (x, y, 宽度, 高度)
    for i, (e, label, prediction) in enumerate(zip(energy, true, pred)):
        if 26 <= e <= 29:  # 仅绘制 26-28 范围的数据
            color = 'blue' if prediction == 1 and label == 1 else \
                    'red' if prediction == 0 and label == 0 else \
                    'green' if prediction == 1 and label == 0 else \
                    'orange'
            inset_ax.scatter(i, e, color=color, s=4, alpha=0.8,edgecolors='darkblue')
    inset_ax.set_ylim([26, 29])
    inset_ax.set_title('Inset (26-29)', fontsize=14)
    inset_ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/Scatter_with_Density_and_Inset.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                    transparent=True)
    plt.show()


if __name__ == "__main__":
    data = np.load('energy_data.npz')
    train_energy = data['train_energy']
    test_energy = data['test_energy']
    valid_energy = data['valid_energy']
    ratio = data['ratio']
    theta = data['theta']
    valid_label = data['valid_label']
    test_label = data['test_label']

    theta = 1.0275862068965518
    ratio = [1, 1, 1, 1]

    score = np.dot(train_energy, ratio)
    threshold = theta * np.mean(score)
    test_score = np.dot(test_energy, ratio)
    valid_score = np.dot(valid_energy, ratio)

    test_labels = np.array(test_label).reshape(-1)
    valid_labels = np.array(valid_label).reshape(-1)
    valid_pred = (valid_score > threshold).astype(int)
    valid_gt = valid_labels.astype(int)
    test_pred = (test_score > threshold).astype(int)
    test_gt = test_labels.astype(int)
    pred = np.append(valid_pred, test_pred)
    gt = np.append(valid_gt, test_gt)
    energy = np.append(valid_score, test_score)
    distribution_scatter_with_inset(energy, gt, pred, threshold)

# import matplotlib.font_manager as font_manager
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from sklearn.utils import shuffle
#
#
# def distribution_scatter(energy, true, pred, threshold, s=20, alpha=0.9, figsize=(8, 6), save_path='../graph_result'):
#     tp_shown = False
#     tn_shown = False
#     fp_shown = False
#     fn_shown = False
#     energy, true, pred = shuffle(energy, true, pred)
#     print(f"Energy shape: {energy.shape}, True shape: {true.shape}, Pred shape: {pred.shape}")
#     fig, ax = plt.subplots(figsize=figsize)
#     for i, (e, label, prediction) in enumerate(zip(energy, true, pred)):
#         if prediction == 1 and label == 1:
#             if not tp_shown:
#                 plt.scatter(i, e, color='blue', s=s, alpha=alpha, label='True Positive (TP)', edgecolor='black')
#                 tp_shown = True
#             else:
#                 plt.scatter(i, e, color='blue', s=s, alpha=alpha, edgecolor='black')
#         elif prediction == 0 and label == 0:
#             if not tn_shown:
#                 plt.scatter(i, e, color='red', s=s, alpha=alpha, label='True Negative (TN)', edgecolor='black')
#                 tn_shown = True
#             else:
#                 plt.scatter(i, e, color='red', s=s, alpha=alpha, edgecolor='black')
#         elif prediction == 1 and label == 0:
#             if not fp_shown:
#                 plt.scatter(i, e, color='green', s=s, alpha=alpha, label='False Positive (FP)', edgecolor='black')
#                 fp_shown = True
#             else:
#                 plt.scatter(i, e, color='green', s=s, alpha=alpha, edgecolor='black')
#         elif prediction == 0 and label == 1:
#             if not fn_shown:
#                 plt.scatter(i, e, color='orange', s=s, alpha=alpha, label='False Negative (FN)', edgecolor='black')
#                 fn_shown = True
#             else:
#                 plt.scatter(i, e, color='orange', s=s, alpha=alpha, edgecolor='black')
#
#     plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.6f}', alpha=0.9)
#     divider = make_axes_locatable(ax)  # 边缘分布图
#     ax_density = divider.append_axes("right", size="15%", pad=0.1)
#     sns.kdeplot(y=energy, ax=ax_density, fill=True, color='purple')
#     ax_density.get_yaxis().set_visible(False)
#     ax_density.tick_params(axis='both', which='major', labelsize=14)  # 边缘图坐标轴字体加粗
#     ax.tick_params(axis='both', which='major', labelsize=16, width=2.5)  # 主图坐标轴字体加粗
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.legend(loc='best', fontsize=14)  # 'upper right'
#     ax.set_title('Distribution Map of Reconstructed Data', fontsize=16, fontweight='bold')
#     bold_font = font_manager.FontProperties(weight='bold', size=16)
#     for label in ax.get_xticklabels() + ax.get_yticklabels():
#         label.set_fontproperties(bold_font)
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path + '/Scatter_with_Density.pdf', format='pdf', dpi=1200, bbox_inches='tight',
#                     transparent=True)
#     plt.show()
#
#
# if __name__ == "__main__":
#     data = np.load('energy_data.npz')
#     train_energy = data['train_energy']
#     test_energy = data['test_energy']
#     valid_energy = data['valid_energy']
#     ratio = data['ratio']
#     theta = data['theta']
#     valid_label = data['valid_label']
#     test_label = data['test_label']
#
#     theta = 1.0275862068965518
#     ratio = [1, 1, 1, 1]
#
#     score = np.dot(train_energy, ratio)
#     threshold = theta * np.mean(score)
#     test_score = np.dot(test_energy, ratio)
#     valid_score = np.dot(valid_energy, ratio)
#
#     test_labels = np.array(test_label).reshape(-1)
#     valid_labels = np.array(valid_label).reshape(-1)
#     valid_pred = (valid_score > threshold).astype(int)  # test > threshold ,标记为异常
#     valid_gt = valid_labels.astype(int)
#     test_pred = (test_score > threshold).astype(int)  # test > threshold ,标记为异常
#     test_gt = test_labels.astype(int)
#     pred = np.append(valid_pred, test_pred)
#     gt = np.append(valid_gt, test_gt)
#     energy = np.append(valid_score, test_score)
#     distribution_scatter(energy,gt,pred,threshold)
