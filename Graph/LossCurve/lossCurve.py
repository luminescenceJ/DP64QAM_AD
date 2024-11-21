import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


def loss_accuracy_f1_curve(train_loss_history, vali_loss_history, test_loss_history, f1_score_history,
                           save_path='../graph_result'):
    f1_score = np.array(f1_score_history)[:, 3]  # F1 Score
    accuracy = np.array(f1_score_history)[:, 0]  # Accuracy

    fig, ax1 = plt.subplots(figsize=(9, 6))
    linewidth = 2
    markersize = 6
    ax1.plot(train_loss_history, label='Train Loss', color='red', marker='s', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax1.plot(vali_loss_history, label='Validation Loss', color='orange', marker='x',
             linestyle='-', linewidth=linewidth, markersize=markersize)
    ax1.plot(test_loss_history, label='Anomaly Loss', color='green', marker='.',
             linestyle='-', linewidth=linewidth, markersize=markersize)
    ax1.set_xlabel("Epochs", fontsize=22)
    ax1.set_ylabel("Loss", fontsize=22)
    font_properties = font_manager.FontProperties(size=20)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax1.grid(True)
    ax1.legend(loc='center right', prop=font_properties, framealpha=0.5, bbox_to_anchor=(0.99, 0.65))
    ax2 = ax1.twinx()
    ax2.plot(accuracy, label='Accuracy', color='brown', marker='h', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax2.plot(f1_score, label='F1 Score', color='MidnightBlue', marker='v', linestyle=':',
             linewidth=linewidth, markersize=markersize)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax2.legend(loc='center right', prop=font_properties, framealpha=0.5, bbox_to_anchor=(0.99, 0.4))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "/loss_accuracy_f1_curve.pdf", format="pdf", dpi=1200, bbox_inches='tight',
                    transparent=True)
    plt.show()


'''并列loss
    def loss_accuracy_f1_curve_compare(lstm_data=None, dp64qam_data=None, save_path='D:\lumin\pythonProject\Graph\graph_result'):
        # # lstm_data = np.load("D:\lumin\lll\OFC\pic\python\\lstm_training_histories.npy", allow_pickle=True).item()
        # dp64qam_data = np.load("D:\lumin\pythonProject\Graph\LossCurve\metrics_histories.npy", allow_pickle=True).item()
        # loss_accuracy_f1_curve_compare(None, dp64qam_data)

        # lstm_data = np.load("D:\lumin\lll\OFC\pic\python\\lstm_training_histories.npy", allow_pickle=True).item()
        dp64qam_data = np.load("D:\lumin\lll\OFC\pic\python\\dp64qam_training_histories.npy", allow_pickle=True).item()

        # # LSTM 数据
        # lstm_train_loss_history = np.array(lstm_data["train_loss_history"])
        # lstm_vali_loss_history = np.array(lstm_data["vali_loss_history"])
        # lstm_f1_score_history = np.array(lstm_data["F1_score_history"])[:, 3]  # F1 Score
        # lstm_accuracy_history = np.array(lstm_data["F1_score_history"])[:, 0]  # Accuracy

        # DP64QAM 数据
        dp64qam_train_loss_history = np.array(dp64qam_data["train_loss_history"])
        dp64qam_vali_loss_history = np.array(dp64qam_data["vali_loss_history"])
        dp64qam_F1_score_history = np.array(dp64qam_data["F1_score_history"])[:, 3]  # F1 Score
        dp64qam_accuracy_history = np.array(dp64qam_data["F1_score_history"])[:, 0]  # Accuracy

        # 确定使用较长的数据长度作为对比标准
        # max_length = max(len(dp64qam_train_loss_history), len(lstm_train_loss_history))

        # 数据插值函数
        def fix_data(data, ExpLength):
            x_original = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, ExpLength)
            interpolator = interp1d(x_original, data, kind='linear', fill_value="extrapolate")
            return interpolator(x_new)

        # 设置图形
        fig, ax1 = plt.subplots(figsize=(14, 10))

        # 绘制 loss 曲线 (左侧y轴)
        linewidth = 3
        markersize = 6
        ax1.plot(dp64qam_train_loss_history, label='DFMT Train Loss', color='red', marker='s', linestyle='-',
                 linewidth=linewidth, markersize=markersize)
        # ax1.plot(fix_data(lstm_train_loss_history, max_length), label='LSTM Train Loss', color='orange', marker='x',
        #          linestyle='-', linewidth=linewidth, markersize=markersize)
        ax1.plot(dp64qam_vali_loss_history, label='DFMT Valid Loss', color='blue', marker='o', linestyle='--',
                 linewidth=linewidth, markersize=markersize)
        # ax1.plot(fix_data(lstm_vali_loss_history, max_length), label='LSTM Valid Loss', color='green', marker='^',
        #          linestyle='--', linewidth=linewidth, markersize=markersize)

        ax1.set_xlabel("Epochs", fontsize=32)
        ax1.set_ylabel("Loss", fontsize=32)
        # font_properties = font_manager.FontProperties(size=16, weight='bold')
        font_properties = font_manager.FontProperties(size=28)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontproperties(font_properties)
        ax1.grid(True)
        legend_font_properties = font_manager.FontProperties(size=28)
        legend = ax1.legend(loc='upper left', prop=legend_font_properties, framealpha=0.5)
        legend.get_frame().set_linewidth(2)  # 设置边框宽度
        legend.get_frame().set_edgecolor('black')  # 设置边框颜色
        # 创建共享x轴的第二y轴 (右侧) 用于F1 Score 和 Accuracy
        ax2 = ax1.twinx()
        ax2.plot(dp64qam_accuracy_history, label='DFMT Accuracy', color='brown', marker='h', linestyle='-',
                 linewidth=linewidth, markersize=markersize)
        # ax2.plot(dp64qam_F1_score_history, label='DFMT F1 Score', color='MidnightBlue', marker='v', linestyle=':',
        #          linewidth=linewidth, markersize=markersize)
        # ax2.plot(fix_data(lstm_accuracy_history, max_length), label='LSTM Accuracy', color='magenta', marker='p',
        #          linestyle='--', linewidth=linewidth, markersize=markersize)
        # ax2.plot(fix_data(lstm_f1_score_history, max_length), label='LSTM F1 Score', color='purple', marker='d',
        #          linestyle='-.', linewidth=linewidth, markersize=markersize)

        for label in ax2.get_yticklabels():
            label.set_fontproperties(font_properties)

        legend = ax2.legend(loc='upper right', prop=legend_font_properties, framealpha=0.5)
        legend.get_frame().set_linewidth(2)  # 设置边框宽度
        legend.get_frame().set_edgecolor('black')  # 设置边框颜色
        if save_path:
            plt.savefig(save_path+"/loss_accuracy_f1_curve_compare.png", dpi=1200, bbox_inches='tight', transparent=True)
        plt.show()

    '''

if "__main__" == __name__:
    metrics_histories = np.load("metrics_histories.npy", allow_pickle=True).item()
    train_loss_history = metrics_histories["train_loss_history"]
    vali_loss_history = metrics_histories["vali_loss_history"]
    test_loss_history = metrics_histories["test_loss_history"]
    F1_score_history = metrics_histories["F1_score_history"]
    loss_accuracy_f1_curve(train_loss_history, vali_loss_history, test_loss_history, F1_score_history)
