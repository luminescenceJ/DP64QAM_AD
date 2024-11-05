import numpy as np
import torch
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.font_manager import FontProperties

from sklearn.utils import shuffle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine
from fastdtw import fastdtw  # 需要安装 fastdtw 库
plt.switch_backend('agg')

# Training
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=1e-3):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
def adjust_learning_rate(optimizer, epoch, args):
    lr_adjust = {epoch: args.learning_rate * (0.2 ** ((epoch - 1) // 1))}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type7':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type6':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type9':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
# Drawing
def distribution_scatter(energy, true, pred, threshold, s=20, alpha=0.9, figsize=(8, 6), save_path='./graph_result'):
    tp_shown = False
    tn_shown = False
    fp_shown = False
    fn_shown = False
    energy, true, pred = shuffle(energy, true, pred)
    # print(f"Energy shape: {energy.shape}, True shape: {true.shape}, Pred shape: {pred.shape}")
    fig, ax = plt.subplots(figsize=figsize)
    for i, (e, label, prediction) in enumerate(zip(energy, true, pred)):
        if prediction == 1 and label == 1:
            if not tp_shown:
                plt.scatter(i, e, color='blue', s=s, alpha=alpha, label='True Positive (TP)', edgecolor='black')
                tp_shown = True
            else:
                plt.scatter(i, e, color='blue', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 0 and label == 0:
            if not tn_shown:
                plt.scatter(i, e, color='red', s=s, alpha=alpha, label='True Negative (TN)', edgecolor='black')
                tn_shown = True
            else:
                plt.scatter(i, e, color='red', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 1 and label == 0:
            if not fp_shown:
                plt.scatter(i, e, color='green', s=s, alpha=alpha, label='False Positive (FP)', edgecolor='black')
                fp_shown = True
            else:
                plt.scatter(i, e, color='green', s=s, alpha=alpha, edgecolor='black')
        elif prediction == 0 and label == 1:
            if not fn_shown:
                plt.scatter(i, e, color='orange', s=s, alpha=alpha, label='False Negative (FN)', edgecolor='black')
                fn_shown = True
            else:
                plt.scatter(i, e, color='orange', s=s, alpha=alpha, edgecolor='black')

    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.6f}', alpha=0.9)
    divider = make_axes_locatable(ax) # 边缘分布图
    ax_density = divider.append_axes("right", size="15%", pad=0.1)
    sns.kdeplot(energy, ax=ax_density, vertical=True, fill=True, color='purple')
    ax_density.get_yaxis().set_visible(False)
    ax_density.tick_params(axis='both', which='major', labelsize=14)  # 边缘图坐标轴字体加粗
    ax.tick_params(axis='both', which='major', labelsize=16, width=2.5)  # 主图坐标轴字体加粗
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', fontsize=14)  # 'upper right'
    ax.set_title('Distribution Map of Reconstructed Data', fontsize=16, fontweight='bold')
    bold_font = font_manager.FontProperties(weight='bold', size=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(bold_font)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/Scatter_with_Density.png', dpi=1200, bbox_inches='tight', transparent=True)
    plt.show()
def barChartOnAccuracy(labels, accuracy, precision, recall, f1_score, save_path='./graph_result'):
    '''
    # 示例调用
    labels = ["DFMT-Net", "XGBoost", "LSTM"]
    accuracy = [98.03, 85.12, 39.93]
    precision = [98.05, 83.29, 41.81]
    recall = [98.03, 85.28, 39.93]
    f1_score = [98.04, 84.27, 38.50]
    f1Comparation(labels, accuracy, precision, recall, f1_score, save_path="./")
    '''
    if not (len(labels) == len(accuracy) == len(precision) == len(recall) == len(f1_score)):
        raise ValueError("All input lists must have the same length.")
    barWidth = 0.2
    fig, ax = plt.subplots(figsize=(max(9, len(labels)*2), 6))  # 动态设置宽度
    r1 = np.arange(len(accuracy))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    # 画柱状图
    plt.bar(r1, accuracy, color='#2b4c4c', edgecolor='grey', width=barWidth, label='Accuracy')  # 深蓝色
    plt.bar(r2, precision, color='#4e8b61', edgecolor='grey', width=barWidth, label='Precision')  # 橙色
    plt.bar(r3, recall, color='#dfc33e', edgecolor='grey', width=barWidth, label='Recall')  # 绿色
    plt.bar(r4, f1_score, color='#935222', edgecolor='grey', width=barWidth, label='F1 score')  # 紫色
    # 在上面添加透明柱状图，应用不同的 hatch 线条
    plt.bar(r1, accuracy, color='none', width=barWidth, hatch='---')  # 红色斜线
    plt.bar(r4, f1_score, color='none', width=barWidth, hatch='xxx')  # 黄色交叉线
    # 添加数据标签
    for i in range(len(accuracy)):
        plt.text(r1[i], accuracy[i] + 1, f'{accuracy[i]:.2f}%', ha='center', fontsize=14, fontweight='bold')
        plt.text(r4[i], f1_score[i] + 1, f'{f1_score[i]:.2f}%', ha='center', fontsize=14, fontweight='bold')
    # 设置图表属性
    plt.ylabel('Evaluation metric (%)', fontweight='bold', fontsize=16)
    plt.xticks([r + barWidth*1.5 for r in range(len(accuracy))], labels, fontsize=16, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    font_properties = FontProperties(size=14, weight='bold')
    plt.legend(loc='upper right', fancybox=True, framealpha=0.5, prop=font_properties)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+'/barChartOnAccuracy.png', format="png", dpi=600, transparent=True)
    plt.show()
def loss_accuracy_f1_curve_compare(lstm_data, dp64qam_data, save_path='./graph_result'):
    # lstm_data = np.load("D:\lumin\lll\OFC\pic\python\\lstm_training_histories.npy", allow_pickle=True).item()
    # dp64qam_data = np.load("D:\lumin\lll\OFC\pic\python\\dp64qam_training_histories.npy", allow_pickle=True).item()

    # LSTM 数据
    lstm_train_loss_history = np.array(lstm_data["train_loss_history"])
    lstm_vali_loss_history = np.array(lstm_data["vali_loss_history"])
    lstm_f1_score_history = np.array(lstm_data["F1_score_history"])[:, 3]  # F1 Score
    lstm_accuracy_history = np.array(lstm_data["F1_score_history"])[:, 0]  # Accuracy

    # DP64QAM 数据
    dp64qam_train_loss_history = np.array(dp64qam_data["train_loss_history"])
    dp64qam_vali_loss_history = np.array(dp64qam_data["vali_loss_history"])
    dp64qam_F1_score_history = np.array(dp64qam_data["F1_score_history"])[:, 3]  # F1 Score
    dp64qam_accuracy_history = np.array(dp64qam_data["F1_score_history"])[:, 0]  # Accuracy

    # 确定使用较长的数据长度作为对比标准
    max_length = max(len(dp64qam_train_loss_history), len(lstm_train_loss_history))

    # 数据插值函数
    def fix_data(data, ExpLength):
        x_original = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, ExpLength)
        interpolator = interp1d(x_original, data, kind='linear', fill_value="extrapolate")
        return interpolator(x_new)

    # 设置图形
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制 loss 曲线 (左侧y轴)
    linewidth = 2
    markersize = 6
    ax1.plot(fix_data(dp64qam_train_loss_history, max_length), label='DFMT Train Loss', color='red', marker='s', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax1.plot(fix_data(lstm_train_loss_history, max_length), label='LSTM Train Loss', color='orange', marker='x',
             linestyle='-', linewidth=linewidth, markersize=markersize)
    ax1.plot(fix_data(dp64qam_vali_loss_history, max_length), label='DFMT Valid Loss', color='blue', marker='o', linestyle='--',
             linewidth=linewidth, markersize=markersize)
    ax1.plot(fix_data(lstm_vali_loss_history, max_length), label='LSTM Valid Loss', color='green', marker='^',
             linestyle='--', linewidth=linewidth, markersize=markersize)

    ax1.set_xlabel("Epochs", fontsize=20, weight='bold')
    ax1.set_ylabel("Loss", fontsize=20, weight='bold')
    font_properties = font_manager.FontProperties(size=16, weight='bold')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax1.grid(True)
    legend_font_properties = font_manager.FontProperties(size=16, weight='bold')
    ax1.legend(loc='upper left', prop=legend_font_properties, framealpha=0.7)

    # 创建共享x轴的第二y轴 (右侧) 用于F1 Score 和 Accuracy
    ax2 = ax1.twinx()
    ax2.plot(dp64qam_accuracy_history, label='DFMT Accuracy', color='brown', marker='h', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax2.plot(dp64qam_F1_score_history, label='DFMT F1 Score', color='MidnightBlue', marker='v', linestyle=':',
             linewidth=linewidth, markersize=markersize)
    ax2.plot(fix_data(lstm_accuracy_history, max_length), label='LSTM Accuracy', color='magenta', marker='p',
             linestyle='--', linewidth=linewidth, markersize=markersize)
    ax2.plot(fix_data(lstm_f1_score_history, max_length), label='LSTM F1 Score', color='purple', marker='d',
             linestyle='-.', linewidth=linewidth, markersize=markersize)

    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)

    ax2.legend(loc='upper right', prop=legend_font_properties, framealpha=0.5)
    if save_path:
        plt.savefig(save_path+"/loss_accuracy_f1_curve_compare.png", dpi=1200, bbox_inches='tight', transparent=True)
    plt.show()
def loss_accuracy_f1_curve(train_loss_history,vali_loss_history,test_loss_history,f1_score_history,save_path='./graph_result'):


    f1_score = np.array(f1_score_history)[:, 3]  # F1 Score
    accuracy = np.array(f1_score_history)[:, 0]  # Accuracy


    fig, ax1 = plt.subplots(figsize=(8, 6))
    linewidth = 2
    markersize = 6
    ax1.plot(train_loss_history, label='Train Loss', color='red', marker='s', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax1.plot(vali_loss_history, label='Validation Loss', color='orange', marker='x',
             linestyle='-', linewidth=linewidth, markersize=markersize)
    ax1.plot(test_loss_history, label='Test Loss', color='green', marker='.',
             linestyle='-', linewidth=linewidth, markersize=markersize)
    ax1.set_xlabel("Epochs", fontsize=20, weight='bold')
    ax1.set_ylabel("Loss", fontsize=20, weight='bold')
    font_properties = font_manager.FontProperties(size=16, weight='bold')
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax1.grid(True)
    legend_font_properties = font_manager.FontProperties(size=16, weight='bold')
    ax1.legend(loc='upper left', prop=legend_font_properties, framealpha=0.7)
    # 创建共享x轴的第二y轴 (右侧) 用于F1 Score 和 Accuracy
    ax2 = ax1.twinx()
    ax2.plot(accuracy, label='Accuracy', color='brown', marker='h', linestyle='-',
             linewidth=linewidth, markersize=markersize)
    ax2.plot(f1_score, label='F1 Score', color='MidnightBlue', marker='v', linestyle=':',
             linewidth=linewidth, markersize=markersize)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_properties)
    ax2.legend(loc='upper right', prop=legend_font_properties, framealpha=0.5)
    if save_path:
        plt.savefig(save_path+"/loss_accuracy_f1_curve.png", dpi=1200, bbox_inches='tight', transparent=True)
    plt.show()

def confusion_maxtrix_graph(y_true, y_pred, save_path='./graph_result'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # 计算准确率、精确率、召回率、F1分数
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    class_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i, label in enumerate(class_labels):
        ax.text(-0.5, i + 0.5, label, ha='center', va='center', fontsize=20, fontweight='bold')  # Y轴旁边的标签
        ax.text(i + 0.5, len(class_labels) + 0.5, label, ha='center', va='center', fontsize=20,fontweight='bold')  # X轴下方的标签
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', annot_kws={"size": 20, "weight": "bold"},
                    cbar=False,
                    linewidths=0.5, linecolor='gray', xticklabels=[f"{i}" for i in range(9)],
                    yticklabels=[f"{i}" for i in range(9)])
    plt.title(f"Accuracy: {accuracy * 100:.2f}%", fontsize=24, fontweight='bold')
    # plt.xlabel("Predicted Label", fontsize=14)
    # plt.ylabel("True Label", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"/confusion_matrix.png", dpi=1200, bbox_inches='tight', transparent=True)
    plt.show()
def channel_Display(num,save_path='./graph_result'):
    def generate_random_ellipse(ax, center, width, height, angle):
        """生成一个椭圆并绘制在给定的轴上，使用深颜色"""
        ellipse = patches.Ellipse(center, width, height, angle=angle, color='grey', alpha=0.5)  # 修改颜色和透明度
        ax.add_patch(ellipse)
    def is_overlap(center, width, height, existing_ellipses):
        """检查新椭圆是否与已有椭圆重叠"""
        for (c, w, h) in existing_ellipses:
            dist = np.linalg.norm(np.array(center) - np.array(c))
            if dist < (width / 2 + w / 2) and dist < (height / 2 + h / 2):
                return True
        return False
    def generate_size():
        """生成椭圆大小：有更多很大的和很小的椭圆"""
        choice = random.random()
        if choice < 0.5:  # 20% 生成非常小的椭圆
            width = random.uniform(0.3, 1.2)
            height = random.uniform(0.6, 1.5)
        else:
            width = random.uniform(0.6, 1.4)
            height = random.uniform(0.3, 1.2)

        return width, height
    def create_fso_channel(num_ellipses):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        # ax.set_title('FSO Channel with Random Obstacles (Varied Sizes)')
        # 移除坐标轴
        ax.axis('off')  # 关闭横纵坐标轴显示
        existing_ellipses = []
        for _ in range(num_ellipses):
            width, height = generate_size()  # 调用生成椭圆大小函数
            center = (random.uniform(0, 10), random.uniform(0, 6))
            angle = random.uniform(0, 360)
            if not is_overlap(center, width, height, existing_ellipses):
                generate_random_ellipse(ax, center, width, height, angle)
                existing_ellipses.append((center, width, height))
        plt.savefig(save_path+"channelCondition.png", format="png", dpi=600, bbox_inches='tight', pad_inches=0,transparent=True)
        plt.show()
    create_fso_channel(num)
def plot_energy_distribution(train_energy, test_energy, threshold, figsize=(10, 6),save_path='./graph_result'):
    """
    绘制训练集和测试集的能量分布图，并标记超过阈值的比例

    参数:
    train_energy (np.array): 训练集的能量值
    test_energy (np.array): 测试集的能量值
    threshold (float): 阈值
    figsize (tuple): 图像大小 (可选参数, 默认为 (10, 6))

    # 示例调用
    # train_energy = np.random.normal(0, 1, 1000)
    # test_energy = np.random.normal(0, 1, 500)
    # threshold = 1.5
    # plot_energy_distribution(train_energy, test_energy, threshold)

    """

    # 计算超过阈值的百分比
    train_above_threshold = np.sum(train_energy > threshold) / len(train_energy) * 100
    test_above_threshold = np.sum(test_energy > threshold) / len(test_energy) * 100

    plt.figure(figsize=figsize)
    energy_ = np.concatenate([sorted(train_energy), sorted(test_energy)], axis=0)
    plt.plot(energy_, label='Energy', color='blue')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.7f}')
    plt.axvline(len(train_energy), color='black', linestyle='-')
    y_min, y_max = plt.ylim()
    x_min, x_max = plt.xlim()
    plt.text(x_max * 0.5, y_max * 0.5,
             f'{train_above_threshold:.1f}% in Normal over Threshold\n{test_above_threshold:.1f}% in Abnormal over Threshold',
             horizontalalignment='center', color='red')
    plt.xlabel('index')
    plt.ylabel('Reconstruct MSE')
    plt.title('Energy Distribution')
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path + "/energy_distribution.png", dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
def score_my(y_true, y_pred, digits=4,detail=False):
    '''
    :param y_true: np.array([0, 1, 2, 2, 0, 1, 1, 0, 2, 0])
    :param y_pred: np.array([0, 2, 2, 2, 0, 0, 1, 0, 1, 2])
    :param usage: classification_report_multiclass(y_true, y_pred)
    :print:
        Type           Precision   Recall   F1-Score    Support
        0              0.75        0.75     0.75        4
        1              0.5         0.3333   0.4         3
        2              0.5         0.6667   0.5714      3
        macro avg      0.5833      0.5833   0.5738      10
        weighted avg   0.6000      0.6000   0.5914      10
        accuracy       0.6000
    :return: accuracy, macro avg precision , macro avg recall, macro avg f1-score
    '''
    if detail:
        print("y_true:", y_true.shape, ",mean:", np.mean(y_true))
        print("y_pred:", y_pred.shape, ",mean:", np.mean(y_pred))

    labels = np.unique(np.concatenate((y_true, y_pred)))
    report = {label: {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0} for label in labels}

    for label in labels:
        TP = sum((y_true == label) & (y_pred == label))  # True Positives
        FP = sum((y_true != label) & (y_pred == label))  # False Positives
        FN = sum((y_true == label) & (y_pred != label))  # False Negatives
        support = sum(y_true == label)  # 支持度（support）

        # 计算 precision, recall 和 f1-score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        report[label]['precision'] = round(precision, digits)
        report[label]['recall'] = round(recall, digits)
        report[label]['f1-score'] = round(f1, digits)
        report[label]['support'] = support
    macro_precision = np.mean([report[label]['precision'] for label in labels])
    macro_recall = np.mean([report[label]['recall'] for label in labels])
    macro_f1 = np.mean([report[label]['f1-score'] for label in labels])
    total_support = sum(report[label]['support'] for label in labels)
    weighted_precision = np.sum([report[label]['precision'] * report[label]['support'] for label in labels]) / total_support
    weighted_recall = np.sum([report[label]['recall'] * report[label]['support'] for label in labels]) / total_support
    weighted_f1 = np.sum([report[label]['f1-score'] * report[label]['support'] for label in labels]) / total_support

    correct_predictions = sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    if detail:
        print(f"{'Type':<15}{'Precision':<12}{'Recall':<9}{'F1-Score':<12}{'Support':<9}")
        for label in labels:
            print(f"{label:<15}{report[label]['precision']:<12}{report[label]['recall']:<9}{report[label]['f1-score']:<12}{report[label]['support']:<9}")
        print(f"{'macro avg':<15}"
              f"{macro_precision:<12.{digits}f}"  # 设置小数位数
              f"{macro_recall:<9.{digits}f}"
              f"{macro_f1:<12.{digits}f}"
              f"{total_support:<9}")
        print(f"{'weighted avg':<15}"
              f"{weighted_precision:<12.{digits}f}"
              f"{weighted_recall:<9.{digits}f}"
              f"{weighted_f1:<12.{digits}f}"
              f"{total_support:<9}")
        print(f"{'accuracy':<15}"
              f"{accuracy:<12.{digits}f}")
    else:
        print(f"accuracy:{accuracy:.4f},precision:{macro_precision:.4f},recall:{macro_recall:.4f},f1-score:{macro_f1:.4f}")
    return accuracy,weighted_precision,weighted_recall,weighted_f1

def visualize(dataset,model,batch_size=16,num=16384,itr=0,save_path='./graph_result',save_name='valid'):

    (data, label) = dataset[0:batch_size]
    output = model(torch.tensor(data).float()).cpu().detach().numpy()

    time_ori = data[:, :, 0]
    freq_ori = data[:, :, 1]
    time_pred = output[:, :, 0]
    freq_pred = output[:, :, 1]

    print(f"label is {label},itr is {itr}")
    # mse
    time_mse = np.mean((time_ori - time_pred) ** 2)
    freq_mse = np.mean((freq_ori - freq_pred) ** 2)
    print(f"time mse: {time_mse:.7f}, freq mse: {freq_mse:.7f}, total mse: {time_mse + freq_mse:.7f}")

    # cos similarity
    time_cos_sim = 1 - cosine(time_ori[itr], time_pred[itr])
    freq_cos_sim = 1 - cosine(freq_ori[itr], freq_pred[itr])
    print(f"time cos_sim :{time_cos_sim:.5f},freq cos_sim:{freq_cos_sim:.5f}")



    # 创建一个图形对象
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # 2行1列的子图
    # 绘制时域数据
    axs[0].set_title("Time")
    axs[0].plot(time_ori[itr, :num], 'blue', label='real')
    axs[0].plot(time_pred[itr, :num], 'red', label='pred')
    axs[0].plot(abs(time_ori[itr, :num] - time_pred[itr, :num]), 'yellow', label='diff')
    axs[0].legend()
    # 绘制频域数据
    axs[1].set_title("Freq")
    axs[1].plot(freq_ori[itr, :num], 'blue', label='real')
    axs[1].plot(freq_pred[itr, :num], 'red', label='pred')
    axs[1].plot(abs(freq_ori[itr, :num] - freq_pred[itr, :num]), 'yellow', label='diff')
    axs[1].legend()
    if save_path:
        plt.savefig(save_path + f"/visualize_{save_name}.png", dpi=600, bbox_inches='tight', transparent=True)
    plt.tight_layout()
    plt.show()



def calculate_errors(normal_input, abnormal_input):
    # 确保输入为numpy数组
    normal_input = np.array(normal_input).flatten()
    abnormal_input = np.array(abnormal_input).flatten()

    print(normal_input.shape)

    # 计算误差
    euclidean_distance = euclidean(normal_input, abnormal_input)
    manhattan_distance = cityblock(normal_input, abnormal_input)
    chebyshev_distance = chebyshev(normal_input, abnormal_input)
    dtw_distance = fastdtw(normal_input, abnormal_input)[0]
    cosine_similarity = 1 - cosine(normal_input.flatten(), abnormal_input.flatten())

    print(f"Euclidean Distance: {euclidean_distance}")
    print(f"Manhattan Distance: {manhattan_distance}")
    print(f"Chebyshev Distance: {chebyshev_distance}")
    print(f"Dynamic Time Warping Distance: {dtw_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    # 将误差放入数组中
    errors = np.array([euclidean_distance, manhattan_distance, chebyshev_distance, dtw_distance, cosine_similarity])

    # 线性加权示例（可以根据需要调整权重）
    weights = np.array([1, 1, 1, 1, -1e-1])  # 注意：余弦相似度需要取负值（越高越好）
    weighted_error = np.dot(weights, errors)

    # 打印结果
    print(f"Errors: {errors}")
    print(f"Weighted Total Error: {weighted_error}")
    return np.array([euclidean_distance, manhattan_distance, chebyshev_distance,dtw_distance,cosine_similarity])



# dataset, _ = exp._get_data(flag='test')  # test中包含异常 不需要可视乎验证
# exp.model.eval()
# exp.model.to("cpu")
# (datax, label) = dataset[0:exp.args.batch_size]
# output = exp.model(torch.tensor(datax).float()).cpu().detach().numpy()
# # # Check the shapes
# print("datax shape:", datax.shape)
# print("output shape:", output.shape)
# time_ori = datax[:, :, 0]  # bs,16384,2
# freq_ori = datax[:, :, 1]
# time_pred = output[:, :, 0]
# freq_pred = output[:, :, 1]
# visualize(time_ori, time_pred, freq_ori, freq_pred, label, num=16384, itr=7)
#
# dataset, _ = exp._get_data(flag='valid')  # test中包含异常 不需要可视乎验证
# exp.model.eval()
# exp.model.to("cpu")
# (datax, label) = dataset[0:exp.args.batch_size]
# output = exp.model(torch.tensor(datax).float()).cpu().detach().numpy()
# # # Check the shapes
# print("datax shape:", datax.shape)
# print("output shape:", output.shape)
# time_ori = datax[:, :, 0]  # bs,16384,2
# freq_ori = datax[:, :, 1]
# time_pred = output[:, :, 0]
# freq_pred = output[:, :, 1]


# visualize(time_ori, time_pred, freq_ori, freq_pred, label, num=16384, itr=7)
