import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score


def evaluate_performance(energy, labels, threshold):
    preds = (energy > threshold).astype(int)  # 将预测值转为0或1（二分类）
    precision = precision_score(labels, preds)  # 计算精确率
    acc = accuracy_score(labels, preds)  # 计算准确率
    recall = recall_score(labels, preds)  # 计算召回率
    f1 = f1_score(labels, preds)  # 计算F1分数
    return precision, acc, recall, f1


data = np.load('D:\lumin\model\\1111111111\\energy_data.npz')
train_energy = data['train_energy']
test_energy = data['test_energy']
valid_energy = data['valid_energy']
valid_label = data['valid_label']
test_label = data['test_label']
ratio = data['ratio']
theta = data['theta']
label = np.append(valid_label, test_label)
print(
    f"train_energy : {np.mean(train_energy, axis=0)}, \ntest_energy : {np.mean(test_energy, axis=0)}, \nvalid_energy : {np.mean(valid_energy, axis=0)}")

theta = 1.0224489795918368
ratio = [1, 1, 1, 1]
# valid_energy = np.array(valid_energy)
# train_energy = np.array(train_energy)
# test_energy = np.array(test_energy)
# train_score = np.dot(train_energy,ratio)
# test_score = np.dot(test_energy,ratio)
# valid_score = np.dot(valid_energy,ratio)
# threshold = np.mean(train_score) * theta
# valid_score_sorted = np.sort(valid_score)
# test_score_sorted = np.sort(test_score)
# plt_score = np.append(valid_score_sorted,test_score_sorted)
# label = np.append(valid_label,test_label)
# score = np.append(valid_score,test_score)
# plt.plot(plt_score)
# plt.vlines(len(valid_score_sorted),0,max(test_score_sorted),linewidth=1)
# plt.hlines(threshold,0,len(plt_score))
# plt.show()
# valid_above_threshold = np.sum((valid_score > threshold).astype(int)) / len(valid_score) * 100
# test_above_threshold = np.sum((test_score > threshold).astype(int)) / len(test_score) * 100
# print(f"threshold is {threshold}")
# print(f"test_score is :{np.mean(test_score)}\nvalid_score is :{np.mean(valid_score)}")
# print(f'{valid_above_threshold:.1f}% in Normal over Threshold\n{test_above_threshold:.1f}% in Abnormal over Threshold')


# theta_range = [2.3]
theta_range = np.linspace(1, 1.05, 30)
# theta_range = np.logspace(np.log10(0.95), np.log10(2), 10)  # 使用对数间隔
ratio_range = [[1, 1, 1, 1]]
# ratio_range = [
#     [a, b, c, d ,e]
#     for a in [0.1, 1, 5]
#     for b in [0.1, 1, 5]
#     for c in [-0.1, -1, -5]
#     for d in [0.1, 1, 5]
#     for e in [0.1, 1, 5]
# ]

# acc:0.9889, f1:0.9912,precision:0.9944,recall:0.9879 ,theta:1.0, ratio:[1, 1, 1, 1]

precision_list, acc_list, recall_list, f1_list, theta_list = [], [], [], [], []
best_params = None
best_f1 = 0
best_acc = 0
for theta in theta_range:
    for ratio in ratio_range:
        train_score = np.dot(train_energy, ratio)
        test_score = np.dot(test_energy, ratio)
        valid_score = np.dot(valid_energy, ratio)
        threshold = np.mean(train_score) * theta
        score = np.append(valid_score, test_score)
        precision, acc, recall, f1 = evaluate_performance(score, label, threshold)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        theta_list.append(theta)
        print(
            f"acc:{acc:.4f}, f1:{f1:.4f},precision:{precision:.4f},recall:{recall:.4f} ,theta:{theta:.1f}, ratio:{ratio}")
        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_params = {"theta": theta, "ratio": ratio}
print(f"Best F1 Score: {best_f1},acc:{best_acc}")
print(f"Best Parameters: {best_params}")

plt.figure(figsize=(6, 4))

best_theta = best_params["theta"]
best_ratio = best_params["ratio"]
best_f1_value = best_f1
best_index = theta_list.index(best_theta)
offset_x = -0.001  # X轴方向的偏移量
offset_y = -0.025  # Y轴方向的偏移量
plt.text(best_theta + offset_x, f1_list[best_index] + offset_y - 0.015,
         f'F1: {best_f1_value:.3f}\nAccuracy: {best_acc:.3f}',
         color='black', fontsize=16, ha='center', va='center', zorder=2,  # 设置水平和垂直居中
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

plt.scatter(best_theta, f1_list[best_index], color='purple', marker='s', s=30, zorder=5, label="Best F1")
arrow_offset_x = 0.001  # 向右偏移
arrow_offset_y = 0  # 向上偏移
plt.annotate('',
             xy=(best_theta, f1_list[best_index]),  # 箭头的起始位置
             xytext=(best_theta + offset_x + arrow_offset_x, f1_list[best_index] + offset_y + arrow_offset_y),
             # 箭头的终点（文本位置）
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.5, zorder=3))

plt.scatter(theta_list, acc_list, s=15, label="Accuracy", color="blue", alpha=0.6)
plt.plot(theta_list, acc_list, color="blue", alpha=0.6)  # 将散点图连接
plt.scatter(theta_list, recall_list, s=15, label="Recall", color="red", alpha=0.6)
plt.plot(theta_list, recall_list, color="red", alpha=0.6)  # 将散点图连接
# plt.scatter(theta_list, precision_list, s=15, label="Precision", color="cornflowerblue", alpha=0.6)  # 预设颜色
# plt.plot(theta_list, precision_list, color="cornflowerblue", alpha=0.6)  # 预设颜色
plt.scatter(theta_list, f1_list, s=15, label="F1 Score", color="green", alpha=0.6)
plt.plot(theta_list, f1_list, color="green", alpha=0.6)  # 将散点图连接

from matplotlib import font_manager

font = font_manager.FontProperties(size=13)
plt.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=1)
plt.legend(prop=font)
plt.tick_params(axis='both', which='major', labelsize=13)  # 设置坐标轴刻度的字体大小
plt.xlabel(r"$\theta$", fontsize=16)
plt.ylabel("Value", fontsize=16)
plt.tight_layout()
plt.savefig('D:\lumin\pythonProject\Graph\graph_result\\thetaSweep.pdf', format="pdf", dpi=1200, bbox_inches='tight',
            transparent=True)
plt.show()

# ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(label, score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Classifier")
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('ROC Curve', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=13)  # 设置坐标轴刻度的字体大小
font = font_manager.FontProperties(size=14)
plt.legend(loc='lower right', prop=font)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('D:\lumin\pythonProject\Graph\graph_result\\roc_curve.pdf', format='pdf', dpi=1200, bbox_inches='tight',
            transparent=True)
plt.show()
