import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, accuracy_score

class machineAl():
    def __init__(self):
        pass

    def RandomForestClassifier(self,X_train, X_test, y_train, y_test):
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=80, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("随机森林分类报告:")
        print(classification_report(y_test, y_pred, digits=4))
        print("准确率:", accuracy_score(y_test, y_pred))
        return rf

    def XGBoostClassifier(self,X_train, X_test, y_train, y_test):
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=9,
            eval_metric='mlogloss',
            use_label_encoder=False,
            n_estimators=250,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("XGBoost分类报告:")
        print(classification_report(y_test, y_pred, digits=4))
        print("准确率:", accuracy_score(y_test, y_pred))

    def visualize_tsne(self,features, labels=None, perplexity=50, n_components=2, random_state=42):
        """
        使用 t-SNE 对提取的特征进行降维并进行可视化。

        参数:
        features (np.ndarray): 输入的特征数组，形状为 [num_samples, num_features]。
        labels (np.ndarray): 可选参数，用于对数据进行着色的标签数组，形状为 [num_samples]。默认为 None。
        perplexity (int): t-SNE 的困惑度参数。默认为 30。
        n_components (int): 降维后的维度。默认为 2，用于二维可视化。
        random_state (int): 随机种子，确保结果可重复。默认为 42。

        返回:
        None
        """
        # x = train_data.data.reshape(-1, 16384*2)
        # y = train_data.label
        # pca = PCA(n_components=100)
        # x_pca = pca.fit_transform(x)
        # scaler = StandardScaler()
        # x = scaler.fit_transform(x_pca)

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_features = tsne.fit_transform(features)

        # 创建一个 DataFrame 方便使用 Seaborn
        df = pd.DataFrame({
            'Component 1': reduced_features[:, 0],
            'Component 2': reduced_features[:, 1],
            'Label': labels.astype(str)  # 转换为字符串以便图例显示
        })

        plt.figure(figsize=(14, 10), dpi=200)
        sns.scatterplot(
            x='Component 1',
            y='Component 2',
            hue='Label',
            palette='tab10',  # 使用离散的调色盘
            data=df,
            s=50,  # 点的大小
            alpha=0.7,  # 点的透明度
            edgecolor='k',  # 点的边框颜色
            linewidth=0.3  # 点的边框宽度
        )

        plt.title('t-SNE Visualization', fontsize=20)
        plt.xlabel('Component 1', fontsize=15)
        plt.ylabel('Component 2', fontsize=15)
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12,
                   title_fontsize=14)  # 将图例放在图外
        plt.tight_layout()
        plt.show()

def extract_features(x):
    """
    提取时间序列和频谱的特征。

    参数:
    x (np.ndarray): 输入数组，形状为 [16384, 2]。第一个通道是时域，第二个通道是频域。

    返回:
    dict: 提取的特征字典。
    """
    features = {}

    # 确保输入形状正确
    if x.shape != (16384, 2):
        raise ValueError(f"输入数组形状应为 (16384, 2)，但收到 {x.shape}")

    # 分离时域和频域数据
    time_data = x[:, 0]
    freq_data = x[:, 1]

    ### 时域特征 ###
    # 统计特征
    features['time_mean'] = np.mean(time_data)
    features['time_median'] = np.median(time_data)
    features['time_std'] = np.std(time_data)
    features['time_variance'] = np.var(time_data)
    features['time_skewness'] = stats.skew(time_data)
    features['time_kurtosis'] = stats.kurtosis(time_data)

    # 时域形状特征
    features['time_max'] = np.max(time_data)
    features['time_min'] = np.min(time_data)
    features['time_peak_to_peak'] = np.ptp(time_data)
    features['time_range'] = np.max(time_data) - np.min(time_data)

    # 零交叉率
    zero_crossings = np.where(np.diff(np.sign(time_data)))[0]
    features['time_zero_crossing_rate'] = len(zero_crossings) / len(time_data)

    # 自相关特征（以滞后1为例）
    autocorr = np.corrcoef(time_data[:-1], time_data[1:])[0, 1]
    features['time_autocorrelation_lag1'] = autocorr

    ### 频域特征 ###
    # 主频（假设频域数据已经是频谱的幅值）
    dominant_freq_index = np.argmax(freq_data)
    features['freq_dominant_frequency'] = dominant_freq_index

    # 频谱质心
    frequencies = np.arange(len(freq_data))
    spectral_centroid = np.sum(frequencies * freq_data) / np.sum(freq_data)
    features['freq_spectral_centroid'] = spectral_centroid

    # 谱熵
    psd_norm = freq_data / np.sum(freq_data)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))  # 加一个小常数避免log(0)
    features['freq_spectral_entropy'] = spectral_entropy

    # 谱峭度
    spectral_kurtosis = stats.kurtosis(freq_data)
    features['freq_spectral_kurtosis'] = spectral_kurtosis

    return features

def extract_all_features(data):
    """
    提取所有输入数据的特征，并返回特征数组。

    参数:
    data (np.ndarray): 输入数据，形状为 [num_samples, 16384, 2]。

    返回:
    np.ndarray: 提取的特征数组，形状为 [num_samples, num_features]。
    """
    all_features = []

    # 遍历每个样本，提取特征
    for sample in data:
        # 调用特征提取函数，并提取特征值
        feature_values = np.array(list(extract_features(sample).values()))
        all_features.append(feature_values)

    # 将列表转换为二维数组，形状为 [num_samples, num_features]
    return np.array(all_features)

def display(exp_c):
    train_data, _ = exp_c._get_data(flag='train')
    test_data, _ = exp_c._get_data(flag='test')
    valid_data ,_ = exp_c._get_data(flag='valid')
    machineLearn = machineAl()
    train_feature_values = extract_all_features(train_data.data)
    test_feature_values = extract_all_features(np.append(test_data.data, valid_data.data, axis=0))
    label = np.append(test_data.label, valid_data.label, axis=0)

    machineLearn.RandomForestClassifier(train_feature_values, test_feature_values, train_data.label, label)
    machineLearn.XGBoostClassifier(train_feature_values, test_feature_values, train_data.label, label)
    machineLearn.visualize_tsne(train_data.data.reshape(train_data.data.shape[0], -1), train_data.label)



