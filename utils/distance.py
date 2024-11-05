import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine
from fastdtw import fastdtw  # 需要安装 fastdtw 库


def calculate_errors(normal_input, abnormal_input):
    normal_input = np.array(normal_input)
    abnormal_input = np.array(abnormal_input)
    target_size = (1366,12)
    padding_size = target_size[0] * target_size[1] - normal_input.shape[0]
    normal_input = np.pad(normal_input, ((0, padding_size), (0, 0)), mode='constant').reshape(1366, -1)
    abnormal_input = np.pad(abnormal_input, ((0, padding_size), (0, 0)), mode='constant').reshape(1366, -1)
    # shape : (1366 ,24)
    # calculate distance

    # 计算欧氏距离 RMSE和
    euclidean_distance = euclidean(normal_input.flatten(), abnormal_input.flatten())
    # 计算曼哈顿距离 绝对值和
    manhattan_distance = cityblock(normal_input.flatten(), abnormal_input.flatten())
    # 计算切比雪夫距离 最大离散绝对差
    chebyshev_distance = chebyshev(normal_input.flatten(), abnormal_input.flatten())
    # 计算动态时间规整（DTW）距离
    dtw_distance = fastdtw(normal_input, abnormal_input)[0]
    # 计算余弦相似度
    cosine_similarity = 1 - cosine(normal_input.flatten(), abnormal_input.flatten())

    # 打印结果
    print(f"Euclidean Distance: {euclidean_distance}")
    print(f"Manhattan Distance: {manhattan_distance}")
    print(f"Chebyshev Distance: {chebyshev_distance}")
    print(f"Dynamic Time Warping Distance: {dtw_distance}")
    print(f"Cosine Similarity: {cosine_similarity}")

    errors = np.array([euclidean_distance, manhattan_distance, chebyshev_distance, cosine_similarity])
    errors_normalized = (errors - errors.min()) / (errors.max() - errors.min())
    weights = np.array([1, 1, 1, -1])  # 注意：余弦相似度需要取负值（越高越好）
    weighted_error = np.dot(weights, errors_normalized)
    print(f"Normalized Errors: {errors_normalized}")
    print(f"Weighted Total Error: {weighted_error}")

    return {
        "euclidean_distance": euclidean_distance,
        "manhattan_distance": manhattan_distance,
        "chebyshev_distance": chebyshev_distance,
        "dtw_distance": dtw_distance,
        "cosine_similarity": cosine_similarity
    }


# normal_input = np.random.rand(16384, 2)  # 10个正常样本
# abnormal_input = np.random.rand(16384, 2) + 0.5  # 10个异常样本，稍微偏移
# calculate_errors(normal_input, abnormal_input)



