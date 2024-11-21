import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, cosine
from fastdtw import fastdtw

def calculate_errors_batch(normal_inputs, abnormal_inputs):
    # 确保输入为numpy数组
    normal_inputs = np.array(normal_inputs)
    abnormal_inputs = np.array(abnormal_inputs)

    # 初始化累加器
    total_euclidean = 0
    total_manhattan = 0
    total_chebyshev = 0
    total_dtw = 0
    total_cosine = 0
    num_groups = normal_inputs.shape[0]  # 获取数据组的数量

    # 遍历每组输入数据进行误差计算
    for i in range(num_groups):  # `num` 是第一维，表示有多少组数据
        normal_input = normal_inputs[i].flatten()
        abnormal_input = abnormal_inputs[i].flatten()

        # 计算误差
        euclidean_distance = euclidean(normal_input, abnormal_input)
        manhattan_distance = cityblock(normal_input, abnormal_input)
        chebyshev_distance = chebyshev(normal_input, abnormal_input)
        dtw_distance = fastdtw(normal_input, abnormal_input)[0]
        cosine_similarity = 1 - cosine(normal_input.flatten(), abnormal_input.flatten())

        # 累加各个距离
        total_euclidean += euclidean_distance
        total_manhattan += manhattan_distance
        total_chebyshev += chebyshev_distance
        total_dtw += dtw_distance
        total_cosine += cosine_similarity

        # 打印每组误差
        print(f"Group {i} - Euclidean Distance: {euclidean_distance}")
        print(f"Group {i} - Manhattan Distance: {manhattan_distance}")
        print(f"Group {i} - Chebyshev Distance: {chebyshev_distance}")
        print(f"Group {i} - Dynamic Time Warping Distance: {dtw_distance}")
        print(f"Group {i} - Cosine Similarity: {cosine_similarity}")

    # 计算平均误差
    avg_euclidean = total_euclidean / num_groups
    avg_manhattan = total_manhattan / num_groups
    avg_chebyshev = total_chebyshev / num_groups
    avg_dtw = total_dtw / num_groups
    avg_cosine = total_cosine / num_groups

    # 打印所有组的平均误差
    print(f"\nAverage Euclidean Distance: {avg_euclidean}")
    print(f"Average Manhattan Distance: {avg_manhattan}")
    print(f"Average Chebyshev Distance: {avg_chebyshev}")
    print(f"Average Dynamic Time Warping Distance: {avg_dtw}")
    print(f"Average Cosine Similarity: {avg_cosine}")

    # 返回平均误差
    return np.array([avg_euclidean, avg_manhattan, avg_chebyshev, avg_dtw, avg_cosine])

# 示例用法
normal_inputs = np.random.random((10, 16384, 2))  # 假设有10组数据，每组形状为 (16384, 2)
abnormal_inputs = np.random.random((10, 16384, 2))

average_errors = calculate_errors_batch(normal_inputs, abnormal_inputs)

