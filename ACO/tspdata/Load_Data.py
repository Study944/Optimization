"""加载TSP数据集"""

import glob
import os
import numpy as np
import tsplib95

# 测试的 6 个标准数据集
RECOMMENDED_DATASETS = {
    "ulysses16": 6859,
    "att48": 10628,
    "berlin52": 7542,
    "kroA100": 21282,
    "tsp225": 3916,
    "pcb442": 50778,
}


def load_all_tsp_data():
    """扫描.tsp 文件，统一标准化为距离矩阵"""
    # 1.搜索文件夹下所有的 .tsp 纯文本文件
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 动态拼接出真正的 dataset 绝对路径
    search_path = os.path.join(current_dir, "dataset", "*.tsp")
    tsp_files = glob.glob(search_path)
    dataset_registry = {}

    for file_path in tsp_files:
        # 2.从路径中提取出数据集名字 (例如 'data/att48.tsp' -> 'att48')
        base_name = os.path.basename(file_path)
        dataset_name = os.path.splitext(base_name)[0]

        try:
            # 3.使用 tsplib95.load 读取本地解压好的纯文本文件
            problem = tsplib95.load(file_path)

            # 4.标准化提取城市节点
            nodes = list(problem.get_nodes())
            num_cities = len(nodes)

            # 5.构建标准的 N x N 距离矩阵
            dist_matrix = np.zeros((num_cities, num_cities))
            for i in range(num_cities):
                for j in range(num_cities):
                    if i == j:
                        dist_matrix[i][j] = float("inf")  # 自身到自身设为无穷大
                    else:
                        # tsplib95 会自动在后台根据 GEO 或 EUC_2D 算好正确的距离
                        dist_matrix[i][j] = problem.get_weight(
                            nodes[i], nodes[j]
                        )

            # 统一打包存入字典
            dataset_registry[dataset_name] = {
                "matrix": dist_matrix,
                "size": num_cities,
                "optimal": RECOMMENDED_DATASETS[dataset_name],
            }
            print(
                f"成功加载本地文本: {dataset_name:<10} | 城市规模: {num_cities:<3} | 官方最优解: {RECOMMENDED_DATASETS[dataset_name]}"
            )

        except Exception as e:
            print(f"❌ 加载 {base_name} 失败，原因: {e}")

    return dataset_registry