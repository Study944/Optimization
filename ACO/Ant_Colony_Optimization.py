"""蚁群算法（Ant Colony Optimization）
    摘要：蚁群算法通过模拟学习蚁群觅食的行为实现，常用于解决非连续问题（如TSP）
    引言：单个蚂蚁往往很难进行觅食，而蚁群却能很精准地觅食，其核心在于蚂蚁觅食过程会留下信息素，信息素越浓就代表路径越好
    核心原理：
        τ_ij ：代表从i到j的信息素浓度，一般浓度越高其重要性也越高，社会经验
        η_ij ：能见度/启发式信息，代表及时的利益，通常设置为 1/d_ij （ij路径长度的倒数）
        τ_ik ：k是i尚未访问过的节点，通常使用一个 tabu_list 记录
        α,β ：权重系数，α是τ的权重，其越大代表蚁群经验越重要，β是η的权重，其越大代表及时经验越重要
        概率公式：P_ij = [τ_ij]^α * [η_ij]^β / Σ(allowed) [τ_ik]^α * [η_ik]^β
        信息素更新：信息素会随着轮次的迭代更新，会进行信息素蒸发和新增，τ_ij（t+1） = （1-p）τ_ij + Δτ_ij
    算法流程：
        参数设置：α∈[1,2]，β∈[2,5]，τ的初始设置一个微小的正常数C/计算一个路径长度然后m/L_nn
        更新操作：τ_ij（t+1） = （1-p）τ_ij + Δτ_ij ，Δτ_ij是新释放的信息素，在 TSP 问题计算方式：
                Δτ_ij = Σ Δτ_ij^k ，Δτ_ij^k = Q/L_k ，L_k是蚂蚁k的总路径长度
"""

import numpy as np


class ACO:
    """蚁群算法 ACO """

    def __init__(self, alpha=1.5, beta=3.5, p=0.2, ger=50, Q=100):
        """初始化"""
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.ger = ger
        self.Q = Q

    def naive_nearest_neighbor(self, dist_matrix):
        """贪心计算一个基准路径长度 L_nn，用于初始化 tau = m / L_nn"""
        # 1.初始化基本参数
        start_city = 0
        num_city = dist_matrix.shape[0]
        visited = [False] * num_city
        visited[start_city] = True
        current_city = start_city
        total_length = 0.0

        # 2.贪心选择最近的城市
        for _ in range(num_city - 1):
            distances = dist_matrix[current_city].copy()
            for idx, is_visited in enumerate(visited):
                if is_visited:
                    distances[idx] = np.inf
            # 获取距离 current_city 最近的城市
            next_city = np.argmin(distances)
            total_length += dist_matrix[current_city][next_city]

            current_city = next_city
            visited[next_city] = True

        # 3.添加最后的回路
        total_length += dist_matrix[current_city][start_city]

        return total_length

    def initialize_pheromone_matrix(self, dist_matrix, m):
        """初始化信息素矩阵 τ_0
        τ_0 = m / L_nn
        """
        # 1.计算 τ_0
        num_city = dist_matrix.shape[0]
        L_nn = self.naive_nearest_neighbor(dist_matrix)
        tau_0 = m / L_nn

        print(f'初始化τ_0完成，贪心路径长度为：{L_nn}，τ_0为：{tau_0}')

        # 2.初始化信息素矩阵
        pheromone_matrix = np.full((num_city, num_city), tau_0)
        np.fill_diagonal(pheromone_matrix, 0.0)  # 对角线（自身到自身）设置为 0

        return pheromone_matrix

    def calculate_lengths(self, dist_matrix, tabu_matrix):
        """计算路径长度"""
        # 1.获取当前城市 eg：[0,2,4,1,3]
        current_cities = tabu_matrix
        # 2.获取下一个城市，np.roll使第一个城市移到末尾 eg：[2,4,1,3,0]
        next_cities = np.roll(tabu_matrix, shift=-1, axis=1)
        # 3.获取 tabu_list 中所有边的长度，dist[[0,2,4,1,3],[2,4,1,3,0]]-->[dist[0,2],dist[2,4]...dist[3,0]]
        all_edges_lengths = dist_matrix[current_cities, next_cities]
        # 4.横向相加得到路径长度
        total_lengths = np.sum(all_edges_lengths, axis=1)

        return total_lengths

    def optimize(self, dist_matrix):
        """核心执行架构"""
        # 1.种群数量m设置
        num_cities = dist_matrix.shape[0]
        if num_cities < 200:
            m = num_cities  # 小于200，直接使用
        else:
            m = int(num_cities * 0.5)  # 大于200，取 0.5 保证运行速率

        # 2.初始化路径
        x = np.array(np.random.permutation(num_cities))

        # 3.初始化信息素矩阵，路径长度和禁忌矩阵
        tau = self.initialize_pheromone_matrix(dist_matrix, m)

        # 记录算法生命周期内的历史绝对最优解
        best_overall_length = np.inf
        best_overall_path = None

        # 4.迭代计算
        for g in range(self.ger):
            tabu_matrix = np.zeros((m, num_cities), dtype=int)
            tabu_matrix[:, 0] = np.random.choice(num_cities, size=m, replace=True)

            # 4.2外循环遍历得城市
            for step in range(0, num_cities - 1):
                # 内循环遍历蚂蚁
                for k in range(m):
                    # 获取当前蚂蚁 k 的 tabu_list
                    tabu_list = tabu_matrix[k].copy()
                    current_city = tabu_list[step]
                    # 获取当前蚂蚁 k 的 allowed_cities
                    visited_cities = tabu_list[:step + 1]
                    all_cities = np.arange(num_cities)
                    allowed_mask = ~np.isin(all_cities, visited_cities)
                    allowed_cities = all_cities[allowed_mask]
                    # 计算 current_city 到 allowed_cities 中每一个城市的概率
                    tau_allowed = tau[current_city, allowed_cities]
                    eta_allowed = 1.0 / dist_matrix[current_city, allowed_cities]
                    targets_weights = (tau_allowed ** self.alpha) * (eta_allowed ** self.beta)
                    denominator = np.sum(targets_weights)
                    # 防御性代码：如果分母为0，则剩余城市概率均分
                    if denominator == 0:
                        P = np.full(len(allowed_cities), 1.0 / len(allowed_cities))
                    else:
                        P = targets_weights / denominator
                    # 轮盘赌选择下一个城市
                    next_city = np.random.choice(allowed_cities, p=P)
                    # 更新禁忌表
                    tabu_matrix[k, step + 1] = next_city

            # 4.3更新 total_lengths
            total_lengths = self.calculate_lengths(dist_matrix, tabu_matrix)

            # 4.4更新全局最优
            min_idx = np.argmin(total_lengths)
            if total_lengths[min_idx] < best_overall_length:
                best_overall_length = total_lengths[min_idx]
                best_overall_path = tabu_matrix[min_idx].copy()

            # 4.5更新信息素
            current_idx = tabu_matrix
            next_idx = np.roll(tabu_matrix, shift=-1, axis=1)
            tau = (1 - self.p) * tau  # 信息素蒸发
            for k in range(m):
                delta = self.Q / total_lengths[k]
                tau[current_idx[k], next_idx[k]] += delta

        # 5.返回结果
        return best_overall_length, best_overall_path
