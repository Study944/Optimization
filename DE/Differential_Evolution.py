"""差分进化算法（Differential Evolution）
    摘要：差分进化算法根据“统计分布”学，通过计算两个个体的差值并作为个体移动的速度进行更新个体位置。
    引言：两个个体差值向量并不是毫无意义的，前期个体距离远差值大即步长大善于范围探索，后期种群聚集在最优值附近差值小
        即步长小善于精准探索。同时差值方向会大概率指向最优值方向（隐式协同进化）。
    核心原理：
        缩放因子 F ：差值（步长）的前置系数，用于控制种群的移动幅度
        交叉概率 CR ：交叉操作使用的参数，用于二项式交叉中的是否交叉判断
        选择操作 ：交叉结束后是否应用新个体的判断，根据适应度函数评估
    算法流程：
        参数设置：F = 0.5，CR = 0.9，NP = 10D
        更新操作：v_i = x_i + F*(x_a - x_b)
        交叉操作：rand(0,1) < CR --> 这个维度的数据选择 v_i 否则选择 x_i，最终得到 x_new
        选择操作：计算对比 x_new 和 x_i 的适应度，若 x_new 适应度更小则进行更新
"""

import numpy as np


class DE:
    """差分进化算法 DE """

    def __init__(self, F=0.5, CR=0.9, dim=2, ger=50):
        """初始化"""
        self.F = F
        self.CR = CR
        self.D = dim
        self.ger = ger

    def _get_indices(self, N):
        """计算r1，r2和r3，r1!=r2!=r3!=i"""
        r1 = np.zeros(N, dtype=int)
        r2 = np.zeros(N, dtype=int)
        r3 = np.zeros(N, dtype=int)
        for i in range(N):
            choices = [idx for idx in range(N) if idx != i]
            idx3 = np.random.choice(choices, 3, replace=False)
            r1[i], r2[i], r3[i] = idx3
        return r1, r2, r3

    def optimize(self, fitness_function, limits):
        """核心执行架构"""
        # 1.种群数量 = 10*D
        N = 10 * self.D
        # 1.2 计算边界
        limits = np.array(limits)
        lower_bounds = limits[:, 0]
        upper_bounds = limits[:, 1]
        # 1.3 利用 numpy 广播机制初始化种群
        x = lower_bounds + np.random.rand(N, self.D) * (upper_bounds - lower_bounds)
        # 1.4 初始化基本参数
        fitness_best = fitness_function(x)

        # 2.迭代计算
        for g in range(self.ger):
            # 2.1 个体位置更新
            r1, r2, r3 = self._get_indices(N)
            v = x[r1] + self.F * (x[r2] - x[r3])
            # 边界判断
            v = np.clip(v, lower_bounds, upper_bounds)

            # 2.2 交叉
            rand_matrix = np.random.rand(N, self.D)
            j_rand = np.random.randint(0, self.D, size=N)
            crossover_mask = (rand_matrix < self.CR) | (np.arange(self.D) == j_rand[:, None])
            x_new = np.where(crossover_mask, v, x)

            # 2.3 选择
            fitness_new = fitness_function(x_new)
            better_mask = fitness_new < fitness_best

            # 2.4 更新
            x = np.where(better_mask[:, None], x_new, x)
            fitness_best = np.where(better_mask, fitness_new, fitness_best)

            # 打印输出
            best_idx = np.argmin(fitness_best)
            print(f'第{g}代最优个体位置：{x[best_idx]},最优适应度：{fitness_best[best_idx]}')

        # 3.返回结果
        best_idx = np.argmin(fitness_best)
        return x[best_idx], fitness_best[best_idx]
