"""LSHADE算法（DE 的最强变种）
L：线性减少的种群数量，种群数量随轮次迭代线性下降，自动淘汰表现不好的个体，集中算力到最优值中
SHAPE：历史成功参数记忆库，通过记录更优的 F 和 CR，不断更新选择最合适的 F 和 CR
"""

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


class LSHADE:
    """LSHADE 算法"""

    def __init__(self, F=0.5, CR=0.9, dim=2, ger=50):
        """初始化"""
        self.F = F
        self.CR = CR
        self.D = dim
        self.ger = ger
        self.H = 5
        self.k = 0

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

    def _get_F(self, M_F, N):
        """柯西分布计算 F
        F = mu + sigma * 标准柯西分布
        """
        # 1.从 M_F 中随机抽取 mu
        H = len(M_F)
        mu_idx = np.random.randint(0, H, size=N)
        mu_cur = M_F[mu_idx]
        # 2.设置sigma，论文默认0.1
        sigma = 0.1
        # 3.计算 F
        F = mu_cur + sigma * np.random.standard_cauchy(size=N)
        F = np.clip(F, 0.0, 1.0)  # 边界处理
        return F

    def _get_CR(self, M_CR, N):
        """高斯分布计算 CR"""
        # 1.从 M_F 中随机抽取 mu
        H = len(M_CR)
        mu_idx = np.random.randint(0, H, size=N)
        mu_cur = M_CR[mu_idx]
        # 2.设置sigma，论文默认0.1
        sigma = 0.1
        # 3.计算 F
        CR = np.random.normal(loc=mu_cur, scale=sigma, size=N)
        CR = np.clip(CR, 0.0, 1.0)  # 边界处理
        return CR

    def optimize(self, fitness_function, limits):
        """核心执行架构"""
        # 1.初始化种群数量
        N_max = 18 * self.D
        N_min = 4
        # 1.2 计算边界
        limits = np.array(limits)
        lower_bounds = limits[:, 0]
        upper_bounds = limits[:, 1]
        # 1.3 利用 numpy 广播机制初始化种群
        x = lower_bounds + np.random.rand(N_max, self.D) * (upper_bounds - lower_bounds)
        # 1.4 初始化基本参数
        fitness_best = fitness_function(x)
        # 1.5 初始化 M_F 和 M_CR
        M_F = np.full((self.H,), self.F)
        M_CR = np.full((self.H,), self.CR)

        # 2.迭代计算
        for g in range(self.ger):
            # --种群数量线性下降--
            N = round(((N_min - N_max) / self.ger) * g + N_max)

            # 更新最优 F 和 CR
            F = self._get_F(M_F, N)
            CR = self._get_CR(M_CR, N)

            # 2.1 个体位置更新
            r1, r2, r3 = self._get_indices(N)
            v = x[r1] + F[:, None] * (x[r2] - x[r3])
            # 边界判断
            v = np.clip(v, lower_bounds, upper_bounds)

            # 2.2 交叉
            rand_matrix = np.random.rand(N, self.D)
            j_rand = np.random.randint(0, self.D, size=N)
            crossover_mask = (rand_matrix < CR[:, None]) | (np.arange(self.D) == j_rand[:, None])
            x_new = np.where(crossover_mask, v, x)

            # 2.3 选择
            fitness_new = fitness_function(x_new)
            fitness_old = np.copy(fitness_best)  # 保存用于计算 CR 更新权重
            better_mask = fitness_new < fitness_best

            # 2.4 更新
            x = np.where(better_mask[:, None], x_new, x)
            fitness_best = np.where(better_mask, fitness_new, fitness_best)

            # 打印输出
            best_idx = np.argmin(fitness_best)
            print(f'第{g}代最优个体位置：{x[best_idx]},最优适应度：{fitness_best[best_idx]},种群数量:{N}')

            # --存储参数记忆库--
            success_indices = np.flatnonzero(better_mask)
            if len(success_indices) > 0:
                # 1. 计算适应度改善量 (做权重)
                delta_fit = np.abs(fitness_old[success_indices] - fitness_new[success_indices])
                weights = delta_fit / (np.sum(delta_fit) + 1e-10)  # 归一化权重
                # 2. 获取成功的参数
                success_F = F[success_indices]
                success_CR = CR[success_indices]
                # 3. 计算 CR_new (算术平均)
                CR_new = np.sum(weights * success_CR)
                # 4. 计算 F_new (Lehmer 平均)
                F_new = np.sum(weights * (success_F ** 2)) / (np.sum(weights * success_F) + 1e-10)
                # 5. 滚动更新记忆库
                M_F[self.k] = F_new
                M_CR[self.k] = CR_new
                # 6. 指针向前移动，满了就从0开始覆盖
                self.k = (self.k + 1) % self.H

            # --种群数量线性下降--
            N_next = round(((N_min - N_max) / self.ger) * (g + 1) + N_max)
            if N_next < N:
                sorted_idx = np.argsort(fitness_best)
                keep_idx = sorted_idx[:N_next]  # 准许晋级下一代的前 N_next 强
                x = x[keep_idx]
                fitness_best = fitness_best[keep_idx]

        # 3.返回结果
        best_idx = np.argmin(fitness_best)
        return x[best_idx], fitness_best[best_idx]
