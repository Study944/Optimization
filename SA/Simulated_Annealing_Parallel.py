"""模拟退火算法(并行)"""

import numpy as np


class SA_Parallel:
    """模拟退火算法(并行)"""

    def __init__(self, L=100, dim=1, p_start=0.85, t_end=1e-6, alpha=0.98, m_seeds=1):
        """初始化"""
        self.L = L
        self.dim = dim
        self.p_start = p_start
        self.t_end = t_end
        self.alpha = alpha
        self.fitness_history = None
        self.m_seeds = m_seeds

    def _template_init(self, fitness_function, limit):
        """
        初始化温度 T_start
            T_start = -ΔE_avg/lnP_start
        """
        # 1.迭代 100 轮计算 ΔE
        delta_E_list = []
        for i in range(100):
            x1 = np.random.uniform(limit[:, 0], limit[:, 1], size=self.dim)
            x2 = np.random.uniform(limit[:, 0], limit[:, 1], size=self.dim)
            delta_E = abs(fitness_function(x1) - fitness_function(x2))
            delta_E_list.append(delta_E)
        # 2.计算 ΔE_avg ，代入计算 T_start
        delta_E_avg = np.mean(delta_E_list)
        t_start = -delta_E_avg / np.log(self.p_start)
        return t_start

    def _get_next(self, x, step_size, limits):
        """
        计算新解位置
            x_new = x + step_size * rand(-1,1)
        """
        # 1.添加随机扰动
        dx = np.random.uniform(-step_size, step_size, size=x.shape)
        x_new = x + dx

        lb, ub = limits[:, 0], limits[:, 1]  # 形状 (dim,)

        # 1. 优雅处理上限越界：如果 x_new > ub，则反弹回 2*ub - x_new，否则保持原样
        # np.where 会自动把 (2,) 的 ub 广播到 (m_seeds, dim)
        x_new = np.where(x_new > ub, 2 * ub - x_new, x_new)

        # 2. 优雅处理下限越界：如果 x_new < lb，则反弹回 2*lb - x_new，否则保持原样
        x_new = np.where(x_new < lb, 2 * lb - x_new, x_new)

        return np.clip(x_new, lb, ub)

    def optimize(self, fitness_function, limits):
        """算法逻辑"""
        # 1，初始化参数
        limits = np.atleast_2d(limits)  # 确保至少是二维的
        if limits.shape[0] == 1 and self.dim > 1:
            limits = np.tile(limits, (self.dim, 1))
        # 生成 m_seeds 个种子
        x = np.random.uniform(limits[:, 0], limits[:, 1], size=(self.m_seeds,self.dim))
        fitness = np.array([fitness_function(ind) for ind in x])

        template = self._template_init(fitness_function, limits)  # 初始化温度
        template_start = template

        # 全局历史最优记录 (从所有种子中挑个最好的)
        best_idx = np.argmin(fitness)
        x_best = np.copy(x[best_idx])
        fitness_best = fitness[best_idx]

        # 用于画图的收敛历史（记录每一轮里所有解之中的最好值）
        self.fitness_history = [fitness_best]

        # 2.迭代计算
        while template > self.t_end:
            # 外层循环控制 步长
            cur_step_size = (limits[:, 1] - limits[:, 0]) * (template / template_start) * 0.1

            for _ in range(self.L):
                # 内层更新位置
                x_new = self._get_next(x, cur_step_size, limits)

                # 计算所有种子的 ΔE
                fitness_new = np.array([fitness_function(ind) for ind in x_new])
                delta_E = fitness_new - fitness

                # 对每一个独立的退火进行判定
                for s in range(self.m_seeds):
                    if delta_E[s] < 0:
                        x[s] = x_new[s]
                        fitness[s] = fitness_new[s]
                        # 更新全局最优
                        if fitness_new[s] < fitness_best:
                            x_best = np.copy(x_new[s])
                            fitness_best = fitness_new[s]
                    else:
                        p = np.exp(-delta_E[s] / template)
                        if np.random.rand() < p:
                            x[s] = x_new[s]
                            fitness[s] = fitness_new[s]

                # 记录当前这一步的全局最优历史，平滑
                # self.fitness_history.append(fitness_best)
                # 记录当前这一步，整个种群里实时表现最好的个体的适应度，震荡
                current_population_best = np.min(fitness)
                self.fitness_history.append(current_population_best)

            # 更新温度
            template *= self.alpha

        return x_best, fitness_best


if __name__ == "__main__":
    def func(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


    sa = SA_Parallel(dim=2,m_seeds=5)
    limit = [[-5.12, 5.12], [-5.12, 5.12]]
    x_best, fitness_best = sa.optimize(func, limit)

    print(f'最小值位置：{x_best}')
    print(f'最佳适应值：{fitness_best}')

    import matplotlib.pyplot as plt

    # 允许 matplotlib 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 提取历史记录
    fitness_history = np.array(sa.fitness_history)
    total_steps = len(fitness_history)

    # 收敛曲线图
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(fitness_history, color='#1f77b4', linewidth=1.5, label='当前解适应度')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.6, label='理论全局最优值(0)')
    plt.title(f'模拟退火算法(SA)收敛曲线（总评估次数: {total_steps}次）', fontsize=12)
    plt.xlabel('内层迭代总步数 (Evaluation Steps)', fontsize=10)
    plt.ylabel('适应度函数值 (Energy / Fitness)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')

    # 局部放大图提示：你会看到前中期曲线上有很多“向上跳跃”的红点，那就是 Metropolis 准则在发挥作用
    plt.tight_layout()
    plt.show()


