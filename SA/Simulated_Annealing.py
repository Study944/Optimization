"""模拟退火算法 SA （Simulated Annealing）
    摘要：模拟退火算法通过模拟物理现象，解决寻找最优解问题，并且实现概率更新到更差的位置以跳出局部最优。
    引言：模拟退火算法源自物理中固体降温结晶原理，缓慢降温（退火）则得到晶体，快速降温得到的则是非晶体。
    核心原理：
        状态（解）：问题的一个可行解
        能量（适应度函数）：衡量状态好坏的标准
        温度：可变参数，用来计算跳出局部最优的概率
        基态（全局最优解）：问题的最优解
    算法流程（最小值问题举例）：
        1.定义问题--适应度函数 E，解 x 的维度，解的区间
        2.定义参数--迭代轮次 L，温度 T 初始值及变化函数
        3.更新条件--ΔE = E(x_new) - E(x)，ΔE < 0 说明新位置 x_new 更优，否则 x 更优
        4.概率更新--当 ΔE > 0 时 x 更优，但是避免全局最优问题，在概率 P = e^-(ΔE/T) 更新
        5.全局最优--避免最后一轮时概率 P = e^-(ΔE/T) 更新，迭代中保存历史最优 x_best
    核心问题：
        T 的初始化：T_Start 应该随着问题规模的变大而变大，而不是固定的，按照 P = e^-(ΔE/T) 反算 T = -ΔE/lnP，
            通常设置 P_Start = 0.85，ΔE 可以在退火之前进行计算 100 轮迭代下 ΔE 的均值 ΔE_avg。
        T 的变化：随着轮次 L 上升 T 应该下降，二者呈负相关。T = α * T，（α属于0-1）L 越大 α 越大，反则越小。
"""

import numpy as np


class SA:
    """模拟退火算法 SA """

    def __init__(self, L=100, dim=1, p_start=0.85, t_end=1e-6, alpha=0.98):
        """初始化"""
        self.L = L
        self.dim = dim
        self.p_start = p_start
        self.t_end = t_end
        self.alpha = alpha
        self.fitness_history = None
        self.x_history = None

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
        dx = np.random.uniform(-step_size, step_size, size=self.dim)
        x_new = x + dx

        # 2. 优雅的反射边界处理（一行矩阵运算搞定所有维度）
        lb, ub = limits[:, 0], limits[:, 1]
        # 处理上限越界：如果 x > ub，反弹回 ub - (x - ub) = 2*ub - x
        over_ub = x_new > ub
        x_new[over_ub] = 2 * ub[over_ub] - x_new[over_ub]

        # 处理下限越界：如果 x < lb，反弹回 lb + (lb - x) = 2*lb - x
        under_lb = x_new < lb
        x_new[under_lb] = 2 * lb[under_lb] - x_new[under_lb]

        return x_new

    def optimize(self, fitness_function, limits):
        """算法逻辑"""
        # 1，初始化参数
        limits = np.atleast_2d(limits)  # 确保至少是二维的
        if limits.shape[0] == 1 and self.dim > 1:
            limits = np.tile(limits, (self.dim, 1))
        x = np.random.uniform(limits[:, 0], limits[:, 1], size=self.dim)  # 利用NumPy自动广播生成单解
        fitness = fitness_function(x)

        template = self._template_init(fitness_function, limits)  # 初始化温度
        template_start = template

        x_best = np.copy(x)  # 保存历史最优解
        fitness_best = fitness

        self.fitness_history = [fitness]
        self.x_history = [np.copy(x)]


        # 2.迭代计算
        while template > self.t_end:
            # 外层循环控制 步长
            cur_step_size = (limits[:, 1] - limits[:, 0]) * (template / template_start) * 0.1

            for _ in range(self.L):
                # 内层更新位置
                x_new = self._get_next(x, cur_step_size, limits)

                # 计算 ΔE
                fitness_new = fitness_function(x_new)
                delta_E = fitness_new - fitness

                # 更新
                if delta_E < 0:
                    # 一定更新
                    x = x_new
                    fitness = fitness_new
                    if fitness_new < fitness_best:
                        x_best = np.copy(x)
                        fitness_best = fitness
                else:
                    # 概率更新 P = e^(ΔE/T)
                    p = np.exp(-delta_E / template)
                    if np.random.rand() < p:
                        x = x_new
                        fitness = fitness_new

                self.fitness_history.append(fitness)
                self.x_history.append(np.copy(x))

            # 更新温度
            template *= self.alpha


        return x_best, fitness_best


if __name__ == "__main__":
    def func(x):
        return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


    sa = SA(dim=2)
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
    x_history = np.array(sa.x_history)
    total_steps = len(fitness_history)

    # ---------------- 图 1：收敛曲线图 ----------------
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

    # ---------------- 图 2：2D 状态空间搜索轨迹图 ----------------
    # 1. 准备 Rastrigin 函数的等高线数据
    X1 = np.linspace(-5.12, 5.12, 200)
    X2 = np.linspace(-5.12, 5.12, 200)
    X1, X2 = np.meshgrid(X1, X2)
    Z = 10 * 2 + (X1 ** 2 - 10 * np.cos(2 * np.pi * X1)) + (X2 ** 2 - 10 * np.cos(2 * np.pi * X2))

    plt.figure(figsize=(8, 7), dpi=100)
    # 画出绚丽的背景等高线
    contour = plt.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Rastrigin 函数值')

    # 画出 SA 单解的移动轨迹（用渐变色代表时间先后：由浅入深）
    colors = plt.cm.autumn(np.linspace(0, 1, total_steps))
    plt.scatter(x_history[:, 0], x_history[:, 1], c=colors, s=3, alpha=0.6, label='解移动轨迹')

    # 标出起点、终点和理论最优
    plt.plot(x_history[0, 0], x_history[0, 1], 'go', markersize=8, label='初始随机起点')
    plt.plot(x_best[0], x_best[1], 'ro', markersize=8, label='算法寻优终点')
    plt.plot(0, 0, 'b*', markersize=12, label='理论全局中心(0,0)')

    plt.title('模拟退火在二维 Rastrigin 函数上的空间搜索轨迹', fontsize=12)
    plt.xlabel('x1', fontsize=10)
    plt.ylabel('x2', fontsize=10)
    plt.xlim(-5.12, 5.12)
    plt.ylim(-5.12, 5.12)
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.show()
