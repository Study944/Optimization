"""遗传算法（最小值版本）
    摘要：将每一个解看作自然中的一个生物，通过模拟自然进化的过程，实行优胜劣汰机制找到最优解。
    引言：一个问题会存在多个可能的解，寻找最优解方法有很多，遗传算法将问题的搜索空间看作自然界，每一个可能的解是一个生物个体，
        然后模拟自然进化过程，淘汰劣的保留优的，再后优秀的生物生成后代，且有变异的可能。
    算法实现：
        (1).编码：问题空间-->自然界。问题的空间往往是连续的，我们无法在一个无限的空间内精准找到最优解，所以将无限的空间映射到
        离散的空间内，常用二进制编码。
            1.二进制位数 n ：用 n 位二进制数表示区间，最大可表达十进制数为 2^n - 1
            2.精度 p ：将连续区间 [L,R] 拆分为 2^n - 1 份，每份大小 p = (R-L)/(2^n - 1)
            3.编码 ：x 属于 [L,R] ，编码为 [0,2^n - 1] D = round[(x-L)/(R-L)*(2^n - 1)]，再计算二进制
        (2).适应度函数 ：Fitness Function 衡量解好坏的唯一标准，接收的参数为原本连续区间 [L,R] 的。
            4.解码 ：将编码后的二进制数解码为连续区间 [L,R] 中真实的数据，x = [D·(R-L)/(2^n - 1)]+L
            5.计算适应度 ：代入解码后的数据 x 到适应度函数 f(x)，得到当前解的适应度
        (3).遗传算子 ：遗传算法的核心，实行优胜劣汰机制保留优解，优解之间交叉基因产生后代，部分解发生变异。
            6.选择(轮盘赌) ：根据每个解的适应度函数结果好坏计算每个解的概率，随机 N 次概率选择后代
            7.交叉 ：按照交叉概率 c ，任意选择 N·c 个解进行基因交叉产生后代，后代直接代替原本解，保证解数量为 N
            8.变异 ：按照变异概率 m ，每个解都有概率 m 发生变异，变异后该解随机一位进行取反，为了跳出局部最优
    案例 ：
        实现遗传算法寻找 Rastrigin 多峰函数最小值
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MyGA:
    """ mini 遗传算法类（最小值版本）
    """

    def __init__(self, pop_size=100, dna_size=10, n_generation=100, cross_rate=0.8, mutation_rate=0.01, dim=None):
        """ 初始化
        """
        self.pop_size = pop_size  # 种群数量
        self.dna_size = dna_size  # 基因长度
        self.n_generation = n_generation  # 迭代次数
        self.cross_rate = cross_rate  # 交叉概率
        self.mutation_rate = mutation_rate  # 变异概率
        self.dim = dim  # 维度
        
        # 记录历史数据
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.population_history = []

    def _translate_dna(self, pop, limit):
        """解码
            将编码后的二进制数据转换回原本实际区间数据
            遵循公式 ： x = [D·(R-L)/(2^n - 1)]+L
        """
        # 1.二进制-->十进制
        pop_reshaped = pop.reshape(self.pop_size, self.dim, self.dna_size)
        powers = 2 ** np.arange(self.dna_size - 1, -1, -1)
        decimal_values = np.dot(pop_reshaped, powers)
        # 2.归一化
        max_decimal = 2 ** self.dna_size - 1
        normalized = decimal_values / max_decimal
        # 3.映射到实际区间[L,R]
        L = limit[:, 0]
        R = limit[:, 1]
        decoded = normalized * (R - L) + L
        return decoded

    def _get_fitness(self, fitness_function, x):
        """计算适应值（最小值版本）
            对于最小值问题，需要将目标函数值转换为适应度
            适应度应该与目标函数值成反比：目标函数值越小，适应度越高
            使用倒数法或最大值减法
        """
        pred = fitness_function(x)
        # 核心：对于最小值问题，使用最大值减去当前值来获得适应度
        # 加上一个小值防止除零或负数
        max_val = np.max(pred)
        min_val = np.min(pred)
        
        # 方法1：使用最大值减法（推荐）
        fitness = max_val - pred + 1e-3
        
        # 方法2：如果所有值都很接近，使用倒数法（备选）
        # if max_val - min_val < 1e-6:
        #     fitness = 1.0 / (pred + 1e-6)
        # else:
        #     fitness = 1.0 / (pred - min_val + 1e-6)
        
        return fitness

    def _select(self, fitness, pop):
        """轮盘赌
            按照概率选择保留的解
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return pop[idx]

    def _evolve(self, pop):
        """交叉变异
            每个解都有 cross_rate 发生交叉，随机选择另一个解和交叉位置
            每个都有 mutation_rate 发生变异，随机选择变异位置基因取反
        """
        new_pop = []
        for father in pop:
            # 交叉
            child = father.copy()
            if np.random.rand() < self.cross_rate:
                mother = pop[np.random.randint(self.pop_size)]
                cross_point = np.random.randint(0, self.dna_size*self.dim)
                child[cross_point:] = mother[cross_point:]
            # 变异
            if np.random.rand() < self.mutation_rate:
                mutate_point = np.random.randint(0, self.dna_size*self.dim)
                child[mutate_point] ^= 1
            new_pop.append(child)
        return np.array(new_pop)

    def optimize(self, fitness_function, limit):
        """ 算法实现（最小值版本）
        """
        # 1.参数初始化
        # np.random.seed(42)
        pop = np.random.randint(2, size=(self.pop_size, self.dna_size * self.dim))
        
        # 清空历史记录
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.population_history = []
        
        # 2.迭代计算
        for i in range(self.n_generation):
            # 3.DNA解码
            x = self._translate_dna(pop, limit)
            # 4.计算适应值
            fitness = self._get_fitness(fitness_function, x)
            
            # 记录历史数据（记录实际的目标函数值，而非适应度）
            actual_fitness = fitness_function(x)
            self.best_fitness_history.append(np.min(actual_fitness))  # 最小值
            self.avg_fitness_history.append(np.mean(actual_fitness))
            self.population_history.append(x.copy())
            
            # 5.轮盘赌
            pop = self._select(fitness, pop)
            # 6.交叉变异
            pop = self._evolve(pop)
        
        # 7.寻找最优解（最小值）
        final_x = self._translate_dna(pop, limit)
        final_fit = fitness_function(final_x)
        best_idx = np.argmin(final_fit)  # 改为 argmin
        return final_x[best_idx], final_fit[best_idx]


# Rastrigin 函数（用于最小化）
class ObjectiveFunction:

    def __call__(self, x):
        if x.ndim == 1:
            n = x.shape[0]
            return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        else:
            n = x.shape[1]
            return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

    def evaluate_mesh(self, X, Y):
        """用于网格数据的评估"""
        return 10 * 2 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y)


def FitnessFunction(x):
    """Rastrigin适应度函数（最小值版本）
    """
    n = x.shape[1]
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


if __name__ == '__main__':
    # ==================== 参数设置 ====================
    pop_size = 100  # 种群数量
    dna_size = 10   # DNA长度
    n_generation = 100  # 迭代次数
    dim = 2  # 维度
    limits = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    
    # ==================== 运行遗传算法 ====================
    print("正在运行遗传算法（最小值版本）...")
    ga = MyGA(pop_size=pop_size, dna_size=dna_size, n_generation=n_generation,
              cross_rate=0.8, mutation_rate=0.01, dim=dim)
    
    best_solution, best_fitness = ga.optimize(FitnessFunction, limits)
    
    print(f"最优解坐标: {best_solution}")
    print(f"最优适应度（最小值）: {best_fitness:.6f}")
    print(f"理论最小值: 0.0 (在点 (0, 0))")
    
    # ==================== 准备可视化数据 ====================
    f = ObjectiveFunction()
    
    # 生成函数曲面数据
    x_range = np.linspace(limits[0, 0], limits[0, 1], 200)
    y_range = np.linspace(limits[1, 0], limits[1, 1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f.evaluate_mesh(X, Y)
    
    # ==================== 图1：静态可视化 ====================
    print('\n正在生成静态图表...')
    
    fig1 = plt.figure(figsize=(18, 6))
    
    # 子图1：3D目标函数曲面图
    ax1 = fig1.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('f(X, Y)', fontsize=12)
    ax1.set_title('3D Rastrigin Function Surface', fontsize=14, fontweight='bold')
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # 标记理论最小值点 (0, 0, 0)
    ax1.scatter([0], [0], [0], color='red', s=200, marker='*', 
               label='Theoretical Min (0, 0, 0)', zorder=5)
    ax1.legend(fontsize=10)
    
    # 子图2：等高线图 + 种群最终位置
    ax2 = fig1.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    contour_lines = ax2.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour, ax=ax2)
    
    # 显示最终种群位置
    final_population = np.array(ga.population_history[-1])
    final_fitness = f(final_population)
    scatter = ax2.scatter(final_population[:, 0], final_population[:, 1], 
                         c=final_fitness, s=100, alpha=0.7, 
                         edgecolors='black', linewidth=1.5, 
                         cmap='hot', label='Population (Final)', zorder=5)
    
    # 标记全局最优位置
    ax2.plot(best_solution[0], best_solution[1], 'm*', markersize=25, 
            label=f'Best Solution ({best_solution[0]:.2f}, {best_solution[1]:.2f})', zorder=6)
    
    # 标记理论最优位置
    ax2.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum (0, 0)', zorder=7)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Population Distribution on Contour Map', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(limits[0, 0], limits[0, 1])
    ax2.set_ylim(limits[1, 0], limits[1, 1])
    
    # 子图3：收敛曲线
    ax3 = fig1.add_subplot(133)
    generations = range(1, n_generation + 1)
    ax3.plot(generations, ga.best_fitness_history, 'g-', linewidth=2, label='Best Fitness (Min)')
    ax3.plot(generations, ga.avg_fitness_history, 'b--', linewidth=1.5, label='Average Fitness', alpha=0.7)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Fitness', fontsize=12)
    ax3.set_title('Convergence Curve', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5, 
               label='Theoretical Min (0)', alpha=0.7)
    ax3.plot(n_generation, best_fitness, 'ro', markersize=10, 
            label=f'Final: {best_fitness:.6f}')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # ==================== 图2：种群在不同代数的位置分布 ====================
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    
    colors = ['red', 'orange', 'green']
    labels = ['Initial', 'Middle', 'Final']
    indices = [0, n_generation // 2, n_generation - 1]
    
    for idx, color, label in zip(indices, colors, labels):
        positions = np.array(ga.population_history[idx])
        fitness = f(positions)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                            c=color, s=80, alpha=0.6, 
                            edgecolors='black', linewidth=1.2, 
                            label=label, zorder=5)
    
    # 标记全局最优位置
    ax.plot(best_solution[0], best_solution[1], 'm*', markersize=25, 
           label=f'Best Solution', zorder=6)
    
    # 标记理论最优位置
    ax.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum (0, 0)', zorder=7)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Population Evolution Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(limits[0, 0], limits[0, 1])
    ax.set_ylim(limits[1, 0], limits[1, 1])
    
    plt.tight_layout()
    plt.show()

    # ==================== 图3：动态动画 ====================
    print('\n正在生成动态动画...')
    
    fig_anim, (ax1_anim, ax2_anim) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图：等高线图 + 种群移动
    contour_anim = ax1_anim.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    contour_lines = ax1_anim.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3, alpha=0.4)
    fig_anim.colorbar(contour_anim, ax=ax1_anim, shrink=0.8)
    
    scatter_anim = ax1_anim.scatter([], [], c='red', s=120, alpha=0.8, 
                                   edgecolors='black', linewidth=1.5, 
                                   label='Population', zorder=5)
    best_point_anim = ax1_anim.plot([], [], 'm*', markersize=25, label='Best Solution')[0]
    theoretical_point = ax1_anim.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum')[0]
    
    ax1_anim.set_xlabel('X', fontsize=12)
    ax1_anim.set_ylabel('Y', fontsize=12)
    ax1_anim.set_title('GA Population Evolution Animation (Rastrigin Min)', fontsize=14, fontweight='bold')
    ax1_anim.legend(fontsize=10, loc='upper right')
    ax1_anim.grid(True, alpha=0.3)
    ax1_anim.set_xlim(limits[0, 0], limits[0, 1])
    ax1_anim.set_ylim(limits[1, 0], limits[1, 1])
    
    text1_anim = ax1_anim.text(0.02, 0.98, '', transform=ax1_anim.transAxes, fontsize=10,
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # 右图：收敛曲线
    line_best_anim, = ax2_anim.plot([], [], 'g-', linewidth=2.5, label='Best Fitness (Min)')
    line_avg_anim, = ax2_anim.plot([], [], 'b--', linewidth=1.5, label='Average Fitness', alpha=0.7)
    best_point_anim2 = ax2_anim.plot([], [], 'ro', markersize=10, label='Current Best')[0]
    theoretical_line = ax2_anim.axhline(y=0, color='r', linestyle='--', linewidth=1.5, 
                                       label='Theoretical Min (0)', alpha=0.7)
    
    ax2_anim.set_xlabel('Generation', fontsize=12)
    ax2_anim.set_ylabel('Fitness', fontsize=12)
    ax2_anim.set_title('Convergence Process', fontsize=14, fontweight='bold')
    ax2_anim.legend(fontsize=10)
    ax2_anim.grid(True, alpha=0.3)
    ax2_anim.set_xlim(0, n_generation)
    
    text2_anim = ax2_anim.text(0.02, 0.98, '', transform=ax2_anim.transAxes, fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def init():
        """初始化动画"""
        scatter_anim.set_offsets(np.empty((0, 2)))
        best_point_anim.set_data([], [])
        line_best_anim.set_data([], [])
        line_avg_anim.set_data([], [])
        best_point_anim2.set_data([], [])
        text1_anim.set_text('')
        text2_anim.set_text('')
        return scatter_anim, best_point_anim, line_best_anim, line_avg_anim, best_point_anim2, text1_anim, text2_anim
    
    def animate(frame):
        """动画更新函数"""
        current_gen = frame + 1
        
        # 更新左图：种群位置
        positions = np.array(ga.population_history[frame])
        fitness = f(positions)
        scatter_anim.set_offsets(positions)
        
        # 更新最优点（最小值）
        best_idx = np.argmin(fitness)  # 改为 argmin
        best_x = positions[best_idx, 0]
        best_y = positions[best_idx, 1]
        best_point_anim.set_data([best_x], [best_y])
        
        # 更新文本信息
        text1_anim.set_text(f'Generation: {current_gen}/{n_generation}\n'
                           f'Population: {pop_size}\n'
                           f'Current Best (Min):\n'
                           f'  x={best_x:.4f}, y={best_y:.4f}\n'
                           f'  f(x,y)={np.min(fitness):.6f}')
        
        # 更新右图：收敛曲线
        gens = np.arange(1, current_gen + 1)
        best_history = np.array(ga.best_fitness_history[:current_gen])
        avg_history = np.array(ga.avg_fitness_history[:current_gen])
        line_best_anim.set_data(gens, best_history)
        line_avg_anim.set_data(gens, avg_history)
        best_point_anim2.set_data([current_gen], [best_history[-1]])
        
        # 动态调整y轴范围
        margin = abs(np.max(best_history) - np.min(best_history)) * 0.1
        if margin < 1:
            margin = 1
        ax2_anim.set_ylim(np.min(best_history) - margin, np.max(best_history) + margin)
        
        text2_anim.set_text(f'Current Generation: {current_gen}\n'
                           f'Best Fitness (Min): {best_history[-1]:.6f}\n'
                           f'Avg Fitness: {avg_history[-1]:.6f}\n'
                           f'Theoretical Min: 0.0')
        
        return scatter_anim, best_point_anim, line_best_anim, line_avg_anim, best_point_anim2, text1_anim, text2_anim
    
    # 创建动画
    anim = FuncAnimation(fig_anim, animate, init_func=init,
                        frames=n_generation, interval=100, blit=True, repeat=False)
    
    print('动画生成完成！')
    plt.show()
