"""
    摘要：使用粒子群算法 PSO 寻找维度为 1 且区间有限的函数 f(x) 最小值。
    引言：常规找区间内函数最小值，通过求导或通过遍历区间得到。遍历区间的时间复杂度极高， PSO 参考鸟群觅食行为，核心假设为通过
        社会信息分享，个体可以从整个群体中获益，从而在寻找最优解中获取优势。
    核心原理：
        使用场景：优化目标达到最优值（最小值/最大值）；
        适应度：粒子位置计算的结果越小/大，适应度越高；
        个人位置 site ：粒子当前的位置（坐标）；
        个人历史最优位置 p_best ：粒子在移动过程中适应度最高的位置；
        全局历史最优位置 g_best : 所有粒子在移动过程中适应度最高的位置。
    数学推导：
        粒子通过 t 轮移动接近最优值
        粒子种群 x ，粒子位置 x_i(t) ，粒子区间x_max
        粒子速度 v_i(t) ，最大速度 v_max （v<=v_max）
        v_i(t+1) = 惯性系数 * v_i(t) + c_1 * (p_best - x_i(t)) + c_2 * (g_best - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
    实现：
        使用 PSO 计算 f(x) = x * sin(x) * cos(2x) − 2x * sin(3x) + 3x * sin(4x)在[0,50]的最小值。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class ObjectiveFunction:
    # 目标函数类
    def __init__(self):
        pass

    def __call__(self, x):
        """
        计算函数值
        f(x) = x * sin(x) * cos(2x) - 2x * sin(3x) + 3x * sin(4x)
        """
        return x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x)

    def evaluate(self, x):
        return self.__call__(x)


if __name__ == '__main__':
    # 1.定义函数
    f = ObjectiveFunction()

    # 2.定义参数
    N = 20  # 种群粒子数量
    dim = 1  # 维度
    ger = 100  # 迭代次数
    limit = [0, 50]  # 范围区间
    v_limit = [-5, 5]  # 速度区间
    w = 0.9  # 惯性系数
    c1 = 2.0  # 自我学习因子
    c2 = 2.0  # 群体学习因子

    # 3.初始化种群粒子
    x = np.random.uniform(limit[0], limit[1], N)
    v = np.random.uniform(v_limit[0], v_limit[1], N)
    p_best = np.copy(x)
    p_best_evaluate = f.evaluate(x)
    g_best = p_best[np.argmin(p_best_evaluate)]
    g_best_evaluate = p_best_evaluate[np.argmin(p_best_evaluate)]

    # 记录历史数据用于绘图
    g_best_history = []
    x_history = []  # 记录所有粒子的位置历史

    # 4.种群移动
    t = 0
    while t < ger:
        # 同步更新惯性权重
        w = 0.9 - (0.5 * t / ger)
        # 4.1 粒子移动
        r1, r2 = np.random.rand(N), np.random.rand(N)
        v = w * v + c1 * r1 * (p_best - x) + c2* r2 * (g_best - x)
        # 限制速度在范围内
        v = np.clip(v, v_limit[0], v_limit[1])
        # 计算新位置
        x = np.clip(x + v, limit[0], limit[1])
        # 计算适应值
        evaluate_new = f.evaluate(x)
        # 4.2更新粒子历史最优
        # for i in range(N):
        #     if evaluate_new[i] < p_best_evaluate[i]:
        #         p_best[i] = x[i]
        #         p_best_evaluate[i] = evaluate_new[i]
        mask = evaluate_new < p_best_evaluate
        p_best[mask] = x[mask]
        p_best_evaluate[mask] = evaluate_new[mask]
        # 4.3更新全局历史最优
        min_idx = np.argmin(evaluate_new)
        if np.min(p_best_evaluate) < g_best_evaluate:
            g_best = p_best[np.argmin(p_best_evaluate)]
            g_best_evaluate = p_best_evaluate[np.argmin(p_best_evaluate)]
        # 记录历史数据
        g_best_history.append(g_best_evaluate)
        x_history.append(x.copy())
        
        # 轮次加1
        t += 1


    print(f'粒子历史最优位置：{p_best}')
    print(f'粒子历史最优适应度：{p_best_evaluate}')
    print(f'\n种群历史最优位置：{g_best}')
    print(f'种群历史最优适应度：{g_best_evaluate}')

    # ==================== 绘制静态图 ====================
    
    fig_static, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x_range = np.linspace(limit[0], limit[1], 1000)
    y_range = f.evaluate(x_range)
    
    axes[0].plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Objective Function', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    min_y_idx = np.argmin(y_range)
    axes[0].plot(x_range[min_y_idx], y_range[min_y_idx], 'r*', markersize=15, 
                label=f'Min: f({x_range[min_y_idx]:.2f}) = {y_range[min_y_idx]:.2f}')
    axes[0].legend(fontsize=10)
    
    axes[1].plot(x_range, y_range, 'b-', linewidth=2, alpha=0.6, label='f(x)')
    
    colors = ['red', 'orange', 'green']
    labels = ['Initial', 'Middle', 'Final']
    indices = [0, ger // 2, ger - 1]
    
    for idx, color, label in zip(indices, colors, labels):
        positions = np.array(x_history[idx])
        fitness = f.evaluate(positions)
        axes[1].scatter(positions, fitness, c=color, s=100, alpha=0.7, 
                       edgecolors='black', linewidth=1.5, label=label, zorder=5)
    
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('f(x)', fontsize=12)
    axes[1].set_title('Particle Movement on Objective Function', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    axes[1].plot(g_best, g_best_evaluate, 'm*', markersize=20, 
                label=f'Global Best: ({g_best:.2f}, {g_best_evaluate:.2f})', zorder=6)
    axes[1].legend(fontsize=9)
    
    generations = range(1, ger + 1)
    axes[2].plot(generations, g_best_history, 'g-', linewidth=2, label='Global Best Fitness')
    axes[2].set_xlabel('Generation', fontsize=12)
    axes[2].set_ylabel('Best Fitness', fontsize=12)
    axes[2].set_title('Convergence Curve', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11)
    
    axes[2].plot(ger, g_best_evaluate, 'ro', markersize=10, 
                label=f'Final: {g_best_evaluate:.4f}')
    axes[2].legend(fontsize=10)
    
    plt.tight_layout()

    
    # ==================== 绘制动态动画 ====================
    print('\n正在生成动态动画...')
    
    fig_anim, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    x_range = np.linspace(limit[0], limit[1], 1000)
    y_range = f.evaluate(x_range)
    
    # 上图：粒子移动
    ax1.plot(x_range, y_range, 'b-', linewidth=2, alpha=0.7, label='f(x)')
    scatter1 = ax1.scatter([], [], c='red', s=100, alpha=0.8, 
                          edgecolors='black', linewidth=1.5, zorder=5, label='Particles')
    best_point1 = ax1.plot([], [], 'm*', markersize=20, label='Global Best')[0]
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('PSO Particle Movement Animation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    text1 = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 下图：收敛曲线
    line2, = ax2.plot([], [], 'g-', linewidth=2, label='Global Best Fitness')
    best_point2 = ax2.plot([], [], 'ro', markersize=10, label='Current Best')[0]
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Best Fitness', fontsize=12)
    ax2.set_title('Convergence Process', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(0, ger)
    text2 = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    def init():
        """初始化动画"""
        scatter1.set_offsets(np.empty((0, 2)))
        best_point1.set_data([], [])
        line2.set_data([], [])
        best_point2.set_data([], [])
        text1.set_text('')
        text2.set_text('')
        return scatter1, best_point1, line2, best_point2, text1, text2
    
    def animate(frame):
        """动画更新函数"""
        current_gen = frame + 1
        
        # 更新上图：粒子位置
        positions = np.array(x_history[frame])
        fitness = f.evaluate(positions)
        points = np.column_stack([positions, fitness])
        scatter1.set_offsets(points)
        
        # 更新全局最优点
        best_x = positions[np.argmin(fitness)]
        best_y = np.min(fitness)
        best_point1.set_data([best_x], [best_y])
        
        # 更新文本信息
        text1.set_text(f'Generation: {current_gen}/{ger}\n'
                      f'Best x: {best_x:.4f}\n'
                      f'Best f(x): {best_y:.4f}')
        
        # 更新下图：收敛曲线
        gens = np.arange(1, current_gen + 1)
        fitness_history = np.array(g_best_history[:current_gen])
        line2.set_data(gens, fitness_history)
        best_point2.set_data([current_gen], [fitness_history[-1]])
        
        # 动态调整y轴范围
        ax2.set_ylim(np.min(fitness_history) - 10, np.max(fitness_history) + 10)
        
        text2.set_text(f'Current Generation: {current_gen}\n'
                      f'Best Fitness: {fitness_history[-1]:.4f}\n'
                      f'Improvement: {fitness_history[0] - fitness_history[-1]:.4f}')
        
        return scatter1, best_point1, line2, best_point2, text1, text2
    
    # 创建动画
    anim = FuncAnimation(fig_anim, animate, init_func=init,
                        frames=ger, interval=100, blit=True, repeat=False)

    plt.show()
