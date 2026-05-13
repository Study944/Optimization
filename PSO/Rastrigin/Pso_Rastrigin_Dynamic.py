"""
    摘要：使用粒子群算法 PSO 寻找维度为 2 且区间有限的函数 f(x，y) 最小值。
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
        v_i(t+1) = 惯性系数 * v_i(t) + c_1 * r1 * (p_best - x_i(t)) + c_2 * r2 * (g_best - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Rastrigin 函数（多峰）
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


if __name__ == '__main__':
    # 1.定义函数
    f = ObjectiveFunction()

    # 2.定义参数
    N = 100  # 种群粒子数量
    dim = 2  # 维度
    ger = 100  # 迭代次数
    limit = [-5.12, 5.12]  # 范围区间
    v_limit = [-1, 1]  # 速度区间
    w = 0.9  # 惯性系数
    c1 = 2.0  # 自我学习因子
    c2 = 2.0  # 群体学习因子

    # 3.初始化种群粒子
    x = np.random.uniform(limit[0], limit[1], (N, dim))
    v = np.random.uniform(v_limit[0], v_limit[1], (N, dim))
    p_best = np.copy(x)
    p_best_evaluate = f(x)
    g_best_idx = np.argmin(p_best_evaluate)
    g_best = np.copy(p_best[g_best_idx])
    g_best_evaluate = p_best_evaluate[g_best_idx]

    # 记录历史数据用于绘图
    g_best_history = []
    x_history = []

    # 4.种群移动
    t = 0
    while t < ger:
        # 更新惯性权重
        w = 0.9 - (0.5 * t / ger)
        # 4.1 粒子移动
        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        # 计算新位置
        v = np.clip(v, v_limit[0], v_limit[1])
        x = np.clip(x + v, limit[0], limit[1])
        # 计算适应值
        evaluate_new = f(x)
        # 4.2更新粒子历史最优
        mask = evaluate_new < p_best_evaluate
        p_best[mask] = x[mask]
        p_best_evaluate[mask] = evaluate_new[mask]
        # 4.3更新全局历史最优
        if np.min(p_best_evaluate) < g_best_evaluate:
            best_idx = np.argmin(p_best_evaluate)
            g_best = np.copy(p_best[best_idx])
            g_best_evaluate = p_best_evaluate[best_idx]

        # 记录历史数据
        g_best_history.append(g_best_evaluate)
        x_history.append(x.copy())
        # 轮次加1
        t += 1

    print(f'粒子历史最优位置：{p_best}\n最优适应度：{p_best_evaluate}')
    print(f'种群历史最优位置：{g_best}\n最优适应度：{float(g_best_evaluate)}')

    # ==================== 生成函数曲面数据 ====================
    x_range = np.linspace(limit[0], limit[1], 200)
    y_range = np.linspace(limit[0], limit[1], 200)
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
    
    # 子图2：等高线图 + 粒子最终位置
    ax2 = fig1.add_subplot(132)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    contour_lines = ax2.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour, ax=ax2)
    
    # 显示最终粒子位置
    final_positions = np.array(x_history[-1])
    final_fitness = f(final_positions)
    scatter = ax2.scatter(final_positions[:, 0], final_positions[:, 1], 
                         c=final_fitness, s=100, alpha=0.7, 
                         edgecolors='black', linewidth=1.5, 
                         cmap='hot', label='Particles (Final)', zorder=5)
    
    # 标记全局最优位置
    ax2.plot(g_best[0], g_best[1], 'm*', markersize=25, 
            label=f'Global Best ({g_best[0]:.2f}, {g_best[1]:.2f})', zorder=6)
    
    # 标记理论最优位置
    ax2.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum (0, 0)', zorder=7)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Particle Distribution on Contour Map', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(limit[0], limit[1])
    ax2.set_ylim(limit[0], limit[1])
    
    # 子图3：收敛曲线
    ax3 = fig1.add_subplot(133)
    generations = range(1, ger + 1)
    ax3.plot(generations, g_best_history, 'g-', linewidth=2, label='Global Best Fitness')
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Best Fitness', fontsize=12)
    ax3.set_title('Convergence Curve', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5, 
               label='Theoretical Min (0)', alpha=0.7)
    ax3.plot(ger, g_best_evaluate, 'ro', markersize=10, 
            label=f'Final: {float(g_best_evaluate):.6f}')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # ==================== 图2：粒子在不同代数的位置分布 ====================
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    
    colors = ['red', 'orange', 'green']
    labels = ['Initial', 'Middle', 'Final']
    indices = [0, ger // 2, ger - 1]
    
    for idx, color, label in zip(indices, colors, labels):
        positions = np.array(x_history[idx])
        fitness = f(positions)
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                            c=color, s=80, alpha=0.6, 
                            edgecolors='black', linewidth=1.2, 
                            label=label, zorder=5)
    
    # 标记全局最优位置
    ax.plot(g_best[0], g_best[1], 'm*', markersize=25, 
           label=f'Global Best', zorder=6)
    
    # 标记理论最优位置
    ax.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum (0, 0)', zorder=7)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Particle Movement Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(limit[0], limit[1])
    ax.set_ylim(limit[0], limit[1])
    
    plt.tight_layout()

    plt.show()

    # ==================== 图3：动态动画 ====================
    print('\n正在生成动态动画...')
    
    fig_anim, (ax1_anim, ax2_anim) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图：等高线图 + 粒子移动
    contour_anim = ax1_anim.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    contour_lines = ax1_anim.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3, alpha=0.4)
    fig_anim.colorbar(contour_anim, ax=ax1_anim, shrink=0.8)
    
    scatter_anim = ax1_anim.scatter([], [], c='red', s=120, alpha=0.8, 
                                   edgecolors='black', linewidth=1.5, 
                                   label='Particles', zorder=5)
    best_point_anim = ax1_anim.plot([], [], 'm*', markersize=25, label='Global Best')[0]
    theoretical_point = ax1_anim.plot(0, 0, 'r*', markersize=20, label='Theoretical Optimum')[0]
    
    ax1_anim.set_xlabel('X', fontsize=12)
    ax1_anim.set_ylabel('Y', fontsize=12)
    ax1_anim.set_title('PSO Particle Movement Animation (Rastrigin)', fontsize=14, fontweight='bold')
    ax1_anim.legend(fontsize=10, loc='upper right')
    ax1_anim.grid(True, alpha=0.3)
    ax1_anim.set_xlim(limit[0], limit[1])
    ax1_anim.set_ylim(limit[0], limit[1])
    
    text1_anim = ax1_anim.text(0.02, 0.98, '', transform=ax1_anim.transAxes, fontsize=10,
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # 右图：收敛曲线
    line_anim, = ax2_anim.plot([], [], 'g-', linewidth=2.5, label='Global Best Fitness')
    best_point_anim2 = ax2_anim.plot([], [], 'ro', markersize=10, label='Current Best')[0]
    theoretical_line = ax2_anim.axhline(y=0, color='r', linestyle='--', linewidth=1.5, 
                                       label='Theoretical Min (0)', alpha=0.7)
    
    ax2_anim.set_xlabel('Generation', fontsize=12)
    ax2_anim.set_ylabel('Best Fitness', fontsize=12)
    ax2_anim.set_title('Convergence Process', fontsize=14, fontweight='bold')
    ax2_anim.legend(fontsize=10)
    ax2_anim.grid(True, alpha=0.3)
    ax2_anim.set_xlim(0, ger)
    
    text2_anim = ax2_anim.text(0.02, 0.98, '', transform=ax2_anim.transAxes, fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    def init():
        """初始化动画"""
        scatter_anim.set_offsets(np.empty((0, 2)))
        best_point_anim.set_data([], [])
        line_anim.set_data([], [])
        best_point_anim2.set_data([], [])
        text1_anim.set_text('')
        text2_anim.set_text('')
        return scatter_anim, best_point_anim, line_anim, best_point_anim2, text1_anim, text2_anim
    
    def animate(frame):
        """动画更新函数"""
        current_gen = frame + 1
        
        # 更新左图：粒子位置
        positions = np.array(x_history[frame])
        fitness = f(positions)
        scatter_anim.set_offsets(positions)
        
        # 更新全局最优点
        best_idx = np.argmin(fitness)
        best_x = positions[best_idx, 0]
        best_y = positions[best_idx, 1]
        best_point_anim.set_data([best_x], [best_y])
        
        # 更新文本信息
        text1_anim.set_text(f'Generation: {current_gen}/{ger}\n'
                           f'Particles: {N}\n'
                           f'Current Best:\n'
                           f'  x={best_x:.4f}, y={best_y:.4f}\n'
                           f'  f(x,y)={np.min(fitness):.6f}')
        
        # 更新右图：收敛曲线
        gens = np.arange(1, current_gen + 1)
        fitness_history = np.array(g_best_history[:current_gen])
        line_anim.set_data(gens, fitness_history)
        best_point_anim2.set_data([current_gen], [fitness_history[-1]])
        
        # 动态调整y轴范围
        margin = abs(np.max(fitness_history) - np.min(fitness_history)) * 0.1
        if margin < 1:
            margin = 1
        ax2_anim.set_ylim(np.min(fitness_history) - margin, np.max(fitness_history) + margin)
        
        text2_anim.set_text(f'Current Generation: {current_gen}\n'
                           f'Best Fitness: {fitness_history[-1]:.6f}\n'
                           f'Improvement: {fitness_history[0] - fitness_history[-1]:.6f}\n'
                           f'Theoretical Min: 0.0')
        
        return scatter_anim, best_point_anim, line_anim, best_point_anim2, text1_anim, text2_anim
    
    # 创建动画
    anim = FuncAnimation(fig_anim, animate, init_func=init,
                        frames=ger, interval=100, blit=True, repeat=False)
    
    print('动画生成完成！')
    plt.show()
