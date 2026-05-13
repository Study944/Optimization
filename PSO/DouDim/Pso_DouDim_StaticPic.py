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
    实现：
        使用 PSO 计算 f(x,y) = 20 + x**2 + y**2 − 10*cos(2𝜋x) − 10*cos(2𝜋𝑦)在[−5.12,5.12]的最小值
"""

import numpy as np
import matplotlib.pyplot as plt

class ObjectiveFunction:
    # 目标函数类
    def __init__(self):
        pass

    def __call__(self, x, y):
        """
        计算函数值
        f(x,y) = 20 + x**2 + y**2 − 10*cos(2𝜋x) − 10*cos(2𝜋𝑦)
        """
        return 20 + x ** 2 + y ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y)

    def evaluate(self, x, y):
        return self.__call__(x, y)

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
    p_best_evaluate = f.evaluate(x[:, 0], x[:, 1])
    g_best_idx = np.argmin(p_best_evaluate)
    g_best = np.copy(p_best[g_best_idx])
    g_best_evaluate = p_best_evaluate[g_best_idx]

    # 记录历史数据用于绘图
    g_best_history = []
    x_history = []  # 记录所有粒子的位置历史

    # 4.种群移动
    t = 0
    while t < ger:
        # 4.1 粒子移动
        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        # 计算新位置
        v = np.clip(v, v_limit[0], v_limit[1])
        x_new = x + v
        x_new = np.clip(x_new, limit[0], limit[1])
        # 计算适应值
        evaluate_new = f.evaluate(x_new[:, 0], x_new[:, 1])
        # 4.2更新粒子历史最优
        for i in range(N):
            if evaluate_new[i] < p_best_evaluate[i]:
                p_best[i] = x_new[i]
                p_best_evaluate[i] = evaluate_new[i]
        x = x_new
        # 4.3更新全局历史最优
        min_idx = np.argmin(evaluate_new)
        if np.min(p_best_evaluate) < g_best_evaluate:
            best_idx = np.argmin(p_best_evaluate)
            g_best = np.copy(p_best[best_idx])
            g_best_evaluate = p_best_evaluate[best_idx]
        # 记录历史数据
        g_best_history.append(g_best_evaluate)
        x_history.append(x.copy())
        
        # 轮次加1
        t += 1
        # 同步更新惯性权重
        w = 0.9 - (0.5 * t / ger)

    print(f'粒子历史最优位置：{p_best}\n最优适应度：{p_best_evaluate}')
    print(f'种群历史最优位置：{g_best}\n最优适应度：{g_best_evaluate}')

    # ==================== 绘图 ====================
    
    # 生成函数曲面数据
    x_range = np.linspace(limit[0], limit[1], 200)
    y_range = np.linspace(limit[0], limit[1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f.evaluate(X, Y)
    
    # 图1：3D目标函数曲面图
    fig1 = plt.figure(figsize=(14, 6))
    
    ax1 = fig1.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('f(X, Y)', fontsize=12)
    ax1.set_title('3D Objective Function Surface', fontsize=14, fontweight='bold')
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # 标记理论最小值点 (0, 0, 0)
    ax1.scatter([0], [0], [0], color='red', s=200, marker='*', 
               label='Theoretical Min (0, 0, 0)', zorder=5)
    ax1.legend(fontsize=10)
    
    # 图2：等高线图 + 粒子最终位置
    ax2 = fig1.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    contour_lines = ax2.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    fig1.colorbar(contour, ax=ax2)
    
    # 显示最终粒子位置
    final_positions = np.array(x_history[-1])
    final_fitness = f.evaluate(final_positions[:, 0], final_positions[:, 1])
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
    
    plt.tight_layout()
    plt.show()
    
    # 图3：收敛曲线
    fig2, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    generations = range(1, ger + 1)
    axes[0].plot(generations, g_best_history, 'g-', linewidth=2, label='Global Best Fitness')
    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('Best Fitness', fontsize=12)
    axes[0].set_title('Convergence Curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, 
                   label='Theoretical Min (0)', alpha=0.7)
    axes[0].legend(fontsize=10)
    
    # 标记最终值
    axes[0].plot(ger, g_best_evaluate, 'ro', markersize=10, 
                label=f'Final: {g_best_evaluate:.6f}')
    axes[0].legend(fontsize=10)
    
    # 图4：粒子在不同代数的位置分布
    axes[1].contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    
    colors = ['red', 'orange', 'green']
    labels = ['Initial', 'Middle', 'Final']
    indices = [0, ger // 2, ger - 1]
    
    for idx, color, label in zip(indices, colors, labels):
        positions = np.array(x_history[idx])
        fitness = f.evaluate(positions[:, 0], positions[:, 1])
        scatter = axes[1].scatter(positions[:, 0], positions[:, 1], 
                                 c=color, s=80, alpha=0.6, 
                                 edgecolors='black', linewidth=1.2, 
                                 label=label, zorder=5)
    
    # 标记全局最优位置
    axes[1].plot(g_best[0], g_best[1], 'm*', markersize=25, 
                label=f'Global Best', zorder=6)
    
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title('Particle Movement Evolution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(limit[0], limit[1])
    axes[1].set_ylim(limit[0], limit[1])
    
    plt.tight_layout()
    plt.show()
    
