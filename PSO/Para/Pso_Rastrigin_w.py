"""
    固定 c1 = c2 = 2.0，对比ω = 0.9、ω = 0.7、ω = 0.4以及线性衰减ω = 0.9 → 0.4四条曲线，横轴迭代次数，纵轴gBest适应度。
"""
from cProfile import label

import numpy as np
import matplotlib.pyplot as plt


# Rastrigin 函数（多峰）
class ObjectiveFunction:

    def __call__(self, x):
        n = x.shape[1]
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def run_pso(w_strategy, N=50, dim=2, ger=100, c1=2.0, c2=2.0):
    # 1.定义函数
    f = ObjectiveFunction()

    # 2.定义参数
    N = N  # 种群粒子数量
    dim = dim  # 维度
    ger = ger  # 迭代次数
    limit = [-5, 5]  # 范围区间
    v_limit = [-5, 5]  # 速度区间
    # v_limit = [-10, 10]  # 速度区间
    c1 = c1  # 自我学习因子
    c2 = c2  # 群体学习因子

    # 3.初始化种群粒子
    # np.random.seed(42)
    x = np.random.uniform(limit[0], limit[1], (N, dim))
    v = np.random.uniform(v_limit[0], v_limit[1], (N, dim))
    p_best = np.copy(x)
    p_best_evaluate = f(x)
    g_best_idx = np.argmin(p_best_evaluate)
    g_best = np.copy(p_best[g_best_idx])
    g_best_evaluate = p_best_evaluate[g_best_idx]
    g_best_evaluate_his = []

    # 4.种群移动
    t = 0
    while t < ger:
        # 更新惯性权重
        if w_strategy == 'liner':
            w = 0.9 - (0.5 * t / ger)
        else:
            w = w_strategy
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
        g_best_evaluate_his.append(g_best_evaluate)
        # 轮次加1
        t += 1

    print(f'粒子历史最优位置：{p_best}\n最优适应度：{p_best_evaluate}')
    print(f'种群历史最优位置：{g_best}\n最优适应度：{g_best_evaluate}')
    return g_best_evaluate_his


if __name__ == '__main__':
    # 分别计算ω = 0.9、ω = 0.7、ω = 0.4以及线性衰减ω = 0.9 → 0.4
    ger = 100 # [100,1000]
    dim = 200 # [2,100]
    his_04 = run_pso(w_strategy=0.4, ger=ger,dim=dim)
    his_07 = run_pso(w_strategy=0.7, ger=ger,dim=dim)
    his_09 = run_pso(w_strategy=0.9, ger=ger,dim=dim)
    his_liner = run_pso(w_strategy='liner', ger=ger,dim=dim)
    # 5.绘图
    plt.figure(figsize=(10, 6))
    plt.plot(range(ger), his_04, label='$\omega = 0.4$')
    plt.plot(range(ger), his_07, label='$\omega = 0.7$')
    plt.plot(range(ger), his_09, label='$\omega = 0.9$')
    plt.plot(range(ger), his_liner, label='$\omega = 0.9 -> 0.4$')
    plt.title(f'Comparison of Different Inertia Weights $\omega$ on Rastrigin Function dim = {dim}')
    plt.xlabel('Iterations')
    plt.ylabel('gBest Fitness Score')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()
