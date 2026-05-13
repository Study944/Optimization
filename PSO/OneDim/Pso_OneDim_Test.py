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
        v_i(t+1) = 惯性系数 * v_i(t) + c_1 * r1 * (g_best - x_i(t)) + c_2 * r2 * (p_best - x_i(t))
        x_i(t+1) = x_i(t) + v_i(t+1)
    实现：
        使用 PSO 计算 f(x) = x * sin(x) * cos(2x) − 2x * sin(3x) + 3x * sin(4x)在[0,50]的最小值。
"""

import numpy as np


class ObjectiveFunction:
    def __call__(self, x):
        return x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x)


if __name__ == '__main__':
    # 1.定义函数
    f = ObjectiveFunction()

    # 2.定义参数
    N = 20  # 种群粒子数量
    dim = 1  # 维度
    ger = 100  # 迭代次数
    limit = [0, 50]  # 范围区间
    v_limit = [-10, 10]  # 速度区间
    w = 0.9 # 惯性系数
    c1 = 2.0  # 自我学习因子
    c2 = 2.0  # 群体学习因子

    # 3.初始化种群粒子
    x = np.random.uniform(limit[0], limit[1], (N,dim))
    v = np.random.uniform(v_limit[0], v_limit[1], (N,dim))
    p_best = np.copy(x)
    p_best_evaluate = f(x)
    best_idx = np.argmin(p_best_evaluate)
    g_best = np.copy(p_best[best_idx])
    g_best_evaluate = p_best_evaluate[best_idx]

    # 4.种群移动
    t = 0
    while t < ger:
        # 更新惯性权重
        w = 0.9 - (0.5 * t / ger)
        # 4.1 粒子移动
        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        v = w * v + c1 * r1 * (p_best - x) + c2* r2 * (g_best - x)
        # 计算新位置
        v = np.clip(v, v_limit[0], v_limit[1])
        x = np.clip(x + v, limit[0], limit[1])
        # 计算适应值
        evaluate_new = f(x)
        # 4.2更新粒子历史最优
        # for i in range(N):
        #     if evaluate_new[i] < p_best_evaluate[i]:
        #         p_best[i] = x[i]
        #         p_best_evaluate[i] = evaluate_new[i]
        # 4.2向量优化更新粒子历史最优
        mask = evaluate_new < p_best_evaluate
        p_best[mask] = x[mask]
        p_best_evaluate[mask] = evaluate_new[mask]
        # 4.3更新全局历史最优
        min_idx = np.argmin(evaluate_new)
        if np.min(p_best_evaluate) < g_best_evaluate:
            best_idx = np.argmin(p_best_evaluate)
            g_best = np.copy(p_best[best_idx])
            g_best_evaluate = p_best_evaluate[best_idx]
        # 轮次加1
        t += 1

    print(f'粒子历史最优位置：{p_best}\n最优适应度：{p_best_evaluate}')
    print(f'种群历史最优位置：{g_best}\n最优适应度：{g_best_evaluate}')