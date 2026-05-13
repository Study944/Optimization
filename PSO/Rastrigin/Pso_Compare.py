"""
    对比手写 PSO 与 Python 自带 pyswarms
"""

from pyswarms.single import GlobalBestPSO
import numpy as np


# 多峰函数类
class ObjectiveFunction:

    def __call__(self, x):
        n = x.shape[1]
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


# 多峰函数
def rastrigin_func(x):
    n = x.shape[1]
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


if __name__ == '__main__':
    # 1.参数定义
    N = 100  # 种群粒子数量
    dim = 2  # 维度
    ger = 100  # 迭代次数
    limit = [-5, 5]  # 范围区间
    v_limit = [-1, 1]  # 速度区间
    w = 0.9  # 惯性系数
    c1 = 2.0  # 自我学习因子
    c2 = 2.0  # 群体学习因子

    # 2.GlobalBestPSO
    optimizer = GlobalBestPSO(
        n_particles=N,
        dimensions=dim,
        options={'c1': c1, 'c2': c2, 'w': w},
        bounds=(np.array([limit[0]] * dim), np.array([limit[1]] * dim)),
        velocity_clamp=(v_limit[0], v_limit[1]),
        vh_strategy="unmodified",
        bh_strategy="intermediate",
    )
    # cost, pos = optimizer.optimize(
    #     ,rastrigin_func
    #     iters=100,
    #     verbose=False
    # )
    # 3. 手动迭代循环
    # 初始化变量，防止未定义错误
    cost, pos = 0.0, np.zeros(dim)
    for i in range(ger):
        # 计算当前迭代的 w (线性递减公式)
        current_w = 0.9 - (0.9 - 0.4) * (i / ger)
        # 更新优化器中的选项
        optimizer.options['w'] = current_w
        cost, pos = optimizer.optimize(
            rastrigin_func,
            iters=1,
            verbose=False
        )

    print(f'pyswarms最优适应度: {cost:.11e}')
    print(f'pyswarms最优位置: [{pos[0]:.11e}, {pos[1]:.11e}]')

    # 3.手写
    f = ObjectiveFunction()
    x = np.random.uniform(limit[0], limit[1], (N, dim))
    v = np.random.uniform(v_limit[0], v_limit[1], (N, dim))
    p_best = np.copy(x)
    p_best_evaluate = f(x)
    g_best_idx = np.argmin(p_best_evaluate)
    g_best = np.copy(p_best[g_best_idx])
    g_best_evaluate = p_best_evaluate[g_best_idx]

    # 种群移动
    t = 0
    while t < ger:
        # 更新惯性权重
        w = 0.9 - (0.5 * t / ger)
        # 粒子移动
        r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        # 计算新位置
        v = np.clip(v, v_limit[0], v_limit[1])
        x = np.clip(x + v, limit[0], limit[1])
        # 计算适应值
        evaluate_new = f(x)
        # 更新粒子历史最优
        mask = evaluate_new < p_best_evaluate
        p_best[mask] = x[mask]
        p_best_evaluate[mask] = evaluate_new[mask]
        # 更新全局历史最优
        min_idx = np.argmin(evaluate_new)
        if np.min(p_best_evaluate) < g_best_evaluate:
            best_idx = np.argmin(p_best_evaluate)
            g_best = np.copy(p_best[best_idx])
            g_best_evaluate = p_best_evaluate[best_idx]
        # 轮次加1
        t += 1
    print(f'手写最优适应度: {g_best_evaluate}')
    print(f'手写最优位置: {g_best}')

    print('手写效果好' if g_best_evaluate < cost else 'pyswarms效果好')
