import numpy as np


class MyPSO:
    # 初始化
    def __init__(self, n=20, dim=2, limit=None):
        self.n = n
        self.dim = dim
        self.limit = limit if limit is not None else [-5, 5]

    # PSO
    def optimizer(self, objective_func, iters=100):
        # 1.定义函数
        f = objective_func
        # 2.定义参数
        N = self.n  # 种群粒子数量
        dim = self.dim  # 维度
        iters = iters  # 迭代次数
        limit = self.limit  # 范围区间
        v_limit = [-1, 1]  # 速度区间
        c1 = 2.0  # 自我学习因子
        c2 = 2.0  # 群体学习因子

        # 3.初始化种群粒子
        if np.ndim(limit) == 1:
            # 统一范围：所有维度使用相同的范围
            x = np.random.uniform(limit[0], limit[1], (N, dim))
            limit_min = limit[0]
            limit_max = limit[1]
        else:
            # 各维度独立范围：每个维度有自己的[min, max]
            limit = np.array(limit)
            limit_min = limit[:, 0]  # 每个维度的最小值
            limit_max = limit[:, 1]  # 每个维度的最大值
            # 为每个粒子生成不同范围的随机值
            x = np.random.uniform(
                limit_min[np.newaxis, :],  # 广播到(N, dim)
                limit_max[np.newaxis, :],
                (N, dim)
            )
        v = np.random.uniform(v_limit[0], v_limit[1], (N, dim))
        p_best = np.copy(x)
        p_best_evaluate = np.array([f(x[i]) for i in range(N)])
        g_best_idx = np.argmin(p_best_evaluate)
        g_best = np.copy(p_best[g_best_idx])
        g_best_evaluate = p_best_evaluate[g_best_idx]

        # 4.种群移动
        t = 0
        while t < iters:
            # 更新惯性权重
            w = 0.9 - (0.5 * t / iters)
            # 4.1 粒子移动
            r1, r2 = np.random.rand(N, dim), np.random.rand(N, dim)
            v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
            # 计算新位置
            v = np.clip(v, v_limit[0], v_limit[1])
            x = x + v

            # 边界处理：根据limit类型选择裁剪方式
            if np.ndim(self.limit) == 1:
                x = np.clip(x, limit_min, limit_max)
            else:
                x = np.clip(x, limit_min[np.newaxis, :], limit_max[np.newaxis, :])

            # 计算适应值
            evaluate_new = np.array([f(x[i]) for i in range(N)])
            # 4.2更新粒子历史最优
            mask = evaluate_new < p_best_evaluate
            p_best[mask] = x[mask]
            p_best_evaluate[mask] = evaluate_new[mask]
            # 4.3更新全局历史最优
            if np.min(p_best_evaluate) < g_best_evaluate:
                best_idx = np.argmin(p_best_evaluate)
                g_best = np.copy(p_best[best_idx])
                g_best_evaluate = p_best_evaluate[best_idx]
            # 轮次加1
            t += 1

        # print(f'粒子历史最优位置：{p_best}\n最优适应度：{p_best_evaluate}')
        # print(f'种群历史最优位置：{g_best}\n最优适应度：{g_best_evaluate}')

        return g_best, g_best_evaluate
