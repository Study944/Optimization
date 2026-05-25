""" 第三方库实现 SA"""

from scipy.optimize import dual_annealing
import numpy as np


def func(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    bounds = [[-5.12,5.12],[-5.12,5.12]]

    res = dual_annealing(
        func,
        bounds=bounds,
        seed=5
    )
    print(f'最小值位置：{res.x}')
    print(f'最佳适应值：{res.fun}')