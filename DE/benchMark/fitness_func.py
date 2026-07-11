"""BenchMark CEC 2022测试函数
选取其中具有代表性的 6 个测试函数用于测试优化算法的性能
"""
import numpy as np


def f1_zakharov(x):
    """单峰函数（只有一个全局最小值），测试算法收敛速度和搜索能力
    最小值点：原点，最小值：0
    f = sum(x^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4
    """
    # 1.获取维度
    x = np.atleast_2d(x)
    D = x.shape[1]
    # 2.计算第一部分 sum(x^2)，axis = 1-->横向跨列计算
    sum1 = np.sum(x**2, axis=1)
    # 3.计算第二三部分基础参数
    i_arr = np.arange(1, D + 1)
    sum2 = np.sum(0.5 * i_arr * x, axis=1)
    # 4.返回结果
    res = sum1 + sum2 ** 2 + sum2 ** 4
    return res


def f2_rosenbrock(x):
    """香蕉函数，全局最小值位置一个狭长、弯曲的抛物线形山谷
    最小值点：所有维度值为 1 的点，最小值：0
    f = sum[100*(x_i+1 - x_i)^2 + (x_i - 1)^2]
    """
    # 1.获取 x_i 和 x_i+1
    x = np.atleast_2d(x)
    x_curr = x[:, :-1]
    x_next = x[:, 1:]
    # 2.计算矩阵和
    val = 100.0 * (x_next - x_curr ** 2) ** 2 + (x_curr - 1.0) ** 2
    res = np.sum(val, axis=1)
    return res


def f3_schaffer_f7(x):
    """基本多峰函数，具有强烈的波动性和频繁的局部陷阱
    最小值点：原点，最小值：0
    """
    x = np.atleast_2d(x)
    D = x.shape[1]

    x_curr = x[:, :-1]
    x_next = x[:, 1:]

    si = np.sqrt(x_curr ** 2 + x_next ** 2)
    tmp = np.sqrt(si) + np.sqrt(si) * (np.sin(50.0 * (si ** 0.2))) ** 2
    return (np.sum(tmp, axis=1) / (D - 1)) ** 2


def f4_non_continuous_rastrigin(x):
    """非连续多峰函数，在原本就布满无数局部极值点的 Rastrigin 函数基础上，引入了非连续化操作
    最小值点：原点，最小值：0
    """
    x = np.atleast_2d(x)

    # 非连续化操作
    x_hat = np.where(np.abs(x) <= 0.5, x, np.round(2 * x) / 2.0)

    val = x_hat ** 2 - 10.0 * np.cos(2.0 * np.pi * x_hat) + 10.0
    return np.sum(val, axis=1)


def f6_levy(x):
    """基本多峰函数，在边界处有大量的局部极小值，非常容易导致算法早熟收敛
    最小值点：所有维度值为 1 的点，最小值：0
    """
    x = np.atleast_2d(x)

    w = 1.0 + (x - 1.0) / 4.0
    w_1 = w[:, 0]
    w_D = w[:, -1]
    w_core = w[:, :-1]

    term1 = (np.sin(np.pi * w_1)) ** 2
    term2 = np.sum((w_core - 1.0) ** 2 * (1.0 + 10.0 * (np.sin(np.pi * w_core + 1.0)) ** 2), axis=1)
    term3 = (w_D - 1.0) ** 2 * (1.0 + (np.sin(2.0 * np.pi * w_D)) ** 2)

    return term1 + term2 + term3


def f9_hybrid_func1(x):
    """混合函数(Modified example: Griewank + Rastrigin)
    将变量空间切分成不同部分，分别应用不同的基础函数
    （如将前 N_1 个变量用 Griewank 优化，后 N_2 个变量用 Rastrigin 优化）
    模拟了现实世界中不同子系统具有不同物理特性的复杂场景。
    """
    x = np.atleast_2d(x)
    D = x.shape[1]
    p1 = D // 2

    x1 = x[:, :p1]
    x2 = x[:, p1:]

    # 子函数1: Griewank
    sum_g = np.sum(x1 ** 2 / 4000.0, axis=1)
    i_arr = np.arange(1, p1 + 1)
    prod_g = np.prod(np.cos(x1 / np.sqrt(i_arr)), axis=1)
    f_griewank = sum_g - prod_g + 1.0

    # 子函数2: Rastrigin
    f_rastrigin = np.sum(x2 ** 2 - 10.0 * np.cos(2.0 * np.pi * x2) + 10.0, axis=1)

    return f_griewank + f_rastrigin

def get_all_func():
    return [f1_zakharov, f2_rosenbrock, f3_schaffer_f7, f4_non_continuous_rastrigin, f6_levy, f9_hybrid_func1]