"""遗传算法
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
        (3).遗传算子 ：遗传算法的核心，实行优胜劣汰机制保留优解，优解之间交叉基因生成后代，部分解发生变异。
            6.选择(轮盘赌) ：根据每个解的适应度函数结果好坏计算每个解的概率，随机 N 次概率选择后代
            7.交叉 ：按照交叉概率 c ，任意选择 N·c 个解进行基因交叉产生后代，后代直接代替原本解，保证解数量为 N
            8.变异 ：按照变异概率 m ，每个解都有概率 m 发生变异，变异后该解随机一位进行取反，为了跳出局部最优
    案例 ：
        实现实现遗传算法寻找 Rastrigin 多峰函数最大值
"""

import numpy as np


class MyGA:
    """ mini 遗传算法类
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
        """计算适应值
            不允许存在负数，若存在负数，考虑整体上移
        """
        pred = fitness_function(x)
        # 核心：为了增强选择压，减去最小值。1e-3 防止 sum 为 0
        return pred - np.min(pred) + 1e-3

    def _select(self, fitness, pop):
        """选择（轮盘赌）
            按照概率选择保留的解
        """
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size,
                               replace=True, p=fitness / fitness.sum())
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
                cross_point = np.random.randint(0, self.dna_size * self.dim)
                child[cross_point:] = mother[cross_point:]
            # 变异
            if np.random.rand() < self.mutation_rate:
                mutate_point = np.random.randint(0, self.dna_size * self.dim)
                child[mutate_point] ^= 1
            new_pop.append(child)
        return np.array(new_pop)

    def optimize(self, fitness_function, limit):
        """ 算法实现
        """
        # 1.参数初始化
        # np.random.seed(42)
        pop = np.random.randint(2, size=(self.pop_size, self.dna_size * self.dim))
        # 2.迭代计算
        for i in range(self.n_generation):
            # 3.DNA解码
            x = self._translate_dna(pop, limit)
            # 4.计算适应值
            fitness = self._get_fitness(fitness_function, x)
            # 5.轮盘赌
            pop = self._select(fitness, pop)
            # 6.交叉变异
            pop = self._evolve(pop)
        # 7.寻找最优解
        final_x = self._translate_dna(pop, limit)
        final_fit = fitness_function(final_x)
        return final_x[np.argmax(final_fit)]


def FitnessFunction(x):
    """Rastrigin适应度函数
    """
    n = x.shape[1]
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


if __name__ == '__main__':
    # 维度为2测试结果
    ga = MyGA(dim=2)
    limits = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    ga_optimize = ga.optimize(FitnessFunction, limits)
    print(f"最终找到的最优解坐标: {ga_optimize}")
    print(f"对应的函数最大值: {FitnessFunction(ga_optimize.reshape(1, -1))[0]}")
