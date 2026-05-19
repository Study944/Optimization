import random
import numpy as np
from deap import base, creator, tools, algorithms

# 1. 定义问题：最大化适应度
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 参数设置
DIM = 2  # 维度
DNA_SIZE = 20  # 每个维度的二进制位数
POP_SIZE = 100  # 种群数量
N_GEN = 100  # 迭代次数
LIMIT = [-5.12, 5.12]  # 变量范围

toolbox = base.Toolbox()

# 2. 编码实现：注册二进制基因生成器
toolbox.register("attr_bin", random.randint, 0, 1)
# 个体由 DIM * DNA_SIZE 个二进制位组成
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bin, n=DNA_SIZE * DIM)
# 种群由多个个体组成
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def decode(individual):
    """解码逻辑：将二进制个体映射回连续空间的实数向量
    二进制-->十进制
    """
    x = []
    max_decimal = 2 ** DNA_SIZE - 1
    for i in range(DIM):
        # 截取对应维度的基因段
        start = i * DNA_SIZE
        end = (i + 1) * DNA_SIZE
        gene_segment = individual[start:end]
        # 二进制转十进制
        decimal_d = 0
        for bit in gene_segment:
            decimal_d = (decimal_d << 1) | bit
        # 映射公式：L + (D / (2^n - 1)) * (R - L)
        res = LIMIT[0] + (decimal_d / max_decimal) * (LIMIT[1] - LIMIT[0])
        x.append(res)
    return np.array(x)


def evalRastrigin(individual):
    """适应度评估：包含解码过程"""
    x = decode(individual)
    n = len(x)
    # 计算Rastrigin函数值（原本连续空间的数据）
    fitness_value = 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    return (fitness_value,)


# 3. 注册遗传算子
toolbox.register("evaluate", evalRastrigin)
toolbox.register("mate", tools.cxTwoPoint)  # 二进制常用两点交叉
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)  # 二进制位翻转变异，
# toolbox.register("select", tools.selRoulette)   # 轮盘赌
toolbox.register("select", tools.selTournament, tournsize=3)  # 锦标赛选择（更通用）


def main():
    # 初始化种群
    population = toolbox.population(n=POP_SIZE)

    # 精英保留：HallOfFame 自动保留历史最优个体
    hof = tools.HallOfFame(1)

    # 统计插件
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 4. 运行进化算法（eaSimple）
    pop, log = algorithms.eaSimple(population, toolbox,
                                   cxpb=0.8,  # 对应 cross_rate
                                   mutpb=0.1,  # 个体变异概率
                                   ngen=N_GEN,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    # 输出结果
    best_ind = hof[0]
    best_x = decode(best_ind)
    print("-" * 30)
    print(f"最优解编码: {best_ind}")
    print(f"解码后的最优坐标: {best_x}")
    print(f"最优适应度: {best_ind.fitness.values[0]:.6f}")

    return pop, log, hof


if __name__ == "__main__":
    main()