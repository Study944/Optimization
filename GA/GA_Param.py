import numpy as np
import matplotlib.pyplot as plt

from GA.GA_Pic import MyGA


def plot_comparison_curves(experiments_data, n_generation, title="Parameter Comparison"):
    """在一个子图中绘制多组参数实验的收敛曲线进行对比

    Args:
        experiments_data: 一个列表，每个元素为一个字典，包含单个实验的数据：
                          [{"label": "P_c=0.5", "best": [...], "avg": [...]}, ...]
        n_generation: 总迭代次数
        title: 图表标题
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    generations = range(1, n_generation + 1)

    # 定义一套颜色和线型，确保对比清晰
    colors = ['g', 'b', 'r', 'c', 'm', 'y']

    for i, data in enumerate(experiments_data):
        color = colors[i % len(colors)]
        label = data["label"]

        # 绘制最佳适应度曲线 (实线)
        ax.plot(generations, data["best"], linestyle='-', color=color, linewidth=2,
                label=f'{label} (Best)')
        # 绘制平均适应度曲线 (虚线)
        ax.plot(generations, data["avg"], linestyle='--', color=color, linewidth=1.2, alpha=0.6,
                label=f'{label} (Avg)')

        # 在最终代标出终点圆点
        final_fitness = data["best"][-1]
        ax.plot(n_generation, final_fitness, marker='o', color=color, markersize=8)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 将图例放在外面或者合适的位置防止遮挡曲线
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.show()


def FitnessFunction(x):
    """Rastrigin适应度函数"""
    n = x.shape[1]
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


if __name__ == '__main__':
    limits = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    fixed_n_generation = 100

    # =========================================================================
    # 实验一：P_c 交叉率参数对比实验 [0.3, 0.6, 0.9] (默认 P_m=0.01)
    # =========================================================================
    pc_experiments = []
    pc_list = [0.3, 0.6, 0.9]

    print("开始 P_c 交叉率实验...")
    for pc in pc_list:
        ga = MyGA(dim=2, cross_rate=pc, n_generation=fixed_n_generation)
        ga.optimize(FitnessFunction, limits)
        # 收集该实验的数据
        pc_experiments.append({
            "label": f"P_c={pc}",
            "best": list(ga.best_fitness_history),
            "avg": list(ga.avg_fitness_history)
        })
        # 将三组 P_c 实验绘制在同一个画板上
    plot_comparison_curves(pc_experiments, fixed_n_generation, title="Crossover Rate (P_c) Comparison (P_m=0.01)")

    # =========================================================================
    # 实验二：P_m 变异率参数对比实验 [0.1, 0.01, 0.001] (默认 P_c=0.8)
    # =========================================================================
    pm_experiments = []
    pm_list = [0.1, 0.01, 0.001]

    print("\n开始 P_m 变异率实验...")
    for pm in pm_list:
        ga = MyGA(dim=2, mutation_rate=pm, n_generation=fixed_n_generation)
        ga.optimize(FitnessFunction, limits)

        # 收集该实验的数据
        pm_experiments.append({
            "label": f"P_m={pm}",
            "best": list(ga.best_fitness_history),
            "avg": list(ga.avg_fitness_history)
        })
    plot_comparison_curves(pm_experiments, fixed_n_generation, title="Mutation Rate (P_m) Comparison (P_c=0.8)")
