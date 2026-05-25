"""模拟退火算法参数对比实验 (修复版 - 使用并行版本)"""

import numpy as np
import matplotlib.pyplot as plt
# 导入并行版本的 SA_Parallel
from Simulated_Annealing_Parallel import SA_Parallel


def rastrigin(x):
    """Rastrigin 函数"""
    if np.any(x < -5.12) or np.any(x > 5.12):
        return 1e6  # 边界外给予极大的惩罚值
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def experiment_alpha_comparison():
    """实验1：不同 alpha 值的对比试验"""
    print("=" * 60)
    print("实验1：不同冷却率 α 的对比试验（并行版本）")
    print("=" * 60)

    np.random.seed(42)
    alphas = [0.85, 0.90, 0.95, 0.99]
    results = {}

    dim = 2
    limit = [[-5.12, 5.12], [-5.12, 5.12]]
    m_seeds = 5

    for alpha in alphas:
        print(f"\n正在运行 α = {alpha} 的实验...")
        sa = SA_Parallel(dim=dim, alpha=alpha, p_start=0.85, L=100, m_seeds=m_seeds)
        x_best, fitness_best = sa.optimize(rastrigin, limit)

        results[alpha] = {
            'fitness_history': np.array(sa.fitness_history),
            'x_best': x_best,
            'fitness_best': fitness_best,
            'total_steps': len(sa.fitness_history)
        }

        print(f"  最优解: {x_best}")
        print(f"  最优值: {fitness_best:.6f}")
        print(f"  迭代次数: {results[alpha]['total_steps']}")

    plot_alpha_comparison(results, alphas)
    return results


def experiment_pstart_comparison():
    """实验2：不同 P_start 值的对比试验"""
    print("\n" + "=" * 60)
    print("实验2：不同初始接受概率 P_start 的对比试验（并行版本）")
    print("=" * 60)

    np.random.seed(42)
    p_starts = [0.7, 0.85, 0.95]
    results = {}

    dim = 2
    limit = [[-5.12, 5.12], [-5.12, 5.12]]
    m_seeds = 5

    for p_start in p_starts:
        print(f"\n正在运行 P_start = {p_start} 的实验...")
        sa = SA_Parallel(dim=dim, alpha=0.98, p_start=p_start, L=100, m_seeds=m_seeds)
        x_best, fitness_best = sa.optimize(rastrigin, limit)

        results[p_start] = {
            'fitness_history': np.array(sa.fitness_history),
            'x_best': x_best,
            'fitness_best': fitness_best,
            'total_steps': len(sa.fitness_history)
        }

        print(f"  最优解: {x_best}")
        print(f"  最优值: {fitness_best:.6f}")
        print(f"  迭代次数: {results[p_start]['total_steps']}")

    plot_pstart_comparison(results, p_starts)
    return results


def plot_alpha_comparison(results, alphas):
    """优雅版：修复乱码与画布挤压问题"""
    plt.figure(figsize=(10, 5.5), dpi=150)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    all_vals = []

    for i, alpha in enumerate(alphas):
        fitness_history = results[alpha]['fitness_history']
        steps = np.arange(len(fitness_history))
        all_vals.extend(fitness_history)

        plot_history = fitness_history + 1e-6
        plt.plot(steps, plot_history,
                 color=colors[i],
                 linewidth=1.8,
                 label=f'冷却率 $\\alpha$ = {alpha}',
                 alpha=0.85)

    plt.axhline(y=0 + 1e-6, color='#d62728', linestyle='--', alpha=0.6, label='理论全局最优值 (0)')

    min_val = max(1e-4, min(all_vals) * 0.5)
    max_val = max(all_vals) * 2
    plt.ylim(min_val, max_val)
    plt.yscale('log')

    plt.title('不同冷却率 ($\\alpha$) 下的适应度收敛曲线对比\n(并行版本, m_seeds=5)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('迭代步数 (Evaluation Steps)', fontsize=10)
    plt.ylabel('适应度函数值 (Fitness, Log Scale)', fontsize=10)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


def plot_pstart_comparison(results, p_starts):
    """优雅版：修复乱码与画布挤压问题"""
    plt.figure(figsize=(10, 5.5), dpi=150)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    all_vals = []

    for i, p_start in enumerate(p_starts):
        fitness_history = results[p_start]['fitness_history']
        steps = np.arange(len(fitness_history))
        all_vals.extend(fitness_history)

        plot_history = fitness_history + 1e-6
        label_text = f'$P_{{start}}$ = {p_start}'
        plt.plot(steps, plot_history,
                 color=colors[i],
                 linewidth=1.8,
                 label=label_text,
                 alpha=0.85)

    plt.axhline(y=0 + 1e-6, color='#d62728', linestyle='-', alpha=0.6, label='理论全局最优值 (0)')

    min_val = max(1e-4, min(all_vals) * 0.5)
    max_val = max(all_vals) * 2
    plt.ylim(min_val, max_val)
    plt.yscale('log')

    plt.title('不同初始接受概率 ($P_{{start}}$) 下的适应度收敛曲线对比\n(并行版本, m_seeds=5)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('迭代步数 (Evaluation Steps)', fontsize=10)
    plt.ylabel('适应度函数值 (Fitness, Log Scale)', fontsize=10)
    plt.grid(True, which="both", linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ✨ 核心配置：在入口处一次性完成最强力的乱码与负号全局初始化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 强制使用标准减号，彻底消灭方块乱码
    plt.rcParams['mathtext.fontset'] = 'cm'     # 数学标号采用标准 LaTeX 计算机现代字体

    # 运行对比实验
    alpha_results = experiment_alpha_comparison()
    pstart_results = experiment_pstart_comparison()

