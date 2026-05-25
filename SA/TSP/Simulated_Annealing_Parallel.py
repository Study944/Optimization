"""模拟退火算法(并行) TSP问题"""

import numpy as np


class SA_Parallel:
    """模拟退火算法(并行)"""

    def __init__(self, L=100, p_start=0.85, t_end=1e-6, alpha=0.98, m_seeds=1):
        """初始化"""
        self.L = L
        self.p_start = p_start
        self.num_cities = None
        self.t_end = t_end
        self.alpha = alpha
        self.fitness_history = None
        self.m_seeds = m_seeds

    def _template_init(self, dist_matrix):
        """
        初始化温度 T_start
            T_start = -ΔE_avg/lnP_start
        """
        # 1.迭代 100 轮计算 ΔE
        delta_E_list = []
        for i in range(100):
            x1 = np.random.permutation(self.num_cities)
            x2 = np.random.permutation(self.num_cities)
            delta_E = abs(self.get_single_path_distance(x1, dist_matrix) -
                          self.get_single_path_distance(x2, dist_matrix))
            delta_E_list.append(delta_E)
        # 2.计算 ΔE_avg ，代入计算 T_start
        delta_E_avg = np.mean(delta_E_list)
        t_start = -delta_E_avg / np.log(self.p_start)
        return t_start

    def get_single_path_distance(self, path, dist_matrix):
        """ 计算单条 TSP 路径的回路总长度 """
        # total_distance = 0
        # num_cities = len(path)
        #
        # # 1. 累加城市之间的去程距离
        # for i in range(num_cities - 1):
        #     city_current = path[i]
        #     city_next = path[i + 1]
        #     total_distance += dist_matrix[city_current, city_next]
        #
        # # 2. 加上收尾返程：从最后一个城市回到起点城市
        # total_distance += dist_matrix[path[-1], path[0]]
        #
        # return total_distance

        go_distance = np.sum(dist_matrix[path[:-1], path[1:]])
        back_distance = dist_matrix[path[-1], path[0]]
        return go_distance + back_distance

    def _get_next(self, x, step_size):
        """
        计算新解位置
            策略1：交换算子，交换任意两个城市位置
            策略2：逆序算子，逆序随机一个区间的城市（使用）
            策略3：插入算子，随机拔出一个城市，随机插入到路径
        """
        max_span = max(15, int(step_size))
        x_new = np.copy(x)

        # 交换算子0.2+逆序算子0.8
        for s in range(self.m_seeds):
            if np.random.rand() < 0.8:
                # ------ 策略 1：逆序算子（保持你测试出的保底 15 跨度） ------
                span = np.random.randint(2, max_span + 1)
                start_idx = np.random.randint(0, self.num_cities - span + 1)
                end_idx = start_idx + span
                x_new[s, start_idx:end_idx] = x_new[s, start_idx:end_idx][::-1]
            else:
                # ------ 策略 2：基于步长约束的交换算子 ------
                # 随机选第一个城市
                idx1 = np.random.randint(0, self.num_cities)
                # 第二个城市不能离得太远，受当前步长限制（后期退化为近邻交换，极其精准）
                max_swap_dist = max(2, int(step_size * 0.5))
                swap_dist = np.random.randint(1, max_swap_dist + 1)

                # 环形边界处理（防止越界）
                idx2 = (idx1 + swap_dist) % self.num_cities

                # 执行硬交换
                x_new[s, idx1], x_new[s, idx2] = x_new[s, idx2], x_new[s, idx1]

        return x_new

    def optimize(self, dist_matrix):
        """算法逻辑"""
        # 1.初始化原始路径
        self.num_cities = dist_matrix.shape[0]
        x = np.array([np.random.permutation(self.num_cities) for _ in range(self.m_seeds)])
        # 初始化温度
        template = self._template_init(dist_matrix)
        template_start = template

        # 全局历史最优记录 (从所有种子中挑路径最短的)
        dist = np.array([self.get_single_path_distance(path, dist_matrix) for path in x])
        best_idx = np.argmin(dist)
        x_best = np.copy(x[best_idx])
        dist_best = dist[best_idx]

        # 用于画图的收敛历史
        self.fitness_history = [dist_best]

        # 2.迭代计算
        while template > self.t_end:
            # 外层循环控制 步长
            cur_step_size = (self.num_cities * 0.5) * (template / template_start)

            for _ in range(self.L):
                # 内层更新位置
                x_new = self._get_next(x, cur_step_size)

                # 计算所有种子的 ΔE
                dist_new = np.array([self.get_single_path_distance(path, dist_matrix) for path in x_new])
                delta_E = dist_new - dist

                # 对每一个独立的退火进行判定
                for s in range(self.m_seeds):
                    if delta_E[s] < 0:
                        x[s] = x_new[s]
                        dist[s] = dist_new[s]
                        # 更新全局最优
                        if dist_new[s] < dist_best:
                            x_best = np.copy(x_new[s])
                            dist_best = dist_new[s]
                    else:
                        p = np.exp(-delta_E[s] / template)
                        if np.random.rand() < p:
                            x[s] = x_new[s]
                            dist[s] = dist_new[s]

                # 记录当前这一步的全局最优历史，平滑
                # self.fitness_history.append(dist_best)
                # 记录当前这一步，整个种群里实时表现最好的个体的适应度，震荡
                current_population_best = np.min(dist)
                self.fitness_history.append(current_population_best)

            # 更新温度
            template *= self.alpha

        return x_best, dist_best


if __name__ == "__main__":
    # 1. berlin52数据集 52个坐标
    coords = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                       [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                       [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                       [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                       [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                       [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                       [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                       [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                       [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                       [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
                       [1340.0, 725.0], [1740.0, 245.0]])

    # 2. 一次性预计算距离矩阵 (C语言级矩阵广播，速度极快)
    num_cities = len(coords)
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1))

    # 3. 实例化算法
    sa_tsp = SA_Parallel(L=300, m_seeds=12, alpha=0.98)
    x_best, dist_best = sa_tsp.optimize(dist_matrix)

    print(f"【优化成功】")
    print(f"最优路线城市顺序：\n{x_best}")
    print(f"跑出来的最短总回路长度：{dist_best:.2f} (官方理论最优值是：7542)")

    import matplotlib.pyplot as plt

    # 1. 允许 matplotlib 显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 创建画布 (左图看收敛速度和震荡，右图看最终路线拓扑)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---- 左图：收敛历史曲线 ----
    fitness_history = np.array(sa_tsp.fitness_history)
    ax1.plot(fitness_history, color='#1f77b4', linewidth=1.5, label='当前种群最优')
    ax1.axhline(y=7542, color='r', linestyle='--', alpha=0.7, label='官方理论最优 (7542)')
    ax1.set_title("模拟退火算法 - 协同赛马收敛轨迹", fontsize=12, fontweight='bold')
    ax1.set_xlabel("内层迭代总步数 (外层循环 × L)", fontsize=10)
    ax1.set_ylabel("TSP 回路总长度", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')

    # ---- 右图：Berlin52 最终路线可视化 ----
    # 按照求出的最优顺序重排坐标，别忘了闭环（首尾相连）
    best_path_idx = list(x_best) + [x_best[0]]
    ordered_coords = coords[best_path_idx]

    # 画出城市节点
    ax2.scatter(coords[:, 0], coords[:, 1], color='#e377c2', s=40, zorder=3, label='城市')
    # 给城市标上序号
    for i, (x, y) in enumerate(coords):
        ax2.text(x + 10, y + 10, str(i), fontsize=8, color='#333333', zorder=4)

    # 画出巡回连线
    ax2.plot(ordered_coords[:, 0], ordered_coords[:, 1], color='#2ca02c', linewidth=2, alpha=0.8, label='最优航线')
    ax2.set_title(f"Berlin52 最终路线轨迹 (总长: {dist_best:.2f})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("X 坐标", fontsize=10)
    ax2.set_ylabel("Y 坐标", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.show()