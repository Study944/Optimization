from sklearn.preprocessing import StandardScaler

from src.pso_optimizer import MyPSO
from src.knn_weight_model import MyKNNClassifier
from sklearn.datasets import load_iris
import numpy as np

""" 适应度函数 
版本 3.1 参数[k值,特征权重w]
"""


def create_objective_func(X, y):
    """创建适应度函数（优化目标）"""

    def objective_func(params):
        # 1.解析参数
        k = int(np.round(params[0]))  # 四舍五入转为整数
        if k < 1:
            k = 1
        w = params[1:]
        # 2.初始化并执行knn
        knn = MyKNNClassifier(k=k, weights=w)
        # 3.使用cv=4计算准确率均值
        accuracy = knn.fit_predict_cv(X, y, cv=4)
        # 4.返回错误率（默认最小化）
        # 添加惩罚系数避免过拟合
        # penalty = 0.1 * (np.sum(w) / len(w))
        return 1 - accuracy

    return objective_func


def run_iris_weight():
    """运行Iris数据集的PSO-KNN优化实验"""

    # 1.加载数据集
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target
    # 数据标准化
    transfer = StandardScaler()
    X = transfer.fit_transform(X)

    # 非常重要，因为部分数据集的原始分布是按标签分布的，可能前50个全是标签1类的集中分布
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # 2.初始化knn和pso
    n_features = X.shape[1]
    dim = 1 + n_features  # 版本 1.1 参数[k]
    limit_k = [1, 50]
    limit_w = [0, 1]
    limit = [limit_k]+[limit_w]*n_features
    pso = MyPSO(dim=dim, limit=limit)

    # 3.执行PSO寻找最优解
    objective_func = create_objective_func(X_shuffled, y_shuffled)
    best_param, best_accuracy = pso.optimizer(objective_func=objective_func, iters=100)

    print('PSO-KNN-weights-iris--------------')
    print(f'最优k值: {int(best_param[0])}')
    print(f'最优权重: {best_param[1:]}')
    print(f'最高准确率: {1 - best_accuracy:.4f}')


if __name__ == '__main__':
    run_iris_weight()
