from sklearn.preprocessing import StandardScaler

from src.pso_optimizer import MyPSO
from src.knn_weight_model import MyKNNClassifier
import numpy as np
import os
import pandas as pd

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
        knn = MyKNNClassifier(algorithm='brute',k=k, weights=w , metric = 'manhattan')
        # 3.使用cv=4计算准确率均值
        accuracy = knn.fit_predict_cv(X, y, cv=5)
        # 4.返回错误率（默认最小化）
        # 添加惩罚系数避免过拟合
        penalty = 0.01 * (np.sum(w) / len(w))
        return 1 - accuracy + penalty

    return objective_func


def run_cancer_weight():
    # 1.1 加载数据集
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'breast-cancer.csv')
    cancer_data = pd.read_csv(data_path, sep=';')
    # 1.2 特征和标签提取
    X_df = cancer_data.drop('class', axis=1)
    y = cancer_data['class'].values.astype(int)
    # 1.3 热编码处理
    X_df = pd.get_dummies(X_df, columns=['age', 'menopause', 'node_caps', 'tumor_size', 'tumor_position', 'breast'])
    # 1.4 数据标准化
    # numeric_columns = X_df.select_dtypes(include=[np.number]).columns
    # transfer = StandardScaler()
    # X_df[numeric_columns] = transfer.fit_transform(X_df[numeric_columns])
    X = X_df.values.astype(float)

    # 1.5 打乱数据分布
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
    limit = [limit_k] + [limit_w] * n_features
    pso = MyPSO(n=50, dim=dim, limit=limit)

    # 3.执行PSO寻找最优解
    objective_func = create_objective_func(X_shuffled, y_shuffled)
    best_param, best_accuracy = pso.optimizer(objective_func=objective_func, iters=100)

    print('PSO-KNN-weights-cancer--------------')
    print(f'最优k值: {int(best_param[0])}')
    print(f'最优权重: {best_param[1:]}')
    print(f'最高准确率: {1 - best_accuracy:.4f}')


if __name__ == '__main__':
    run_cancer_weight()
