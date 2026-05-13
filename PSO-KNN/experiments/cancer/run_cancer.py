from sklearn.preprocessing import StandardScaler

from src.pso_optimizer import MyPSO
from src.knn_model import MyKNNClassifier
import numpy as np
import pandas as pd

""" 适应度函数 
版本 1.1 参数[k]
"""


def create_objective_func(X, y):
    """创建适应度函数（优化目标）"""

    def objective_func(params):
        # 1.解析参数
        k = int(np.round(params[0]))  # 四舍五入转为整数
        if k < 1:
            k = 1
        # 2.初始化并执行knn
        knn = MyKNNClassifier(k=k)
        # 3.使用cv=4计算准确率均值
        accuracy = knn.fit_predict_cv(X, y, cv=4)
        # 4.返回错误率（默认最小化）
        return 1 - accuracy

    return objective_func

import os
def run_cancer():
    """运行breast-cancer数据集的PSO-KNN优化实验"""
    # 1.初始化knn和pso
    dim = 1  # 版本 1.1 参数[k]
    limit = [1, 50]
    pso = MyPSO(dim=dim, limit=limit)

    # 2.加载数据集
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'breast-cancer.csv')
    cancer_data = pd.read_csv(data_path,sep=';')
    # 热编码处理
    X_df = cancer_data.drop('class', axis=1)
    y = cancer_data['class'].values.astype(int)
    X_df = pd.get_dummies(X_df, columns=['age', 'tumor_size', 'tumor_position', 'breast'])
    X = X_df.values.astype(float)
    # 数据标准化
    transfer = StandardScaler()
    X = transfer.fit_transform(X)

    # 非常重要，因为部分数据集的原始分布是按标签分布的，可能前50个全是标签1类的集中分布
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # 3.执行PSO寻找最优解
    objective_func = create_objective_func(X_shuffled, y_shuffled)
    best_k, best_accuracy = pso.optimizer(objective_func=objective_func, iters=100)

    print('PSO-KNN-breast-cancer--------------')
    print(f'最优k值: {int(best_k[0])}')
    print(f'最高准确率: {1 - best_accuracy:.4f}')


if __name__ == '__main__':
    run_cancer()
