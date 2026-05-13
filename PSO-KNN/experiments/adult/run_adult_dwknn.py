from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.pso_optimizer import MyPSO
from src.dwknn_model import MyKNNClassifier
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
        knn = MyKNNClassifier(k=k,weights='dw')
        # 3.使用cv=4计算准确率均值
        accuracy = knn.fit_predict_cv(X, y, cv=4)
        # 4.返回错误率（默认最小化）
        return 1 - accuracy

    return objective_func


import os


def run_adult_dwknn():
    # 1.初始化knn和pso
    dim = 1  # 版本 1.1 参数[k]
    limit = [1, 50]
    pso = MyPSO(dim=dim, limit=limit)
    # 2.加载数据集
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'adult.csv')
    adult_data = pd.read_csv(data_path)
    # adult_data.info()
    # 3.数据处理
    adult_data.replace('?', np.nan, inplace=True)
    adult_data = adult_data.dropna()
    # 重置索引（非常重要！删除行后索引不连续）
    adult_data.reset_index(drop=True, inplace=True)
    # adult_data.info()
    sample_size = 1000  # 可根据需要调整：2000-10000
    if len(adult_data) > sample_size:
        adult_data = adult_data.sample(n=sample_size, random_state=42)
    # 热编码处理
    X = adult_data.drop('income', axis=1)
    y = adult_data['income']
    X = pd.get_dummies(X, drop_first=True)
    # X.info()
    # 标签编码处理
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # 数据标准化
    transfer = StandardScaler()
    X = transfer.fit_transform(X)

    # 非常重要，因为部分数据集的原始分布是按标签分布的，可能前50个全是标签1类的集中分布
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y_encoded[indices]

    # 3.执行PSO寻找最优解
    objective_func = create_objective_func(X_shuffled, y_shuffled)
    best_k, best_accuracy = pso.optimizer(objective_func=objective_func, iters=100)

    print('PSO-KNN-adult--------------')
    print(f'最优k值: {int(best_k[0])}')
    print(f'最高准确率: {1 - best_accuracy:.4f}')


if __name__ == '__main__':
    run_adult_dwknn()
