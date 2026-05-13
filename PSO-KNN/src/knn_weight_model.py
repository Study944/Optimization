import numpy as np
from sklearn.neighbors import KDTree, BallTree

"""特征权重
版本3.1 添加特征权重作为PSO新的维度
"""
class MyKNNClassifier:
    # 初始化
    def __init__(self, k=5, metric='euclidean', algorithm='auto', leaf_size=30, weights=None):
        self.y_train = None
        self.X_train = None
        self.k = k
        self.metric = metric
        # kb-tree和ball-tree优化
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.tree = None
        # 特征权重
        self.weights = weights

    # 计算距离
    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        return None

    # 计算距离
    def _distance_weights(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum(self.weights*(a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.weights*(a - b)))
        return None

    # 拟合函数
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        if self.weights is not None:
            x_weighted = x * self.weights
            x_to_build = x_weighted
        else:
            x_to_build = x
        # 使用不同数据结构存储
        if self.algorithm == 'kdtree':
            self.tree = KDTree(x_to_build, leaf_size=self.leaf_size)
        elif self.algorithm == 'balltree':
            self.tree = BallTree(x_to_build, leaf_size=self.leaf_size)
        elif self.algorithm == 'auto':
            if x.shape[1] < 20:
                self.tree = KDTree(x_to_build, leaf_size=self.leaf_size)
            else:
                self.tree = BallTree(x_to_build, leaf_size=self.leaf_size)

    # 预测函数 , 返回预测值
    def predict(self, X):
        label_list = []
        # 暴力搜索
        if self.tree is None:
            for x in X:
                distance = np.array([self._distance(x, x_train) for x_train in self.X_train])
                k_ind = np.argsort(distance)[:self.k]
                k_labels = self.y_train[k_ind]
                label_list.append(int(np.bincount(k_labels).argmax()))
        # kdtree和balltree - 批量查询
        else:
            _, k_ind = self.tree.query(X, k=self.k)
            for i in range(len(X)):
                k_labels = self.y_train[k_ind[i]]
                label_list.append(int(np.bincount(k_labels).argmax()))

        return np.array(label_list)

    # 预测函数_交叉验证 ，返回cv的平均值
    def fit_predict_cv(self, X, y, cv=5):
        n_samples = len(X)
        indices = np.arange(n_samples)
        cv = cv

        # 计算每折的大小
        fold_size = n_samples // cv
        accuracies = []

        for i in range(cv):
            # 确定验证集的索引范围
            val_start = i * fold_size
            val_end = val_start + fold_size if i < cv - 1 else n_samples

            # 验证集索引
            val_indices = indices[val_start:val_end]

            # 训练集索引（除了验证集以外的所有数据）
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            # 拆分训练集和验证集
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_test_fold = X[val_indices]
            y_test_fold = y[val_indices]

            # 训练模型
            self.fit(X_train_fold, y_train_fold)

            # 预测并计算准确率
            y_pre = self.predict(X_test_fold)
            acc = score(y_test_fold, y_pre)
            accuracies.append(acc)

        # 返回平均准确率
        return np.mean(accuracies)


def score(y, y_pre):
    accuracy = np.sum(y == y_pre) / len(y)
    return accuracy
