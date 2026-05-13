import numpy as np
from fontTools.misc.bezierTools import epsilon
from sklearn.neighbors import KDTree, BallTree


class MyKNNClassifier:
    # 初始化
    def __init__(self, k=5, metric='euclidean', algorithm='auto', leaf_size=30, weights='uniform'):
        self.y_train = None
        self.X_train = None
        self.k = k
        self.metric = metric
        # kb-tree和ball-tree优化
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.tree = None
        # 样本权重(uniform:统一权重，dw:双重权重)
        self.weights = weights
        self.v = None

    # 计算距离
    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        return None

    """权重
    版本 2.1 双重权重
    邻居有效性权重计算
    """

    # 拟合函数
    def fit(self, x, y):
        self.X_train = x
        self.y_train = y
        # 使用不同数据结构存储
        if self.algorithm == 'kdtree':
            self.tree = KDTree(x, leaf_size=self.leaf_size)
        elif self.algorithm == 'balltree':
            self.tree = BallTree(x, leaf_size=self.leaf_size)
        elif self.algorithm == 'auto':
            if x.shape[1] < 20:
                self.tree = KDTree(x, leaf_size=self.leaf_size)
            else:
                self.tree = BallTree(x, leaf_size=self.leaf_size)

        # 计算邻居有效性权重
        if self.weights == 'dw':
            self.v = []
            if self.tree is None:
                for i in range(len(self.X_train)):
                    distance = np.array([self._distance(self.X_train[i], x_train) for x_train in self.X_train])
                    k_ind = np.argsort(distance)[:self.k + 1]
                    k_ind = k_ind[distance[k_ind] > 1e-8][:self.k]
                    k_labels = self.y_train[k_ind]
                    validity = np.sum(k_labels == self.y_train[i]) / self.k
                    self.v.append(validity)
            else:
                _, k_ind = self.tree.query(self.X_train, k=self.k + 1)
                self.v = []
                for i in range(len(self.X_train)):
                    k_ind_i = k_ind[i][1:]
                    k_labels = self.y_train[k_ind_i]
                    # v = 邻居中同一类别样本数量 / 邻居数量
                    validity = np.sum(k_labels == self.y_train[i]) / self.k
                    self.v.append(validity)
            self.v = np.array(self.v)
            # print(f'邻居有效性权重：{self.v[:5]}')

    """权重
    版本 2.1 双重权重
    池化计算
    """

    # 预测函数 , 返回预测值
    def predict(self, X):
        label_list = []
        # 统一处理：确保 X 是二维的
        if X.ndim == 1: X = X.reshape(1, -1)

        # 1. 获取邻居 (优先使用 Tree 提升效率)
        if self.tree is not None:
            distances, indices = self.tree.query(X, k=self.k)
        else:
            # 暴力搜索逻辑
            all_dist = np.array([[self._distance(x, x_train) for x_train in self.X_train] for x in X])
            indices = np.argsort(all_dist, axis=1)[:, :self.k]
            distances = np.take_along_axis(all_dist, indices, axis=1)

        # 获取所有训练集的标签
        unique_labels = np.unique(self.y_train)
        num_classes = len(unique_labels)

        for i in range(len(X)):
            k_indices = indices[i]
            k_dists = distances[i]
            k_labels = self.y_train[k_indices]

            if self.v is None:
                # 标准 KNN 逻辑
                label_list.append(int(np.bincount(k_labels).argmax()))
            else:
                k_v = self.v[k_indices]
                # 初始化类别池化器
                class_scores = np.zeros(num_classes)
                # 动态 Sigma：使用当前邻居的最大距离保证数值稳定性
                sigma = np.max(k_dists) + 1e-8
                for l_idx, label_val in enumerate(unique_labels):
                    mask = (k_labels == label_val)
                    if not np.any(mask):
                        continue
                    # 1. 距离池化 (Mean/Min Pooling)
                    pooled_dist = np.mean(k_dists[mask])
                    # 2. 有效性池化 (Mean Pooling)
                    pooled_v = np.mean(k_v[mask])
                    # 3. 权重计算
                    w_d = np.exp(-pooled_dist / sigma)
                    # 4. 频率补偿 (非常重要)：避免只有1个邻居的类别因为距离近而产生误判
                    # 加上一个小权重的频率项，或者直接按论文公式：Score = V * Wd
                    frequency_factor = np.sum(mask) / self.k
                    class_scores[l_idx] = pooled_v * w_d * (1 + frequency_factor)
                label_list.append(unique_labels[np.argmax(class_scores)])

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
