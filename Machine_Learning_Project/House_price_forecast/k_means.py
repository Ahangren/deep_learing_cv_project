import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def knn_regression(train_data,train_labels,test_date,k):
    test_label=np.array([])
    for test_x in test_date:
        distances=np.array([])
        for train_x in train_data:
            distance=np.sqrt(np.sum(np.square(train_x-test_x)))
            distances=np.append(distances,distance)
        sort_distances=distances.argsort()
        labels=train_labels[sort_distances[:k]]
        label_mean=np.mean(labels)
        test_label=np.append(test_label,label_mean)
    return test_label

train_data = np.array(
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]
)
# 训练样本目标值
train_labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 测试样本特征
test_data = np.array([[1.2, 1.3], [3.7, 3.5], [5.5, 6.2], [7.1, 7.9]])
# 测试样本目标值
test_label=knn_regression(train_data, train_labels, test_data, k=3)
print(test_label)
