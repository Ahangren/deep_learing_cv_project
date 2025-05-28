import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

x,y=make_blobs(n_samples=60,centers=2,random_state=39,cluster_std=0.8)
plt.figure(figsize=(10,8))
plt.scatter(x[:,0],x[:,1],c=y,s=40,cmap='bwr')

x_temp = np.linspace(0, 6)
print(x_temp)
for m, b, d in [(1, -8, 0.2), (0.5, -6.5, 0.55), (-0.2, -4.25, 0.75)]:
    y_temp = m * x_temp + b
    plt.plot(x_temp, y_temp, "-k")
    plt.fill_between(x_temp, y_temp - d, y_temp + d, color="#f3e17d", alpha=0.5)

# sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
# C: 支持向量机中对应的惩罚参数。
#
# kernel: 核函数，linear, poly, rbf, sigmoid, precomputed 可选，下文详细介绍。
#
# degree: poly 多项式核函数的指数。
#
# tol: 收敛停止的容许值。

linear_svc=SVC(kernel='linear')
linear_svc.fit(x,y)
print(linear_svc.intercept_,linear_svc.coef_)
print(linear_svc.support_vectors_)


def svc_plot(model):
    ax=plt.gca()
    x=np.linspace(ax.get_xlim()[0],ax.get_xlim()[1],50)
    y=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],50)
    Y,X=np.meshgrid(y,x)
    xy=np.vstack([X.ravel(),Y.ravel()]).T
    p=model.decision_function(xy).reshape(X.shape)
    ax.contour(X,Y,p,colors='green',levels=[-1,0,1],linestyles='--')

    ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],c='green',s=100)

plt.figure(figsize=(10,8))
plt.scatter(x[:,0],x[:,1],c=y,s=40,cmap='bwr')
svc_plot(linear_svc)

# 向原数据集中加入噪声点
x = np.concatenate((x, np.array([[3, -4], [4, -3.8], [2.5, -6.3], [3.3, -5.8]])))
y = np.concatenate((y, np.array([1, 1, 0, 0])))

plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="bwr")

linear_svc.fit(x, y)  # 训练

plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="bwr")
svc_plot(linear_svc)

plt.show()