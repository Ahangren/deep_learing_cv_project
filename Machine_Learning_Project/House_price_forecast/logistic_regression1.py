import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



scores = [
    [1],
    [1],
    [2],
    [2],
    [3],
    [3],
    [3],
    [4],
    [4],
    [5],
    [6],
    [6],
    [7],
    [7],
    [8],
    [8],
    [8],
    [9],
    [9],
    [10],
]
passed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]

plt.scatter(scores,passed,color='r')
# plt.show()

model=LinearRegression()
model.fit(scores,passed)

print(model.coef_,model.intercept_)

x=np.linspace(-2,12,100)

plt.plot(x,model.coef_[0]*x,+model.intercept_)
plt.scatter(scores,passed,color='r')
plt.xlabel('scores')
plt.ylabel('passed')
# plt.show()

def sigmoid(z):
    sigmoid=1/(1+np.exp(-z))
    return sigmoid

plt.plot(x,sigmoid(x),color='r')
# plt.show()

def loss(h,y):
    loss=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()
    return loss

def gradient(X,h,y):
    gradient=np.dot(X.T,(h-y))/y.shape[0]
    return gradient

df=pd.read_csv('./data/course-8-data.csv')

plt.figure(figsize=(10,6))
plt.scatter(df['X0'],df.X1,c=df['Y'])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss(h,y):
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()

def gradient(x,h,y):
    return np.dot(x.T,(h-y))/y.shape[0]

def Logstic_Regression(x,y,lr,num_iter):
    intercept=np.ones((x.shape[0],1))
    x=np.concatenate((intercept,x),axis=1)
    w=np.zeros(x.shape[1])

    for i in range(num_iter):
        z=np.dot(x,w)
        h=sigmoid(z)
        g=gradient(x,h,y)
        w-=lr*g

        l=loss(h,y)
    return l,w

x=df[['X0','X1']].values
y=df['Y'].values
lr=0.01
num_iter=30000
L=Logstic_Regression(x,y,lr,num_iter)

print(L)

plt.show()












