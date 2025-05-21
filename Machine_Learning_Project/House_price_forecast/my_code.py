import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('./data/challenge-1-beijing.csv')
print(data.head())
y=data['每平米价格']
x=data[['公交', '写字楼', '医院', '商场', '地铁', '学校', '建造时间', '楼层',
       '面积']]
print(x.columns)
features=np.array(x)
target=np.array(y)
print(features[0])
print(target[0])

x_train,x_test,y_train,y_test=train_test_split(features,target,train_size=0.7,random_state=39,shuffle=True)

print(len(x_train),len(x_test),len(y_train),len(y_test))

model=LinearRegression()
model.fit(x_train,y_train)

print(model.coef_[:3],len(model.coef_))

def mape(y_true,y_pred):
    n = len(y_true)
    mape = 100 * np.sum(np.abs((y_true - y_pred) / y_true)) / n
    return mape

y_true=y_test
y_pred=model.predict(x_test)

print(mape(y_true,y_pred))