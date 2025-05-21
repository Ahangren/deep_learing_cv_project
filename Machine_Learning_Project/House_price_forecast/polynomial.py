import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


data=pd.read_csv('./data/challenge-2-bitcoin.csv')
print(data.head())
y=data['btc_market_price']
x=data[['btc_market_price','btc_total_bitcoins','btc_market_cap']]
print(x.head())
fig,axes=plt.subplots(1,3,figsize=(16,5))
axes[0].plot(x['btc_market_price'],'green')
axes[0].set_xlabel('time')
axes[0].set_ylabel('btc_market_price')

axes[1].plot(x['btc_total_bitcoins'],'blue')
axes[1].set_xlabel('time')
axes[1].set_ylabel('btc_total_bitcoins')

axes[2].plot(x['btc_market_cap'],'brown')
axes[2].set_xlabel('time')
axes[2].set_ylabel('btc_market_cap')
plt.show()

data1=data[['btc_total_bitcoins','btc_transaction_fees']]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=39,shuffle=True)

def poly3(num):
    poly_features=PolynomialFeatures(degree=num,include_bias=False)
    poly_x_train=poly_features.fit_transform(x_train)
    poly_x_test=poly_features.fit_transform(x_test)

    model=LinearRegression()
    model.fit(poly_x_train,y_train)
    pre_y=model.predict(poly_x_test)

    mae=mean_absolute_error(y_test,pre_y)
    return mae
result=[]
for i in range(1,11):

    result.append(poly3(i))
print(result)

plt.plot(range(1,11),result,color="red")

plt.show()