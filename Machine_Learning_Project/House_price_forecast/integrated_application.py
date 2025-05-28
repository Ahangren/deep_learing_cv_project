import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data=pd.read_csv('./data/advertising.csv')
x=data[[i for i in list(data.columns)[:-1]]]
y=data['sales']
p_init=np.random.randn(2)

def func(p,x):
    w0,w1=p
    f=w0+w1*x
    return f

def err_func(p,x,y):
    ret=func(p,x)-y
    return ret
params_tv=leastsq(err_func,p_init,args=(data.tv,data.sales))
params_redio=leastsq(err_func,p_init,args=(data.radio,data.sales))
params_newspaper=leastsq(err_func,p_init,args=(data.newspaper,data.sales))

print(params_tv,params_redio,params_newspaper)

fit,axis=plt.subplots(1,3,figsize=(15,5))

data.plot(kind='scatter',x='tv',y='sales',ax=axis[0])
data.plot(kind='scatter',x='radio',y='sales',ax=axis[1])
data.plot(kind='scatter',x='newspaper',y='sales',ax=axis[2])

x_tv=np.array([data.tv.min(),data.tv.max()])
axis[0].plot(x_tv,params_tv[0][1]*x_tv+params_tv[0][0],'r')


x_radio=np.array([data.radio.min(),data.radio.max()])
axis[1].plot(x_radio,params_redio[0][1]*x_radio+params_redio[0][0],'g')

x_newspaper=np.array([data.newspaper.min(),data.newspaper.max()])
axis[2].plot(x_newspaper,params_newspaper[0][1]*x_newspaper+params_newspaper[0][0],'r')

plt.show()

model=LinearRegression()
model.fit(x,y)
print(model.intercept_,model.coef_)

results=smf.ols(formula='sales ~ tv + radio + newspaper',data=data).fit()

print(results.summary2())