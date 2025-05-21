from mpmath.libmp import normalize
from scipy.linalg import hilbert
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import Ridge,Lasso
import matplotlib.pyplot as plt

x=hilbert(10)
np.random.seed(10)
w=np.random.randint(2,10,10)
y_temp=np.matrix(x)*np.matrix(w).T
y=np.array(y_temp.T)[0]

print("实际参数 w",w)
print('实际函数值：y',y)

func=lambda p,x:np.dot(p,x)
err_func=lambda p,x,y:y-func(p,x)
p_init=np.random.randint(1,2,10)

parameters=leastsq(err_func,p_init,args=(x,y))
print('拟合参数：w',parameters[0])


model=Ridge(alpha=1.0,fit_intercept=False,copy_X=True,max_iter=None,tol=0.001,solver='auto',random_state=None)

model.fit(x,y)

alphas=np.linspace(1,10,20)
coefs=[]
for a  in alphas:
    ridge=Ridge(alpha=a,fit_intercept=False)
    ridge.fit(x,y)
    coefs.append(ridge.coef_)

plt.plot(alphas,coefs)
plt.scatter(np.linspace(1,0,10),parameters[0])
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Ridge Regression')
plt.show()


lasso_cofes=[]
for a in alphas:
    lasso=Lasso(alpha=a,fit_intercept=False)
    lasso.fit(x,y)
    lasso_cofes.append(lasso.coef_)

plt.plot(alphas,lasso_cofes)
plt.scatter(np.linspace(1,0,10),parameters[0])
plt.xlabel('alpha')
plt.ylabel('w')
plt.title('Lasso Regression')
plt.show()

