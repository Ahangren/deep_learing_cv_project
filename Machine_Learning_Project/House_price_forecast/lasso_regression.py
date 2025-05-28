import numpy as np
from sklearn.linear_model import Ridge

def ridge_regression(x,y,alpha):

    XTX=x.T*x
    reg=XTX+alpha*np.eye(x.shape[1])
    W=reg.I*(x.T*y)
    return W

np.random.seed(10)

X=np.matrix(np.random.randint(5,size=(10,10)))
Y=np.matrix(np.random.randint(10,size=(10,1)))
alpha=0.5

w=ridge_regression(X,Y,alpha)
print(w)


model=Ridge(alpha=alpha,fit_intercept=False)
model.fit(np.array(X),np.array(Y))
print(model.coef_)