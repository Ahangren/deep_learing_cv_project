import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ols_algebra(x,y):
    # w0=(np.pow(x)*y-np.pow(x)*np.dot(x,y))/(x.shape[0]*np.pow(x)-)
    n=len(x)
    w1=(n*sum(x*y)-sum(x)*sum(y))/(n*sum(x*x)-sum(x)*sum(x))
    w0=(sum(x*x)*sum(y)-sum(x)*sum(x*y))/(n*sum(x*x)-sum(x)*sum(x))
    return w1,w0

x = np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73])
y = np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93])

w1,w0=ols_algebra(x,y)
print(w1,w0)

