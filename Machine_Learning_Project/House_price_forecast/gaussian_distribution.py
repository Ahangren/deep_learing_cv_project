import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,u,d):
    d_2 = d * d * 2
    zhishu = -(np.square(x - u) / d_2)
    exp = np.exp(zhishu)
    pi = np.pi
    xishu = 1 / (np.sqrt(2 * pi) * d)
    p = xishu * exp
    return p

x=np.linspace(-5,5,100)
u=3.2
d=5.5
g=gaussian(x,u,d)
print(len(g))
print(g[10])


y1=gaussian(x,0,1)
y2=gaussian(x,-1,2)
y3=gaussian(x,1,0.5)
y4=gaussian(x,0.5,5)

plt.figure(figsize=(8,5))
plt.plot(x,y1,c='r',label='01')
plt.plot(x,y2,c='b',label='-12')
plt.plot(x,y3,c='g',label='10.5')
plt.plot(x,y4,c='y',label='0.55')
plt.legend()
plt.show()



