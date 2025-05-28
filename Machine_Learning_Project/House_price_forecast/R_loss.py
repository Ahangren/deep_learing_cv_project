import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x = np.array([4, 8, 12, 25, 32, 43, 58, 63, 69, 79]).reshape(-1, 1)
y1 = np.array([9, 17, 23, 51, 62, 83, 115, 125, 137, 159]).reshape(-1, 1)
y2 = np.array([20, 33, 50, 56, 42, 31, 33, 46, 65, 75]).reshape(-1, 1)

fig,axis=plt.subplots(1,2,figsize=(12,4))

axis[0].scatter(x,y1)
axis[1].scatter(x,y2)

plt.show()


model1=LinearRegression()
model1.fit(x,y1)
model2=LinearRegression()
model2.fit(x,y2)

fig,axis=plt.subplots(1,2,figsize=(12,4))
axis[0].scatter(x,y1)
axis[0].plot(x,model1.predict(x))

axis[1].scatter(x,y2)
axis[1].plot(x,model2.predict(x))
plt.show()

rss=np.sum(np.pow((y1-model1.predict(x)),2))
tss=np.sum(np.pow((y1-np.mean(y1)),2))

loss1=r2_score(y1,model1.predict(x))
loss2=r2_score(y2,model2.predict(x))
print(loss1)
print(loss2)

print(1-rss/tss)

