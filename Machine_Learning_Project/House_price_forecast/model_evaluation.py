import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

data=pd.read_csv('./data/credit_risk_train.csv')
print(data.head())

data.RISK=data.RISK.replace({'LOW':0,'HIGH':1})

train_data=data.iloc[:,:-1]
train_data=pd.get_dummies(train_data)
train_data=scale(train_data)

train_target=data['RISK']

x_train,x_test,y_train,y_test=train_test_split(train_data,train_target,train_size=0.7,random_state=39)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(y_pred)

def get_accuracy(test_labels,pred_labels):
    correct=np.sum(test_labels==pred_labels)
    n=len(test_labels)
    acc=correct/n
    return acc

predict_acc=get_accuracy(y_test,y_pred)
print(predict_acc)
print(accuracy_score(y_test,y_pred))
print(model.score(x_test,y_test))
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))


y_score=model.decision_function(x_test)
fpr,tpr,_=roc_curve(y_test,y_score)
roc_auc=auc(fpr,tpr)

plt.plot(fpr,tpr,label='ROC')
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.show()