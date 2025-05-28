import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier



data=pd.read_csv('./data/challenge-6-abalone.csv')
data.columns=data.iloc[-1].values
data=data.drop(data.index[-1])
print(data.head())

data['Rings']=pd.to_numeric(data['Rings'])
data['Rings']=pd.cut(data['Rings'],bins=[0,10,20,30],labels=['small','middle','large'])
data['Sex']=data.Sex.replace({'M':0,'F':1,'I':2})
print(data.head())

model=KFold(n_splits=10,shuffle=True,random_state=39)
for train_index,test_index in model.split(data):
    print('TRAIN',len(train_index),'TEST:',len(test_index))
features=data.iloc[:,0:8]
target=data['Rings']

knn=KNeighborsClassifier()

cvs=cross_val_score(knn,X=features,y=target,cv=10)
print(cvs)

def classifiers():
    scores=[]
    models=[
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(),
        MLPClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
    ]
    for m in models:
        score=cross_val_score(m,X=features,y=target,cv=10)
        mean_score=np.mean(score)
        scores.append(mean_score)
    return scores

scores=classifiers()
print(scores)