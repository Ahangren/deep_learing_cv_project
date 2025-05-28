import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

digits=load_digits()

model=DecisionTreeClassifier(random_state=39)
model.fit(digits.data,digits.target)
cvs=cross_val_score(model,digits.data,digits.target,cv=5)
print(np.mean(cvs))

tuned_parameters={'min_samples_split':[2,10,20],'min_samples_leaf':[1,5,10]}
grid_search=GridSearchCV(model,tuned_parameters,cv=10)
grid_search.fit(digits.data,digits.target)
print(grid_search.best_score_)
print(grid_search.best_estimator_)

tuned_parameters1={'min_samples_split':np.random.randint(2,20),'min_samples_leaf':np.random.randint(1,10)}

rs_model=RandomizedSearchCV(model,tuned_parameters1,n_iter=10,cv=5)
rs_model.fit(digits.data,digits.target)

print(rs_model.best_score_)
print(rs_model.best_estimator_)



