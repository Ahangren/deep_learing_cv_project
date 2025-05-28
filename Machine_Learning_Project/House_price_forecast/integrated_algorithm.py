import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

stu_data=pd.read_csv('./data/course-14-student.csv',index_col=0)
x_train,x_test,y_train,y_test=train_test_split(stu_data.iloc[:,:-1],stu_data['G3'],test_size=0.3,random_state=39)

clf1=LogisticRegression(
    solver='lbfgs',multi_class='auto',max_iter=1000,random_state=39
)
clf2=DecisionTreeClassifier(random_state=39)
clf3=GaussianNB()
eclf=VotingClassifier(
    estimators=[('lr',clf1),('dt',clf2),('gnb',clf3)],voting='hard'
)
for clf, label in zip([clf1, clf2, clf3, eclf],
                      ['LogisticRegression:', 'DecisionTreeClassifier:',
                       'GaussianNB:', 'VotingClassifier:']):
    clf.fit(x_train,y_train)
    score=clf.score(x_test,y_test)
    print(label,round(score,2))

