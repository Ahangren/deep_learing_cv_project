from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_name,faces.images.shape)

fig,axes=plt.subplots(1,5,figsize=(12,6))
for i,image in enumerate(faces.images[:5]):
    axes[i].imshow(image)
    axes[i].set_xlabel(faces.target_names[faces.target[i]])
plt.show()

pca=PCA(n_components=150,whiten=True,random_state=39)
pca_data=pca.fit_transform(faces.data)
print(pca_data.shape)

x_train,x_test,y_train,y_test=train_test_split(pca_data,faces.target,random_state=39)

model=SVC(kernel='poly',C=10,gamma=0.001)
model.fit(x_train,y_train)
score=model.score(x_test,y_test)
print(score)



