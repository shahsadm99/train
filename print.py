import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import matplotlib.image as img
from sklearn.tree import DecisionTreeClassifier
#image = img.imread('ig.png')

data=pd.read_csv("train.csv").as_matrix()
clf=DecisionTreeClassifier()

#xtrain=data[0:5,1:]
#train_label=data[0:5,0]

#clf.fit(xtrain,train_label)

xtest=data[0:,1:]
actual_label=data[0:,0]

d=xtest[2]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
#print(clf.predict([xtest[0]]))

pt.show()
