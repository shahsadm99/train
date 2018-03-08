import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import matplotlib.image as img
import cv2





def autocrop(image, threshold=0):

    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[:, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
data=pd.read_csv("train.csv")
#data6=pd.read_csv("train.csv").as_matrix()
print (data)


print("enter a no")
input_x=int(input())
#df2 = pd.DataFrame([[input_x]], columns=list('A'))
#print(df2)
#a=df2.iat[0,0]
#print(df2.iat[0,0])
print(input_x)
print("enter a no")
input_y=int(input())
print(input_y)
#result_1=pd.filter(like=input_x, axis=0)
#print (result_1)
result_1 = data[data["label"]==input_x].as_matrix()
print(result_1)
result_2 = data[data['label']==input_y].as_matrix()
data_1=result_1[0:1,1:]
data_2=result_2[0:1,1:]
#data_2=data[1:2,1:]




#pt.imshow(data_1)
data_1.shape=(28,28)
pt.imshow(255-data_1,cmap='gray')
pt.show()
data_3=autocrop(data_1)
pt.imshow(255-data_3,cmap='gray')
pt.show()
data_2.shape=(28,28)
pt.imshow(255-data_2,cmap='gray')
pt.show()
data_4=autocrop(data_2)
pt.imshow(255-data_4,cmap='gray')
pt.show()
vis = np.concatenate((data_3, data_4), axis=1)
#vis = np.concatenate((data_1, data_2), axis=0)
pt.imshow(255-vis,cmap='gray')
pt.show()
