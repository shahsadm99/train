import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import matplotlib.image as img
import cv2

data=pd.read_csv("train.csv").as_matrix()

data_1=data[0:1,1:]
data_2=data[1:2,1:]
#pt.imshow(data_1)
data_1.shape=(28,28)
pt.imshow(255-data_1,cmap='gray')
pt.show()
data_2.shape=(28,28)
pt.imshow(255-data_2,cmap='gray')
pt.show()
vis = np.concatenate((data_1, data_2), axis=1)
#vis = np.concatenate((data_1, data_2), axis=0)
pt.imshow(255-vis,cmap='gray')
pt.show()
