import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import matplotlib.image as img
import cv2



def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
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

data=pd.read_csv("train.csv").as_matrix()

data_1=data[0:1,1:]
data_2=data[1:2,1:]
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
