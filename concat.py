import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import matplotlib.image as img

data=pd.read_csv("new.csv").as_matrix()

data_1=data[0:1,1:]
data_2=data[1:2,1:]
