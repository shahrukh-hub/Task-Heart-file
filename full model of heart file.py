
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:48:26 2020

@author: Shahrukh Khan
"""
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ar1=pd.read_csv(r'D:\A.I python\heart.csv')
ar=pd.get_dummies(ar1)


print("===========Sampel===========")
print(ar1)

print("================ILoc=X-DAta===================")
x = ar1.iloc[0: , 0:13] 
print(x)
print("================ILoc=Y-Data==================")
y = ar1.iloc[0: ,13:] 
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("================X-Train==================")
print(x_train)
print("================X-Test==================")
print(x_test,"\n",x_test.shape)
print("================Y-Train==================")
print(y_train)
print("================Y-Test==================")
print(y_test,"\n",y_test.shape)

print("====================Dimension=======================")
print("Dimension:",ar.ndim)

print("====================Transpose=======================")
print("Transpose: ",ar.transpose())

print("====================Minimum=======================")
print("Minimum: ",ar.min())
print("=====================Maximum======================")
print("Maximum: ",ar.max())

print("====================Shape=================")
print("Shape: ",ar.shape)


print("=============Info=======================")
print(ar.info())


print("=============Not-Null=======================")
print(ar.count())

print("===============SquareRoot==============")
print("SquareRoot: ",np.sqrt(ar))

print("==============Standard Daviation================")
print("Standard Daviation: ",ar.std())

print("=====================Mean======================")
print("Mean: ",ar.mean())

print("=====================Median======================")
print("Median: ",ar.median())

print("=====================Mode======================")
print("Mode: ",ar.mode())

print("===================linspace==================")
lp=np.linspace(5,6,num=9)
print(lp)

print("==========Horizontal Axis===========")
print(ar.sum(axis=0))

print("===========Vertical Axis===========")
print(ar.sum(axis=1))

print("===============Sin-Degree============")
Sin=np.sin(ar)
print("Value of Sin on Iris-Data: ",Sin)

print("===============Cos-Degree============")
Cos=np.cos(ar)
print("Value of Cos on Iris-Data: ",Cos)

print("===================Ravel===========")
rav=np.ravel(ar1)
print(rav)
print("============Ravel in Int===========")
rav=np.ravel(ar)
print(rav)


#print("============Graph===========")

#print("============Sampel-Graph===========")

plt.subplot(3,2,1)
plt.title("Sampel of Heart File")
plt.plot(ar)



#print("============X-Data-Graph===========")
plt.subplot(3,2,2)
plt.title("X-Data")
plt.plot(x)

#print("============Y-Data-Graph===========")
plt.subplot(3,2,5)
plt.title("Y-Data")
plt.plot(y)
plt.show()


