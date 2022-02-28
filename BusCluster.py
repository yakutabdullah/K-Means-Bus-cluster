# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans

# Loads the iris dataset
df = pd.read_csv('segment_val12.csv')
df.head()
#print(iris)
arr = np.array(df['sure'])
arr

y = np.zeros(len(arr))
plt.scatter(arr,y,color = 'm')
plt.xticks(np.arange(0, 120, 10))
plt.xlabel('CGPA VALUE')

k_range = range(1,11)
inertias = []
for k in k_range:
    km = KMeans(n_clusters = k)
    km.fit(df[['sure']])
    inertias.append(km.inertia_)
y = np.zeros(len(inertias))
inertias

plt.plot(inertias)
plt.xlabel('K Value')
plt.ylabel('SUM OF SQUARE ERROR')
plt.xticks(np.arange(0, 9, 1))

km = KMeans(n_clusters = 2)
y_predicted = km.fit_predict(df[['sure']])
y_predicted

df['cluster']=y_predicted
df

color = ['tomato','limegreen','midnightblue','blueviolet', 'g', 'r', 'c', 'm', 'wheat', 'k']
for i in range(2):
    df0 = df[df.cluster == i]
    y = np.zeros(len(df0['sure']))
    plt.scatter(df0.sure,y,color = color[i],label = 'CLUSTER ' + str(i))
y = np.zeros(len(km.cluster_centers_))   
plt.scatter(km.cluster_centers_[:],y,color = 'yellow',marker = '*',label = 'CENTRIOD',s = 500)
plt.legend()
plt.xlabel("CGPA VALUES")
plt.xticks(np.arange(50, 250, 25))

