import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from matplotlib import cm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)

data = pd.read_csv('dataset.csv')

f4 = data['WBC'].values
f5 = data['RBC'].values

X = np.array(list(zip(f4, f5)))

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = ["WBC", "RBC"])
X.plot.scatter(x = "WBC", y = "RBC" ,s =200)


outlier_detection = DBSCAN(eps = 0.3,metric="euclidean",min_samples = 3,n_jobs = -1)
clusters = outlier_detection.fit_predict(X)


cmap = cm.get_cmap('Accent')
X.plot.scatter(x = "WBC",y = "RBC",c = clusters,cmap = cmap,colorbar = False,s=200)