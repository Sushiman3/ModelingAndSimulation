# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file=pd.read_csv('sample_data2.csv')
x = np.array(file['x'], dtype=float)
y = np.array(file['y'], dtype=float)
xy = np.column_stack([x,y])

# k-means by sklearn
from sklearn.cluster import KMeans
for k in range(1, 15):
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(xy)
    print('SSE for k =', k, 'is', km.inertia_)

# %%
km = KMeans(n_clusters=9, n_init=10)
km.fit(xy)
z=km.predict(xy)
print('SSE =',km.inertia_)

# Plot Data and Center of Cluster
plt.scatter(x, y, c=z)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=200, marker='*', c='red')
plt.xlabel('x'); plt.ylabel('y')
plt.show()

# %%
