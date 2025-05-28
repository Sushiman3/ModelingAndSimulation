# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file=pd.read_csv('price_calorie.csv')
xy = np.column_stack([file['price'],file['calorie']])
l = np.array(file['id'])

# 標準化
xy_scaled = (xy - np.mean(xy, axis=0)) / np.std(xy, axis=0)

# プロットは元の値で
plt.scatter(xy[:,0], xy[:,1])
for i, label in enumerate(l):
    plt.text(xy[i,0], xy[i,1], label, fontsize=8, ha='right')
plt.xlabel('price')
plt.ylabel('calorie')
plt.title('Price vs Calorie')
plt.show()
# %%

from scipy.cluster.hierarchy import linkage, dendrogram
result=linkage(xy_scaled, method = 'ward', metric='euclidean')
dendrogram(result, labels=l)
plt.show()
# %%
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# linkage
result = linkage(xy_scaled, method='ward', metric='euclidean')

# assign clusters (e.g., 3 clusters)
clusters = fcluster(result, t=4, criterion='maxclust')

# scatter plot with cluster colors
plt.scatter(xy[:,0], xy[:,1], c=clusters, cmap='tab10')
for i, label in enumerate(l):
    plt.text(xy[i,0], xy[i,1], label, fontsize=8, ha='right')
plt.xlabel('price')
plt.ylabel('calorie')
plt.title('Hierarchical Clustering Classes')
plt.show()
# %%