
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from boschParser_py import arr



# Configuration options
epsilon = 15
min_samples = 10


# Generate 2D pointcloud

# num_samples_total = 1000
# cluster_centers = [(3, 3), (4.5, 4), (1, 1), (2, 2)]
# num_classes = len(cluster_centers)

# X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)
# X : array of shape [n_samples, n_features] The generated samples.
# print(np.shape(X)) 
# # print(np.shape(y)) 
# y : array of shape [n_samples], The integer labels for cluster membership of each sample.

# # plot 2D data points
# plt.scatter(X[:,0], X[:,1], marker="o", picker=True)
# plt.title('2D clusters ')
# plt.xlabel('Axis X[0]')
# plt.ylabel('Axis X[1]')
# plt.show()


# db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)   # Perform DBSCAN clustering from features, or distance matrix.
# selfobject, Returns a fitted instance of self
# labels = db.labels_   # attribute in dbscan, Cluster labels. Noisy samples are given the label -1.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

# no_clusters = len(np.unique(labels) ) # find unique label number
# no_noise = np.sum(np.array(labels) == -1, axis=0)

# print('Estimated no. of clusters: %d' % (no_clusters - 1))
# print('Estimated no. of noise points: %d' % no_noise)

# Generate scatter plot for training data
# (no necessary) colors = list(map(lambda x: 'b' if x == 1 else ('g' if x == 0 else ('y' if x == -1 else 'r')), labels))
# plt.scatter(X[:,0], X[:,1], c = labels, marker = "o", picker = True)
# plt.title('2D clusters with dbscan')
# plt.xlabel('Axis X[0]')
# plt.ylabel('Axis X[1]')
# plt.show()



# Generate 3D pointcloud

# num_samples_total = 2000
# cluster_centers = [(3, 3, 0), (4.5, 4, 1), (1, 1, 1), (2, 2, 2)]
# num_classes = len(cluster_centers)

# X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5)
# X : array of shape [n_samples, n_features] The generated samples.
# print(np.shape(X)) 
# print(np.shape(y)) 
# y : array of shape [n_samples], The integer labels for cluster membership of each sample.


# # plot 3d data points

# fig = plt.figure(figsize = (7, 7))
# ax = Axes3D(fig)

# # Data for three-dimensional scattered points

# ax.scatter3D(X[:,0], X[:,1], X[:,2])
# ax.set_title("3D clustering")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")   
# ax.set_zlabel("Z") 
# plt.show()

# db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)   # Perform DBSCAN clustering from features, or distance matrix.
# selfobject, Returns a fitted instance of self
# labels = db.labels_   # attribute in dbscan, Cluster labels. Noisy samples are given the label -1.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

# no_clusters = len(np.unique(labels) ) # find unique label number
# no_noise = np.sum(np.array(labels) == -1, axis=0)

# print('Estimated no. of clusters: %d' % (no_clusters - 1))
# print('Estimated no. of noise points: %d' % no_noise)

# Generate scatter plot for training data
# (no necessary) colors = list(map(lambda x: 'b' if x == 1 else ('g' if x == 0 else ('y' if x == -1 else 'r')), labels))
# ax.scatter3D(X[:,0], X[:,1], X[:,2], c=labels, marker="o", picker=True )
# ax.set_title("3D clusers with dbscan")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")   
# ax.set_zlabel("Z") 
# plt.show()




# plot 2D radar data (x, y)


arr_2d = arr[:, :2]
rangeY = 30

plt.scatter(arr_2d[:,0], arr_2d[:,1], marker="o", picker=True)
plt.title('2D radar data')
plt.xlabel('Axis X')
plt.ylabel('Axis Y')
plt.ylim(-rangeY, rangeY)
plt.show()
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(arr_2d)   # Perform DBSCAN clustering from features, or distance matrix.
# selfobject, Returns a fitted instance of self
labels = db.labels_   # attribute in dbscan, Cluster labels. Noisy samples are given the label -1.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

no_clusters = len(np.unique(labels) ) # find unique label number
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % (no_clusters - 1))
print('Estimated no. of noise points: %d' % no_noise)


# Generate scatter plot for training data
# (no necessary) colors = list(map(lambda x: 'b' if x == 1 else ('g' if x == 0 else ('y' if x == -1 else 'r')), labels))
plt.scatter(arr_2d[:,0], arr_2d[:,1], c = labels, marker = "o", picker = True)
plt.title('2D radar data with dbscan')
plt.xlabel('Axis X')
plt.ylabel('Axis Y')
plt.ylim(-rangeY, rangeY)
plt.show()





# plot 3d radar data 

rangeY = 30
fig = plt.figure(figsize = (7, 7))
ax = Axes3D(fig)
# Data for three-dimensional scattered points
# ax.scatter3D(X[:,0], X[:,1], X[:,2] )
ax.scatter3D(arr[:,0], arr[:,1], arr[:,2])


ax.set_title("3D radar datan")
ax.set_xlabel("X")
ax.set_ylabel("Y")  
ax.set_ylim(-rangeY, rangeY) 
ax.set_zlabel("Z") 

plt.show()
print("hi")
# Compute DBSCAN

db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(arr)   # Perform DBSCAN clustering from features, or distance matrix.
# selfobject, Returns a fitted instance of self
labels = db.labels_   # attribute in dbscan, Cluster labels. Noisy samples are given the label -1.
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

no_clusters = len(np.unique(labels) ) # find unique label number
no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % (no_clusters - 1))
print('Estimated no. of noise points: %d' % no_noise)

# Generate scatter plot for training data
# colors = list(map(lambda x: 'b' if x == 1 else ('g' if x == 0 else ('y' if x == -1 else 'r')), labels))
# plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
# plt.title('3D radar data')
# plt.xlabel('Axis X[0]')
# plt.ylabel('Axis X[1]')
# plt.show()




# plot 3d data points
fig = plt.figure(figsize = (7, 7))
ax = Axes3D(fig)



# Data for three-dimensional scattered points
# ax.scatter3D(X[:,0], X[:,1], X[:,2], c=labels, marker="o", picker=True )
ax.scatter3D(arr[:,0], arr[:,1], arr[:,2], c=labels, marker="o", picker=True )
ax.set_title("3D radar data with dbscan")
ax.set_xlabel("X")
ax.set_ylabel("Y")  
ax.set_ylim(-rangeY, rangeY)
ax.set_zlabel("Z") 
ax.auto_scale_xyz((0, 300), (-30, 30), (-40, 40))
plt.show()



