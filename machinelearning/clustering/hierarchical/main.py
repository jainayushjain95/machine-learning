print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


print('Step 2 - Importing the dataset')
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

print('Step 3 - Using the dendrogram to find the optimal number of clusters')
import scipy.cluster.hierarchy as sch
## Generating a dendrogram to visualize the hierarchical clustering process
## Creating a linkage matrix using Ward's method
###'ward' minimizes the variance within clusters for better compactness

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

print('Step 4 - Training the HC model on the dataset')
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
print(y_hc)


print('Step 5 - Visualising the clusters')


# Scatter plot for Cluster 1

# y_hc is an array containing the cluster labels for each data point.
# Example: If we have 6 data points, y_hc might look like this:
# y_hc = [0, 1, 0, 2, 1, 0]
# This means:
# - The 1st, 3rd, and 6th data points belong to Cluster 1 (label 0).
# - The 2nd and 5th belong to Cluster 2 (label 1).
# - The 4th belongs to Cluster 3 (label 2).

# X is a NumPy array (or pandas DataFrame) containing the dataset with two features (columns).
# Each row represents a data point with (x, y) coordinates.

# y_hc == 0 creates a boolean mask:
# Example result: [True, False, True, False, False, True]
# This mask will help us select only the rows in X where y_hc == 0 (Cluster 1).

# X[y_hc == 0] applies this mask to X and selects only the rows where y_hc is 0.
# Example:
# If X = [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12]]
# Then X[y_hc == 0] will return:
# [[1,2],  # 1st row (Cluster 1)
#  [5,6],  # 3rd row (Cluster 1)
#  [11,12]] # 6th row (Cluster 1)

plt.scatter(
    X[y_hc == 0, 0],  # Extracts the first column (x-coordinates) from the selected rows.
                      # Example: If X[y_hc == 0] is [[1,2], [5,6], [11,12]], then
                      # X[y_hc == 0, 0] will return [1, 5, 11] (x-values).

    X[y_hc == 0, 1],  # Extracts the second column (y-coordinates) from the selected rows.
                      # Example: If X[y_hc == 0] is [[1,2], [5,6], [11,12]], then
                      # X[y_hc == 0, 1] will return [2, 6, 12] (y-values).

    s=100,            # Setting the size of the data points to make them more visible.
    c='red',          # Assigning the color red to differentiate Cluster 1 from other clusters.
    label='Cluster 1' # Adding a label for Cluster 1, which will be shown in the legend.
)

plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters of customers')
plt.legend()
plt.show()