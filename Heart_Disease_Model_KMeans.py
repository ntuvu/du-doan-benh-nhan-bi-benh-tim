# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Dataset
dataset = pd.read_csv('heart.csv')

# Selecting Features
X = dataset.iloc[:, :-1]

# Selecting Target
y = dataset.iloc[:, -1]

# Printing Shapes
print(X.shape)
print(y.shape)

# Splitting Training and testing Data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

# Plotting SSE for different K values
v = []
k_range = list(range(2, 30))
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(train_X)
    v.append(kmeans.inertia_)

plt.plot(k_range, v, c='orange')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()

# Training model with best K value
best_k = 20  # Update with the best K value you determined
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(train_X)

# Predicting cluster labels for test set
test_clusters = kmeans.predict(test_X)

# Converting cluster labels to target values
test_prediction = np.where(test_clusters == 0, 0, 1)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(test_X, test_clusters)
print("Silhouette Score:", silhouette_avg)

# Dumping file to pickle to make Python instances
pickle.dump(kmeans, open('model_kmeans.pkl', 'wb'))

# Accuracy Score
from sklearn import metrics

