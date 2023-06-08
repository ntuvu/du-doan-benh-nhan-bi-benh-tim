# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import requests

# Load Dataset
dataset = pd.read_csv('heart.csv')

# Selecting Features
X = dataset.iloc[:, :-1]

# Selecting Target
y = dataset.iloc[:, -1]

# Printing Features And Target names
# print('Features :' , X)
# print('Target :', y)

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


# K-Means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train_X)

# Predicting cluster labels for test set
test_clusters = kmeans.predict(test_X)

# Converting cluster labels to target values
test_prediction = np.where(test_clusters == 0, 0, 1)

# Accuracy Score
from sklearn import metrics
print("AUC score: {:.5f}".format(metrics.accuracy_score(test_y, test_prediction)))  # OUTPUT: AUC score: 0.72527
print("MAE score: {:.5f}".format(metrics.mean_absolute_error(test_y, test_prediction)))  # OUTPUT: MAE score: 0.27473

# Plotting SSE for different K values
v = []
k_range = list(range(1, 10))
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(train_X)
    v.append(kmeans.inertia_)

plt.plot(k_range, v, c='orange')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.show()

# Training model with best K value
kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(train_X)
test_clusters = kmeans.predict(test_X)
test_prediction = np.where(test_clusters == 0, 0, 1)

# Dumping file to pickle to make Python instances
pickle.dump(kmeans, open('model_kmeans.pkl', 'wb'))

print("AUC score: {:.5f}".format(metrics.accuracy_score(test_y, test_prediction)))  # OUTPUT: AUC score: 0.72527
print("MAE score: {:.5f}".format(metrics.mean_absolute_error(test_y, test_prediction)))  # OUTPUT: MAE score: 0.27473