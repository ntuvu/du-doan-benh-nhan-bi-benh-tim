import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load Dataset
dataset = pd.read_csv('heart.csv')

# Selecting Features
X = dataset.iloc[:, :-1]

# Selecting Target
y = dataset.iloc[:, -1]

# Splitting Training and Testing Data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

# KNeighborsClassifier Training Model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_X, train_y)

# Predicting values for the test set
test_prediction = knn.predict(test_X)

# Accuracy Score
accuracy = metrics.accuracy_score(test_y, test_prediction)
mae = metrics.mean_absolute_error(test_y, test_prediction)
print("Accuracy score: {:.5f}".format(accuracy))
print("MAE score: {:.5f}".format(mae))

# Plotting best K value for KNN
k_range = list(range(1, 50))
v = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_y)
    pred = knn.predict(test_X)
    v.append(metrics.accuracy_score(test_y, pred))

plt.plot(k_range, v, c='orange')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy')
plt.show()

# Training model with the best K value
best_k = np.argmax(v) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(train_X, train_y)
test_prediction = knn.predict(test_X)

# Dumping model to pickle file
pickle.dump(knn, open('model.pkl', 'wb'))

# Evaluating the model with the best K value
accuracy = metrics.accuracy_score(test_y, test_prediction)
mae = metrics.mean_absolute_error(test_y, test_prediction)
print("Accuracy score with best K value: {:.5f}".format(accuracy))
print("MAE score with best K value: {:.5f}".format(mae))
