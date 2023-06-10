# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)

# SGD Classifier Training Model
sgd = SGDClassifier(loss='log', random_state=0)
sgd.fit(train_X, train_y)

# Predicting value from test set
test_prediction = sgd.predict(test_X)

# Accuracy Score
print("AUC score: {:.5f}".format(metrics.accuracy_score(test_y, test_prediction)))
print("MAE score: {:.5f}".format(metrics.mean_absolute_error(test_y, test_prediction)))

# Printing the results
precision = precision_score(test_y, test_prediction)
recall = recall_score(test_y, test_prediction)
f1_measure = f1_score(test_y, test_prediction)
roc_auc = roc_auc_score(test_y, test_prediction)

print("Precision: {:.5f}".format(precision))
print("Recall: {:.5f}".format(recall))
print("F-measure: {:.5f}".format(f1_measure))
print("ROC AUC: {:.5f}".format(roc_auc))

# ROC Curve
fpr, tpr, thresholds = roc_curve(test_y, test_prediction)

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = {:.5f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
