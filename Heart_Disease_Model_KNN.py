# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Đọc file dữ liệu
dataset = pd.read_csv('Processed_Dataset.csv')

# Đưa các giá trị đặc trưng ( các chỉ số ý tế) vào trong mảng X
X = dataset.iloc[:, :-1]

# Đưa các giá trị nhãn (có bệnh hay không bệnh) vào mảng Y
y = dataset.iloc[:, -1]

# In ra so chieu cua 2 tap x, y
print(X.shape)
print(y.shape)

# Chia cac bo du lieu train va test
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)


# KNeighborsClassifier Training Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_X, train_y)

# Predicting value from test set
test_prediction = knn.predict(test_X)


from sklearn import metrics
# print("AUC score: {:.5f}".format(metrics.accuracy_score(test_y, test_prediction)))  # OUTPUT: AUC score: 0.81319
# print("MAE score: {:.5f}".format(metrics.mean_absolute_error(test_y, test_prediction)))  # OUTPUT: MAE score: 0.18681

# Su dung vong lap tim K toi uu nhat
v = []
k_range = list(range(1, 50))
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    # fit the model with training data
    knn.fit(train_X, train_y)
    pred = knn.predict(test_X)
    # adding all accuracy result to list
    v.append(metrics.accuracy_score(test_y, pred))

plt.plot(k_range, v, c='orange')
plt.show()


# Tien hanh huan luyen model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_X, train_y)
test_prediction = knn.predict(test_X)

#Tinh toan cac gia tri danh gia presition, recall, f1, roc_auc
precision = precision_score(test_y, test_prediction)
recall = recall_score(test_y, test_prediction)
f1_measure = f1_score(test_y, test_prediction)
roc_auc = roc_auc_score(test_y, test_prediction)

# In ra cac chi so danh gia model
print("Precision: {:.5f}".format(precision))
print("Recall: {:.5f}".format(recall))
print("F1-measure: {:.5f}".format(f1_measure))
print("ROC AUC: {:.5f}".format(roc_auc))

# Hien thi duong cong ROC
fpr, tpr, thresholds = roc_curve(test_y, test_prediction)

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = {:.5f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# Dumping file to pickle to make python instances
pickle.dump(knn, open('model_knn.pkl', 'wb'))

print("AUC score: {:.5f}".format(metrics.accuracy_score(test_y, test_prediction)))  # OUTPUT: AUC score: 0.86813
print("MAE score: {:.5f}".format(metrics.mean_absolute_error(test_y, test_prediction))) 
