import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Đọc dữ liệu từ file CSV
dataset = pd.read_csv('Processed_Dataset.csv')

# Chia dữ liệu thành features (đặc trưng) và target (nhãn)
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Phân chia dữ liệu thành tập train và tập test
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

# Tạo mô hình Logistic Regression
logreg = LogisticRegression()

# Huấn luyện mô hình trên tập train
logreg.fit(train_X, train_y)

# Dự đoán nhãn của tập test
test_prediction = logreg.predict(test_X)

# Tính toán các giá trị đánh giá (precision, recall, F1-measure, ROC AUC)
precision = precision_score(test_y, test_prediction)
recall = recall_score(test_y, test_prediction)
f1_measure = f1_score(test_y, test_prediction)
roc_auc = roc_auc_score(test_y, test_prediction)

# Xuất mô hình thành tệp pickle
pickle.dump(logreg, open('model_logistic_regression.pkl', 'wb'))

# In ra các giá trị đánh giá
print("Precision: {:.5f}".format(precision))
print("Recall: {:.5f}".format(recall))
print("F1-measure: {:.5f}".format(f1_measure))
print("ROC AUC: {:.5f}".format(roc_auc))

# Vẽ đường cong ROC
fpr, tpr, thresholds = roc_curve(test_y, test_prediction)
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = {:.5f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
