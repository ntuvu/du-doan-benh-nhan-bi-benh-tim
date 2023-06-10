import pandas as pd

input_file = "heart_disease_test_data_raw.data"
output_file = "Processed_Test.csv"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Đọc file dữ liệu raw và tạo 1 DataFrame
df = pd.read_csv(input_file, header=None, names=column_names, na_values='?')

# Đưa các giá trị bị trống hoặc ngoại lai về thành kiểu NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Lấp lại các giá trị bị thiếu bằng cách tính giá trị trung bình của cột tương ứng
df.fillna(df.mean(), inplace=True)

#Chuyển giá trị nhãn target về dạng nhị phân(0 , 1)
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

#Xuất DataFrame ra file output
df.to_csv(output_file, index=False)

print(f"The data has been converted and saved to {output_file}.")