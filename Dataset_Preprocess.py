import pandas as pd

input_file = "heart_disease_data_raw.data"
output_file = "converted_data.csv"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Read the input file and create a DataFrame
df = pd.read_csv(input_file, header=None, names=column_names, na_values='?')



# Convert non-numeric values to NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with the mean of the respective column
df.fillna(df.mean(), inplace=True)

# Convert the "target" column values
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Save the converted DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"The data has been converted and saved to {output_file}.")