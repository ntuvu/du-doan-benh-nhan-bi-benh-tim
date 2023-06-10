import csv
import json

csv_file = 'Processed_Test.csv'
json_file = 'test.json'

data = []

# Read CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Write JSON file
with open(json_file, 'w') as file:
    json.dump(data, file, indent=4)

print("CSV to JSON conversion completed.")