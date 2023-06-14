import csv
import json

csv_file = 'Processed_Test.csv'
json_file = 'data.json'

data = []

# Read CSV file
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Read the header row
    for row in reader:
        # Convert each value in the row to an integer
        converted_row = [float(value) for value in row]
        data.append(dict(zip(header, converted_row)))

# Write JSON file
with open(json_file, 'w') as file:
    json.dump(data, file, indent=4)

print("CSV to JSON conversion completed.")