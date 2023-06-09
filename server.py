
# Libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

# Using Flask create API
app = Flask(__name__)

# Build API Route


@app.route('/api/v01', methods=['POST'])
def predict():
    # get data from post request
    data = request.get_json(force=True)

    # Load the model
    model = pickle.load(open('model.pkl', 'rb'))

    # storing data
    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trestbps = data['trestbps']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca']
    thal = data['thal']

    # Making array
    X = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang , oldpeak, slope, ca, thal]]

    # convert it to array and make prediction
    prediction = model.predict(X)

    # Take the first value of prediction
    output = prediction[0]
    
    # Changing to string
    if output == 0:
        output = 'No'
    else:
        output = 'Yes'

    # return as a json
    return jsonify(output)

# Code to run server


if __name__ == '__main__':
    app.run(port=1000, debug=True)
