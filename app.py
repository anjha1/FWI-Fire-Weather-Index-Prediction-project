import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load Ridge regressor model and standard scaler from pickle files
ridge_model = pickle.load(open('model.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get values from form input
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Combine inputs into a numpy array and scale them
        new_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Make prediction
        prediction = ridge_model.predict(new_data)

        # Render the result on the home page
        return render_template('index.html', result=prediction[0])
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
