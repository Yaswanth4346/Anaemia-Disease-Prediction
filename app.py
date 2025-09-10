# app.py
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model and scaler
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Home route - renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route - handles form submission and returns prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    gender = int(request.form['gender'])
    hemoglobin = float(request.form['hemoglobin'])
    mch = float(request.form['mch'])
    mchc = float(request.form['mchc'])
    mcv = float(request.form['mcv'])

    # Prepare the input data as a 2D array (needed for the scaler)
    input_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
    
    # Scale the data
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)

    # Convert prediction to a human-readable result
    result = 'Anaemia Detected' if prediction[0] == 1 else 'No Anaemia'

    # Return the result to the user
    return render_template('index.html', prediction_text='Prediction: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
