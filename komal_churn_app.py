
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load Trained Model
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load Scaler and Validate
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
    if not hasattr(scaler, "transform"):
        raise ValueError("Loaded scaler object is not a valid StandardScaler!")

@app.route('/')
def home():
    return "Flask App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON Data
        data = request.json

        # Extract Features
        age = data["Age"]
        subscription = data["Subscription"]

        # Convert Subscription to One-Hot Encoding (Matches Training)
        subscription_mapping = {"Basic": [1, 0], "Standard": [0, 1], "Premium": [0, 0]}
        subscription_encoded = subscription_mapping.get(subscription, [0, 0])

        # Prepare Input Array
        input_data = np.array([[age] + subscription_encoded])  # Ensure correct shape

        # Apply Feature Scaling
        input_scaled = scaler.transform(input_data)

        # Make Prediction
        prediction = model.predict(input_scaled)
        churn_result = "Yes" if prediction[0] == 1 else "No"

        return jsonify({"Churn Prediction": churn_result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
