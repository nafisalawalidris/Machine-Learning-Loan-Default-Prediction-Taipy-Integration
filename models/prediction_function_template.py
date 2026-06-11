# Prediction Function for Loan Default Model
import pandas as pd
import numpy as np
import joblib

def load_model():
    """Load the trained model and preprocessing artefacts"""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoders = joblib.load('models/encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, encoders, feature_names

def preprocess_input(data, scaler, encoders, feature_names):
    """Preprocess input data"""
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Apply label encoding for categorical variables
    label_encoders = encoders.get('label_encoders', {})
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Scale numerical features
    scaled_data = scaler.transform(df[feature_names])
    return scaled_data

def predict_default(data):
    """Make prediction for a single loan application"""
    model, scaler, encoders, feature_names = load_model()
    processed_data = preprocess_input(data, scaler, encoders, feature_names)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    return {
        'default_predicted': bool(prediction),
        'default_probability': float(probability),
        'risk_level': 'High' if probability > 0.5 else 'Medium' if probability > 0.3 else 'Low'
    }

# Example usage
if __name__ == "__main__":
    # Sample input (adjust based on your features)
    sample_input = {
        # Add feature names and values here
    }
    result = predict_default(sample_input)
    print(result)
