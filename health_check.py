# health_check.py
import os
import sys

print("Health Check - Loan Default Prediction System")
print("="*50)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Models directory exists: {os.path.exists('./models')}")

if os.path.exists('./models'):
    import joblib
    try:
        model = joblib.load('./models/best_tuned_model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model load error: {e}")

print("Health check complete")