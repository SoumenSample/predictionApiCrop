from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

app = FastAPI()

# ✅ Enable CORS to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Crop Recommendation API is running!"}

@app.post("/predict/")
def predict(features: dict):
    try:
        # Convert input dictionary to DataFrame
        feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        input_data = pd.DataFrame([features], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_data)

        return {"predicted_crop": prediction[0]}
    
    except Exception as e:
        return {"error": str(e)}

