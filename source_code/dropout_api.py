from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this
import pandas as pd
import joblib
import os
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any

MODEL_DIR = "models"

app = FastAPI(title="Student Dropout Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
model = joblib.load(os.path.join(MODEL_DIR, 'catboost_dropout_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

class StudentData(BaseModel):
    data: List[Dict[str, Any]]

@app.post("/predict")
def predict_dropout_api(payload: StudentData):
    try:
        df = pd.DataFrame(payload.data)
        X_scaled = scaler.transform(df)
        preds = model.predict(X_scaled)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}"}
        )

@app.post("/predict-file")
def predict_dropout_file(file: UploadFile = File(...)):
    try:
        # Check file extension to determine how to read it
        filename = file.filename.lower()
        print(f"Processing file: {filename}")  # Debug log
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format. Please upload .csv, .xlsx, or .xls files"}
            )
        
        print(f"File loaded successfully. Shape: {df.shape}")  # Debug log
        print(f"Columns: {df.columns.tolist()}")  # Debug log
        
        # Handle different file formats
        if 'student_id' in df.columns and 'Target' in df.columns:
            # File has student_id and Target columns
            X = df.drop(['student_id', 'Target'], axis=1)
            student_ids = df['student_id'].tolist()
        elif 'student_id' in df.columns:
            # File has student_id but no Target
            X = df.drop(['student_id'], axis=1)
            student_ids = df['student_id'].tolist()
        else:
            X = df
            student_ids = [f"student_{i+1}" for i in range(len(df))]
        
        print(f"Features shape: {X.shape}")
        
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        
        return JSONResponse(content={
            "student_ids": student_ids,
            "predictions": preds.tolist()
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing file: {str(e)}")
        print(f"Full traceback: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={"error": f"File prediction failed: {str(e)}", "details": error_details}
        )

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Python API is running"}

if __name__ == "__main__":
    host = "127.0.0.1" 
    port = 8000
    print(f"Starting server on {host}:{port}")
    uvicorn.run("dropout_api:app", host=host, port=port, reload=True)