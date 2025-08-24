import pandas as pd
import numpy as np
import joblib
import logging
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.combine import SMOTEENN
import shap
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_model():
    df = pd.read_csv('student_data_VN.csv')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['Target'] = df['Target'].map({'Graduate': 0, 'Enrolled': 0, 'Dropout': 1})
    X = df.drop(['student_id', 'Target'], axis=1)
    y = df['Target']

    # Train/test split + scaler
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

    # SMOTEENN balancing
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train_scaled, y_train)

    # CatBoost model training
    model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logging.info("Model trained. F1-score (Dropout): %.4f", report['1']['f1-score'])
    joblib.dump(model, os.path.join(MODEL_DIR, 'catboost_dropout_model.pkl'))

    return model, scaler, X.columns

# Predict dropout 
def predict_dropout(new_df):
    model = joblib.load(os.path.join(MODEL_DIR, 'catboost_dropout_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    X_scaled = scaler.transform(new_df)
    predictions = model.predict(X_scaled)
    return predictions.tolist()

# SHAP explainer
def explain_dropout(new_df):
    model = joblib.load(os.path.join(MODEL_DIR, 'catboost_dropout_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    X_scaled = scaler.transform(new_df)
    df_scaled = pd.DataFrame(X_scaled, columns=new_df.columns)

    explainer = shap.Explainer(model, df_scaled)
    shap_values = explainer(df_scaled)

    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_prediction.png"), dpi=300)
    plt.close()

if __name__ == '__main__':
    model, scaler, feature_names = train_model()

    df = pd.read_csv('student_data_VN.csv')
    df.fillna(df.median(numeric_only=True), inplace=True)
    df_input = df.drop(['student_id', 'Target'], axis=1)
    student_ids = df['student_id'].values[:10]

    preds = predict_dropout(df_input.head(10))
    print("\nPredictions for first 5 students:")
    for sid, p in zip(student_ids, preds):
        status = 'Dropout' if p == 1 else 'Not Dropout'
        print(f"[Student ID: {sid}] Prediction: {status}")

    explain_dropout(df_input.head(10))