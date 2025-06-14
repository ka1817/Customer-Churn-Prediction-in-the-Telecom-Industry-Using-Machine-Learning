# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import logging
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'pipeline.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def preprocess_data(df):
    logging.info("Starting preprocessing")

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    ordinal_features = ['Contract']
    ordinal_mapping = [['Month-to-month', 'One year', 'Two year']]

    nominal_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod'
    ]
    binary_features = ['SeniorCitizen']
    numeric_features = ['tenure', 'MonthlyCharges']

    target = 'Churn'
    features = ordinal_features + nominal_features + binary_features + numeric_features

    X = df[features]
    y = df[target]

    preprocessor = ColumnTransformer(transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_mapping), ordinal_features),
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
        ('num', StandardScaler(), numeric_features),
        ('bin', 'passthrough', binary_features)
    ])

    logging.info("Preprocessing complete")
    return X, y, preprocessor

if __name__ == '__main__':
    from src.data_ingestion import data_ingestion
    df = data_ingestion()
    X, y, preprocessor = preprocess_data(df)
