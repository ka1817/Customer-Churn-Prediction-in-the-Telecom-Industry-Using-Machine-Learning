import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(df):
    logging.info("Starting preprocessing...")

    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    nominal_features = [
        'Contract', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'PaperlessBilling', 'PaymentMethod'
    ]
    binary_features = ['SeniorCitizen']
    numeric_features = ['TotalCharges']

    X = df[nominal_features + binary_features + numeric_features]
    y = df['Churn']

    preprocessor = ColumnTransformer(transformers=[
        ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
        ('num', StandardScaler(), numeric_features),
        ('bin', 'passthrough', binary_features)
    ])

    logging.info("Preprocessing complete.")
    return X, y, preprocessor


if __name__ == '__main__':
    from src.data_ingestion import data_ingestion
    df = data_ingestion()
    X, y, preprocessor = preprocess_data(df)
