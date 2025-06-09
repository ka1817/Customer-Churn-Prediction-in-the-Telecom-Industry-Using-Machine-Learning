import pandas as pd
import os

def data_ingestion():
   
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    train_path = os.path.join(base_dir, "data", "telecom_churn_data.csv")

    df = pd.read_csv(train_path)
    
    return df

if __name__ == '__main__':
    data_ingestion()