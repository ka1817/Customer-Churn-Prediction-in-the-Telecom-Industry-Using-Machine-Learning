import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_ingestion():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "data", "telecom_churn_data.csv")
        
        df = pd.read_csv(train_path)
        logging.info(f"✅ Data loaded successfully from {train_path}")
        logging.info(f"📊 Dataset shape: {df.shape}")
        
        return df

    except FileNotFoundError:
        logging.error(f"❌ File not found at path: {train_path}")
        raise
    except Exception as e:
        logging.error(f"❌ Error occurred during data ingestion: {e}")
        raise

if __name__ == '__main__':
    df = data_ingestion()
    print(df.head())
