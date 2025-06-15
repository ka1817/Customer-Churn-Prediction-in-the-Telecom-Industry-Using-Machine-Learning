import unittest
import pandas as pd
from src.data_ingestion import data_ingestion

class TestDataIngestion(unittest.TestCase):
    def test_data_ingestion_output(self):
        df = data_ingestion()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn('Churn', df.columns, "'Churn' column should be present")
