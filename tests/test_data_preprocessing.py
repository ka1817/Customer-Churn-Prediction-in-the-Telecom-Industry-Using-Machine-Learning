import unittest
from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_df = data_ingestion()

    def test_preprocess_shape(self):
        X, y, preprocessor = preprocess_data(self.raw_df)
        self.assertEqual(len(X), len(y), "Features and target must have the same number of Samples")

    