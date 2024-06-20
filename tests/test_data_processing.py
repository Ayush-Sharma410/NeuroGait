import unittest
from src.data_processing import load_data, preprocess_data

class TestDataProcessing(unittest.TestCase):
    def test_load_data(self):
        df = load_data('data/raw/sample.csv')
        self.assertIsNotNone(df)

    def test_preprocess_data(self):
        df = preprocess_data(pd.DataFrame({'a': [1, 2, 3]}))
        self.assertIsNotNone(df)
