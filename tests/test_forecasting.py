
import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from forecasting_script import forecast_sarimax
from advanced_forecasting_script import forecast_lstm

class TestForecasting(unittest.TestCase):

    def setUp(self):
        """Set up a test DataFrame."""
        data = {
            'Year': pd.to_datetime(['2010', '2011', '2012', '2013', '2014', '2015']),
            'Value': [100, 110, 120, 130, 140, 150],
            'GDP_USD': [1000, 1100, 1200, 1300, 1400, 1500]
        }
        self.test_df = pd.DataFrame(data)

    def test_forecast_sarimax(self):
        """Test the SARIMAX forecasting function."""
        forecast_df = forecast_sarimax(self.test_df)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 5)
        self.assertIn('mean', forecast_df.columns)

    def test_forecast_lstm(self):
        """Test the LSTM forecasting function."""
        forecast_df = forecast_lstm(self.test_df)
        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), 5)
        self.assertIn('mean', forecast_df.columns)

if __name__ == '__main__':
    unittest.main()
