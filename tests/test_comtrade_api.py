
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

# Adjust path to import src module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.comtrade_api import get_comtrade_data

class TestComtradeApi(unittest.TestCase):

    @patch('src.comtrade_api.comtradeapicall.previewFinalData')
    def test_get_comtrade_data_success(self, mock_preview):
        """Test successful data retrieval from the Comtrade API."""
        # Mock the return value of the API call to be a DataFrame
        mock_df = pd.DataFrame({
            'period': [2022],
            'reporterDesc': ['USA'],
            'partnerDesc': ['World'],
            'cmdDesc': ['Cars'],
            'primaryValue': [1000000]
        })
        mock_preview.return_value = mock_df

        df = get_comtrade_data('842', '0', '8703')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertIn('Value', df.columns)
        self.assertEqual(df['Value'].iloc[0], 1.0)  # Value should be in millions

    @patch('src.comtrade_api.comtradeapicall.previewFinalData')
    def test_get_comtrade_data_no_data(self, mock_preview):
        """Test the case where the API returns no data."""
        mock_preview.return_value = pd.DataFrame() # Return an empty DataFrame

        df = get_comtrade_data('1', '1', '1')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    @patch('src.comtrade_api.comtradeapicall.previewFinalData')
    def test_get_comtrade_data_api_error(self, mock_preview):
        """Test the case where the API call raises an exception."""
        mock_preview.side_effect = Exception("API Error")

        df = get_comtrade_data('1', '1', '1')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()
