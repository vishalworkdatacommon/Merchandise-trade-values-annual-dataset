import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from comtrade_api import get_comtrade_data

class TestComtradeApi(unittest.TestCase):

    @patch('comtrade_api.requests.get')
    def test_get_comtrade_data_success(self, mock_get):
        """Test successful data retrieval from the Comtrade API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'period': 2022,
                    'reporterDesc': 'USA',
                    'partnerDesc': 'World',
                    'cmdDesc': 'Cars',
                    'primaryValue': 1000000
                }
            ]
        }
        mock_get.return_value = mock_response

        df = get_comtrade_data('842', '0', '8703')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertIn('Value', df.columns)
        self.assertEqual(df['Value'].iloc[0], 1.0)  # Value should be in millions

    @patch('comtrade_api.requests.get')
    def test_get_comtrade_data_no_data(self, mock_get):
        """Test the case where the API returns no data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': []}
        mock_get.return_value = mock_response

        df = get_comtrade_data('1', '1', '1')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    @patch('comtrade_api.requests.get')
    def test_get_comtrade_data_api_error(self, mock_get):
        """Test the case where the API returns an error."""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        df = get_comtrade_data('1', '1', '1')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

if __name__ == '__main__':
    unittest.main()