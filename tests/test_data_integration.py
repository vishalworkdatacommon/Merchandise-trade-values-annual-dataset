
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_integration_script import integrate_external_data

class TestDataIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a test DataFrame."""
        data = {
            'Year': [2000, 2001, 2002],
            'Value': [100, 110, 120]
        }
        self.test_df = pd.DataFrame(data)

    @patch('data_integration_script.wb.data.DataFrame')
    def test_integrate_external_data_success(self, mock_wb_data):
        """Test successful integration of GDP data."""
        gdp_data = {
            'YR2000': [1e12],
            'YR2001': [1.1e12],
            'YR2002': [1.2e12]
        }
        mock_gdp_df = pd.DataFrame(gdp_data).transpose()
        mock_gdp_df.columns = ['CHN']
        mock_wb_data.return_value = mock_gdp_df

        enriched_df = integrate_external_data(self.test_df, country_code="CHN")

        self.assertIsInstance(enriched_df, pd.DataFrame)
        self.assertIn('GDP_USD', enriched_df.columns)
        self.assertFalse(enriched_df['GDP_USD'].isnull().any())
        self.assertEqual(len(enriched_df), 3)

    @patch('data_integration_script.wb.data.DataFrame')
    def test_integrate_external_data_api_error(self, mock_wb_data):
        """Test the case where the World Bank API call fails."""
        mock_wb_data.side_effect = Exception("API Error")

        enriched_df = integrate_external_data(self.test_df, country_code="CHN")

        self.assertIsInstance(enriched_df, pd.DataFrame)
        self.assertIn('GDP_USD', enriched_df.columns)
        self.assertTrue((enriched_df['GDP_USD'] == 0).all())

if __name__ == '__main__':
    unittest.main()
