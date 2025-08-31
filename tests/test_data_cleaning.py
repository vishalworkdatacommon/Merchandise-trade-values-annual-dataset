
import unittest
import pandas as pd
import os
import sys

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_cleaning_script import clean_and_treat_outliers

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """Set up a temporary test environment."""
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a dummy CSV file for testing
        self.test_input_path = os.path.join(self.test_dir, 'test_input.csv')
        self.test_output_path = os.path.join(self.test_dir, 'test_output.csv')
        
        data = {
            'Year': [2000, 2001, 2002, 2003, 2004],
            'Value': [100, 110, 1000, 130, 140] # 1000 is an outlier
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_input_path, index=False)

    def tearDown(self):
        """Clean up the test environment."""
        os.remove(self.test_input_path)
        os.remove(self.test_output_path)
        os.rmdir(self.test_dir)

    def test_outlier_treatment(self):
        """Test that the outlier is correctly identified and treated."""
        clean_and_treat_outliers(self.test_input_path, self.test_output_path)
        
        # Load the output and check if the outlier was handled
        result_df = pd.read_csv(self.test_output_path)
        
        # The outlier (1000) should be replaced by the rolling median (130)
        self.assertNotIn(1000, result_df['Value'].values)
        self.assertIn(130.0, result_df['Value'].values)

if __name__ == '__main__':
    unittest.main()
