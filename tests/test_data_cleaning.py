
import unittest
import pandas as pd
import os
import sys
import numpy as np

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_cleaning_script import clean_and_treat_outliers

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """Set up a test DataFrame."""
        data = {
            'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006],
            'Value': [100, 110, 1000, 130, 140, 150, 160] # 1000 is an outlier
        }
        self.test_df = pd.DataFrame(data)

    def test_outlier_treatment(self):
        """Test that the outlier is correctly identified and treated."""
        cleaned_df = clean_and_treat_outliers(self.test_df)
        
        # The outlier (1000) should be replaced by the rolling median.
        # In this case, the window is 5, centered. For 2002, the window is [2000, 2001, 2002, 2003, 2004]
        # The values are [100, 110, 1000, 130, 140]. The median is 130.
        self.assertFalse(1000 in cleaned_df['Value'].values)
        
        # Check if the value for the year 2002 is close to the median
        self.assertTrue(np.isclose(cleaned_df.loc[cleaned_df.index.year == 2002, 'Value'].iloc[0], 130.0))

if __name__ == '__main__':
    unittest.main()
