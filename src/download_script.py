# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script downloads and unzips the Merchandise Trade Values Annual Dataset.
"""

import os
import sys
from absl import app, logging

import download_util_script


def download_source(url: str, base_folder: str, input_folder: str):
    """Downloads and processes a single data source.

    Args:
        url: The URL of the zip file to download.
        base_folder: The base directory for downloads.
        input_folder: The directory to store the final CSV.
    """
    try:
        zip_filename_base = os.path.basename(url)
        dataset_name = zip_filename_base.replace('.zip', '')

        zip_filepath = os.path.join(base_folder, zip_filename_base)
        original_csv_filepath = os.path.join(base_folder, f"{dataset_name}.csv")
        final_csv_filename = os.path.join(
            input_folder, f"{dataset_name.replace('_dataset', '_input')}.csv")

        logging.info(f"Downloading from: {url}")
        if not download_util_script.download_file(url, base_folder, unzip=True):
            logging.error(f"Failed to download and unzip the file from {url}.")
            return

        logging.info(f"Successfully downloaded and unzipped the file from {url}.")

        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
            logging.info(f"Removed zip file: {zip_filepath}")

        if os.path.exists(original_csv_filepath):
            os.rename(original_csv_filepath, final_csv_filename)
            logging.info(f"Renamed and moved CSV to: {final_csv_filename}")
        else:
            logging.warning(
                f"Could not find expected CSV file: {original_csv_filepath}")

    except Exception as e:
        logging.error(f"An error occurred while processing {url}: {e}")


def download_and_process_data(_):
    """
    Downloads and unzips the Merchandise Trade Values Annual Dataset.
    """
    urls = [
        "https://stats.wto.org/assets/UserGuide/merchandise_values_annual_dataset.zip",
        "https://stats.wto.org/assets/UserGuide/merchandise_indices_annual_dataset.zip"
    ]
    base_folder = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_folder, "data")
    os.makedirs(input_folder, exist_ok=True)

    logging.info(f"Output folder: {input_folder}")

    for url in urls:
        download_source(url, base_folder, input_folder)


if __name__ == "__main__":
    app.run(download_and_process_data)
