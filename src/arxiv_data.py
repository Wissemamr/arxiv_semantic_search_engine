# ===================================================================
# ArXiv Data handling
# ===================================================================

import os
import pandas as pd
import logging
from colorama import Fore
from utils import load_config
from icecream import ic
from typing import Dict, Any

config = load_config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class ArxivData:
    """
    A class to handle operations on an ArXiv dataset stored in a CSV file.
    """

    def __init__(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        logging.info(Fore.GREEN + f"Loading data from: {filepath}" + Fore.RESET)
        self.df = pd.read_csv(filepath)

    def clean_df(self) -> pd.DataFrame:
        """
        Clean the dataset by removing rows with missing values and duplicates.
        """
        initial_rows = len(self.df)
        logging.info(f"Initial rows: {initial_rows}")
        self.df.dropna(subset=["abstract", "title", "label"], inplace=True)
        self.df.drop_duplicates(subset=["abstract"], inplace=True)
        cleaned_rows = len(self.df)
        logging.info(f"Rows after cleaning: {cleaned_rows}")
        logging.info(f"Removed {initial_rows - cleaned_rows} rows during cleaning.")
        return self.df

    def save_cleaned_df(self, output_path: str) -> None:
        """
        Save the cleaned DataFrame to a specified output path.
        """
        try:
            self.df.to_csv(output_path, index=False)
            logging.info(
                Fore.GREEN + f"Cleaned data saved to: {output_path}" + Fore.RESET
            )
        except Exception as e:
            logging.error(Fore.RED + f"Failed to save cleaned data: {e}" + Fore.RESET)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        """
        return {
            "df_shape": self.df.shape,
            "df_columns": list(self.df.columns),
            "label_counts": self.df["label"].value_counts().to_dict(),
        }


if __name__ == "__main__":
    try:
        data = ArxivData(filepath=config.data.filepath)
        data.clean_df()
        output_path = "../data/ArXiv-10_.csv"
        data.save_cleaned_df(output_path=output_path)
        df_stats = data.get_stats()
        ic(df_stats)
    except Exception as e:
        logging.error(Fore.RED + f"Error: {e}" + Fore.RESET)
