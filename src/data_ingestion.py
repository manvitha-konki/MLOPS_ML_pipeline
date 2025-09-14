# This file get the data from some sourse like MongoDB, AWS S3, GCP, Azure Blob Storage etc

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Creating a logger object - logging configuration
logger = logging.getLogger('data_ingestion')    # data_ingestion is logger object name
logger.setLevel('DEBUG')            # DEBUG level - Gives function starts and ends msgs (Basic level Msgs)

# Making Console handler where all the logs are printing in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Making File handler where all the logs are saved in file format
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter object
# Format: time, name: data_ingestion, level: DEBUG, message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# For these 2 types of handlers we are defining format
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# For this logger we are defining the handlers and formats
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Load data from csv files
def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded successfully from {data_url}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse CSV file from {data_url} {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data from {data_url}: {e}')
        raise


# Preprocess the data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed successfully')
        return df
    except Exception as e:
        logger.error(f'Error during data preprocessing: {e}')
        raise

# Save train and test datasets
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_path = os.path.join(raw_data_path, 'train.csv')
        test_path = os.path.join(raw_data_path, 'test.csv')
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.debug(f'Train data saved successfully at {train_path}')
        logger.debug(f'Test data saved successfully at {test_path}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

def main():
    try:
        test_size = 0.21
        data_path = 'https://raw.githubusercontent.com/manvitha-konki/MLOPS_ML_pipeline/refs/heads/main/experiments/spam.csv'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, './data')
    except Exception as e:
        logger.error(f'Error in main data ingestion process: {e}')
        print(f"Error: {e}")


if __name__ == '__main__':
    main()