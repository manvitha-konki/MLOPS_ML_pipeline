
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer


# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Creating a logger object - logging configuration
logger = logging.getLogger('feature_engineering')    # data_ingestion is logger object name
logger.setLevel('DEBUG')            # DEBUG level - Gives function starts and ends msgs (Basic level Msgs)

# Making Console handler where all the logs are printing in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Making File handler where all the logs are saved in file format
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
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


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded successfully from {file_path}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse CSV file from {file_path} {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading data from {file_path}: {e}')
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

        # Clean NaN -> empty string
        train_data['text'] = train_data['text'].fillna("").astype(str)
        test_data['text'] = test_data['text'].fillna("").astype(str)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('Bag of words applied and data transformed successfully')
        return train_df, test_df
    
    except Exception as e:
        logger.error(f'Error during Bag of words transformation: {e}')
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        max_features = 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()