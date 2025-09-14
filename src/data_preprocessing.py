# This file preprocesses the data after getting it from data_ingestion.py

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords   
import string
import nltk
import logging

nltk.download('stopwords')  # Stopwords
nltk.download('punkt')      # Tokenizer

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Creating a logger object - logging configuration
logger = logging.getLogger('data_preprocessing')    # data_ingestion is logger object name
logger.setLevel('DEBUG')            # DEBUG level - Gives function starts and ends msgs (Basic level Msgs)

# Making Console handler where all the logs are printing in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Making File handler where all the logs are saved in file format
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
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


# Transform the data by converting into lower case, tokenization, stemmatization
def transform_data(text):
    ps = PorterStemmer()
    text = text.lower() # Convert to lowercase
    text = nltk.word_tokenize(text) # Tokenization
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] # Remove stopwords and punctuation
    text = [word for word in text if word.isalnum()] # Remove non-alphanumeric tokens
    text = [ps.stem(word) for word in text] # Stemming
    return " ".join(text) # Join the tokens back into a single string


# Preprocess the DataFrame by encoding the target column and transforming the text column
def preprocess_df(df, text_col = 'text', target_col = 'target'):
    try:
        encoder = LabelEncoder()
        df[target_col] = encoder.fit_transform(df[target_col]) # Encode target column
        logger.debug('Target column encoded successfully')

        df = df.drop_duplicates() # Drop duplicates
        logger.debug('Duplicates dropped successfully')

        df = df.dropna() # Drop missing values
        logger.debug('Missing values dropped successfully')

        # Apply text transformations
        df[text_col] = df[text_col].apply(transform_data) # Transform text column
        logger.debug('DataFrame preprocessing completed successfully')
        return df
    except Exception as e:
        logger.error(f'Column not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during text normalization: {e}')
        raise

def main(text_col = 'text', target_col = 'target'):
    try:
        # Fetch data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Train and test data loaded successfully')

        # Transform the data
        train_processed = preprocess_df(train_data, text_col, target_col)
        test_processed = preprocess_df(test_data, text_col, target_col)

        # Store the data inside data/processed
        processed_data_path = './data/interim'
        os.makedirs(processed_data_path, exist_ok=True)

        train_processed.to_csv(os.path.join(processed_data_path, 'train_processed.csv'), index=False)
        test_processed.to_csv(os.path.join(processed_data_path, 'test_processed.csv'), index=False)

        logger.debug(f'Processed data saved successfully at {processed_data_path}')
    
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
    except pd.errors.EmptyDataError as e:
        logger.error(f'Empty data: {e}')
    except Exception as e:
        logger.error(f'Error in main preprocessing process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()