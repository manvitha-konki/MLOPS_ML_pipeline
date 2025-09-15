import os
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.ensemble import RandomForestClassifier

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Creating a logger object - logging configuration
logger = logging.getLogger('model_building')    # data_ingestion is logger object name
logger.setLevel('DEBUG')            # DEBUG level - Gives function starts and ends msgs (Basic level Msgs)

# Making Console handler where all the logs are printing in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Making File handler where all the logs are saved in file format
log_file_path = os.path.join(log_dir, 'model_building.log')
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



# Adding params from params.yaml file
def load_params(params_path: str)-> dict:
    try:
        with open(params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            logger.debug(f'Params loaded successfully from {params_path}')
            return params
    except FileNotFoundError as e:
        logger.error(f'Params file not found: {e}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'Error parsing YAML file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected Error: {e}')
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f'Data loaded from {file_path} with shape {df.shape}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except FileNotFoundError as e:
        logger.error(f'File not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, random_state: int) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters')
        clf = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)
        
        logger.debug(f'Model training started with {X_train.shape[0]} samples')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        return clf
    
    except ValueError as e:
        logger.error(f'ValueError during model training: {e}')
        raise
    except Exception as e:
        logger.error(f'Error during model training: {e}')
        raise


def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug(f'Model saved to {file_path}')
    except FileNotFoundError as e:
        logger.error(f'File path not found: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occurred while saving the model: {e}')
        raise

def main():
    try:
        params = load_params('params.yaml')
        n_estimators = params['model_building']['n_estimators']
        random_state = params['model_building']['random_state']
        # params = {'n_estimators': 25, 'random_state': 42}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, n_estimators, random_state)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error(f'Failed to complete the model building process: {e}')
        print(f"Error: {e}")

if __name__ == '__main__':
    main()