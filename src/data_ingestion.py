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