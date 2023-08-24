import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

## initialize data ingestion config

@dataclass # it is a placeholder to create the class, with this we can straight away create class variables without using init method -- explore on this
class DataIngestionConfig:
    train_data_dath:str = os.path.join("artifacts", "train.csv")
    test_data_dath:str = os.path.join("artifacts", "test.csv")
    raw_data_dath:str = os.path.join("artifacts", "raw.csv")
    
## create data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))
            logging.info("Dataset read as pandas dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_dath), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_dath)
            logging.info("Raw data is created")
            
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_dath, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_dath, index=False, header=True)
            logging.info("Data Ingestion is completed")
            
            return(
                self.ingestion_config.train_data_dath,
                self.ingestion_config.test_data_dath
            )
        
        except Exception as e:
            logging.info("Exception occured at data ingestion stage")
            raise CustomException(e, sys)