import pandas as pd
from config import DataConfig
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.data = None
        self.X = None   
        self.y = None

    def load_data(self, path: str) -> 'DataProcessor':
        logger.info(f"Load data from {path}")
        self.data = pd.read_csv(path)
        logger.info(f"{len(self.data)} lines uploaded")
        return self
    
    def preprocess(self) -> 'DataProcessor':
        logger.info("Start preprocessing")

        self.data['TotalCharges'] = pd.to_numeric(
            self.data['TotalCharges'], 
            errors='coerce'
        )

        self.data['TotalCharges'] = self.data['TotalCharges'].fillna(
            self.data['TotalCharges'].median()
        )

        logger.info(f"Filled in the gaps in TotalCharges: {self.data['TotalCharges'].isna().sum()}")

        self.X = self.data.drop(
            columns=[
                self.config.target_column, 
                self.config.id_column
            ]
        )

        self.y = (self.data[self.config.target_column] == 'Yes').astype(int)
        
        logger.info(f"Preprocessing is complete. X: {self.X.shape}, y: {self.y.shape}")

        return self
    
    def get_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y
    
    def get_full_data(self):
        return self.data