import os
from src.logger import logger
from src.utils.common import get_size
from src.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config     # will contain the data dir path as "data" as config[data_ingestion] = data
        logger.info(f"Raw data path is - {config.root_dir}")
        
        