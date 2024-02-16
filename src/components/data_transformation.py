import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import logger
from src.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.dataset_name = None
        
    def get_text_data(self):
        with open(self.config.data_path, 'r') as file:
            book = file.read()
        filepath = Path(self.config.data_path)
        filedir, filename = os.path.split(filepath)
        logger.info(f"Loaded text data from ******* {filename} ******* succcessfully")
        self.config.tokenizer_name = filename
        self.dataset_name = filename.split('.')[0]
        logger.info("Tokenizer name set")
        
        return book
        

        
        
        
    