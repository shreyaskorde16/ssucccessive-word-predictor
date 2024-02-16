from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity import (DataIngestionConfig, DataValidationConfig, 
                                      DataTransformationConfig)



class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH, params_filepath= PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
         # will create artifacts directory, we have to pas it in list
        create_directories([self.config.artifacts_root])  
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:   #Finally, it returns this DataIngestionConfig object.
        raw_data = self.config.data_ingestion
        
        #creating data dir
        create_directories([raw_data])
        
        data_ingestion_config = DataIngestionConfig(
                    root_dir=raw_data      #  we can add as many paths as in data_ingestion dict in config.yaml
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,           
            
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
           root_dir = config.root_dir,
           data_path=config.data_path,
           tokenizer_name=config.tokenizer_name            
        )
        
        return data_transformation_config
        