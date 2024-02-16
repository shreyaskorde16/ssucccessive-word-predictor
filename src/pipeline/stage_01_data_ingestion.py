from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.logger import logger



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
        except Exception as e:
            raise e   