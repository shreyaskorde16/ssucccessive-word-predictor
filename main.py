from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationPipeline
from src.logger import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"{'>'*10} stage {STAGE_NAME} started {'<'*10}")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"{'>'*10} stage {STAGE_NAME} completed {'<'*10}\n\nX==========X") 
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f"{'>'*10} stage {STAGE_NAME} started {'<'*10}")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"{'>'*10} stage {STAGE_NAME} completed {'<'*10}\n\nX==========X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data transformation stage"
try:
    logger.info(f"{'>'*10} stage {STAGE_NAME} started {'<'*10}")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f"{'>'*10} stage {STAGE_NAME} completed {'<'*10}\n\nX==========X")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Training stage"
try:
    logger.info(f"{'>'*10} Stage {STAGE_NAME} Started {'<'*10}")
    X = data_transformation.X
    y = data_transformation.y
    logger.info(f"{'>'*10} Stage {STAGE_NAME} completed {'<'*10}\n\nX==========X")
except Exception as e:
    logger.exception(e)
    raise e

