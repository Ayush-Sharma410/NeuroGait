from src.parkinsons_detection.logger import logging
from src.parkinsons_detection.exception import CustomException
from src.parkinsons_detection.components.data_ingestion import DataIngestion
from src.parkinsons_detection.components.data_ingestion import DataIngestionConfig
from src.parkinsons_detection.components.data_transformation import DataTransformationConfig,DataTransformation
from src.parkinsons_detection.components.model_trainer import ModelTrainerConfig,ModelTrainer

import sys

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_= data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        ## Model Training
        model_trainer=ModelTrainer()
        logging.info(f'train arr type {type(train_arr)}')
        logging.info(f'train arr shape {(train_arr.shape)}')
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)