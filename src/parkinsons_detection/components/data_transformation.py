import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.parkinsons_detection.utils import save_object
from src.parkinsons_detection.exception import CustomException
from src.parkinsons_detection.logger import logging
import os



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
  

    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            input_columns = ['SP_U', 'RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U',
       'SYM_U', 'R_JERK_U', 'L_JERK_U', 'ASA_U', 'ASYM_IND_U', 'TRA_U',
       'T_AMP_U', 'CAD_U', 'STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U',
       'JERK_T_U', 'SP__DT', 'RA_AMP_DT', 'LA_AMP_DT', 'RA_STD_DT',
       'LA_STD_DT', 'SYM_DT', 'R_JERK_DT', 'L_JERK_DT', 'ASA_DT',
       'ASYM_IND_DT', 'TRA_DT', 'T_AMP_DT', 'CAD_DT', 'STR_T_DT', 'STR_CV_DT',
       'STEP_REG_DT', 'STEP_SYM_DT', 'JERK_T_DT', 'SW_VEL_OP', 'SW_PATH_OP',
       'SW_FREQ_OP', 'SW_JERK_OP', 'SW_VEL_CL', 'SW_PATH_CL', 'SW_FREQ_CL',
       'SW_JERK_CL', 'TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_STRAIGHT_DUR',
       'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_SYM', 'TUG2_DUR',
       'TUG2_STEP_NUM', 'TUG2_STRAIGHT_DUR', 'TUG2_TURNS_DUR', 'TUG2_STEP_REG',
       'TUG2_STEP_SYM']
            output_columns = ["COHORT"]

            input_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='mean')),
                ('scalar',MinMaxScaler(feature_range=(0, 1)))

            ])
           

            logging.info(f"Input Columns:{input_columns}")
            logging.info(f"Output Columns:{output_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("input_pipeline",input_pipeline,input_columns),
                    
                ]

            )
            return preprocessor
            

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transormation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the train and test file")
            

            preprocessing_obj=self.get_data_transformer_object()
            # logging.info(f"checking the preprocessor obj {preprocessing_obj}")

            target_column_name="COHORT"
            input_columns = ['SP_U', 'RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U',
       'SYM_U', 'R_JERK_U', 'L_JERK_U', 'ASA_U', 'ASYM_IND_U', 'TRA_U',
       'T_AMP_U', 'CAD_U', 'STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U',
       'JERK_T_U', 'SP__DT', 'RA_AMP_DT', 'LA_AMP_DT', 'RA_STD_DT',
       'LA_STD_DT', 'SYM_DT', 'R_JERK_DT', 'L_JERK_DT', 'ASA_DT',
       'ASYM_IND_DT', 'TRA_DT', 'T_AMP_DT', 'CAD_DT', 'STR_T_DT', 'STR_CV_DT',
       'STEP_REG_DT', 'STEP_SYM_DT', 'JERK_T_DT', 'SW_VEL_OP', 'SW_PATH_OP',
       'SW_FREQ_OP', 'SW_JERK_OP', 'SW_VEL_CL', 'SW_PATH_CL', 'SW_FREQ_CL',
       'SW_JERK_CL', 'TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_STRAIGHT_DUR',
       'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_SYM', 'TUG2_DUR',
       'TUG2_STEP_NUM', 'TUG2_STRAIGHT_DUR', 'TUG2_TURNS_DUR', 'TUG2_STEP_REG',
       'TUG2_STEP_SYM']

            ## divide the train dataset to independent and dependent feature

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")
            # logging.info(f"input_features_train_df:{input_features_train_df}")
            input_feature_train_arr=preprocessing_obj.fit_transform(train_df.drop(columns=[target_column_name],axis=1))
            input_feature_test_arr=preprocessing_obj.transform(test_df.drop(columns=[target_column_name],axis=1))
            logging.info(f'target_feature_train {target_feature_train_df.shape}')

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(sys,e)