import sys
import pandas as pd
import os
import numpy as np
from src.parkinsons_detection.exception import CustomException
from src.parkinsons_detection.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            # (data_scaled)
            data_scaled = np.array(data_scaled).reshape(-1, 2, 28, 1).reshape(data_scaled.shape[0], -1)
            print((data_scaled.shape))
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {key: [value] for key, value in self.data.items()}
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
