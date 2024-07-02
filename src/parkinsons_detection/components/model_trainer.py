import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from tensorflow.keras.callbacks import Callback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import tensorflow as tf

from src.parkinsons_detection.exception import CustomException
from src.parkinsons_detection.logger import logging
from src.parkinsons_detection.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred, average='weighted')
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, f1, roc_auc

    def cnn_feature_extraction(self,X_train, y_train, X_test, y_test):
        try:
            logging.info(f'in cnn_extrtaction X train shape: {X_train.shape} ')
            logging.info(f'in cnn_extrtaction X test shape: {X_test.shape} ')
            X_train_cnn = np.array(X_train).reshape(-1, 2, 28, 1)
            X_test_cnn = np.array(X_test).reshape(-1, 2, 28, 1)
            y_train = np.array(y_train)
            y_test = np.array(y_test)


            model_cnn = Sequential()
            K.set_image_data_format('channels_last')
            model_cnn.add(Conv2D(128,3,3, padding='same', input_shape=(2,28,1),activation='relu',name = 'convo_2d_1'))
            # model_cnn.add(MaxPooling2D(pool_size=(1,1),padding='same',name = 'maxpool_1'))
            model_cnn.add(Dropout(0.6))
            model_cnn.add(Conv2D(64, 3, 3, activation= 'relu',padding='same' ,name = 'convo_2d_2'))
            # model_cnn.add(MaxPooling2D(pool_size=(1,1),padding='same',name = 'maxpool_2'))
            model_cnn.add(Dropout(0.6))
            model_cnn.add(Conv2D(32, 5, 5, activation= 'relu',padding='same' ,name = 'convo_2d_3'))
            model_cnn.add(Dropout(0.6))
            # model_cnn.add(Conv2D(32, 5, 5, activation= 'relu',padding='same' ,name = 'convo_2d_4'))
            # model_cnn.add(Dropout(0.5))
            model_cnn.add(Flatten(name = 'flatten'))
            model_cnn.add(Dense(128, activation= 'relu',name = 'dense_layer1' ))
            model_cnn.add(Dense(64, activation= 'relu',name = 'dense_layer_2' ))
            model_cnn.add(Dense(1, activation= 'sigmoid' ))
            model_cnn.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

            logging.info(f'model summary :{print(model_cnn.summary())}')

            checkpoint_filepath = '/tmp/ckpt/checkpoint.weights.h5'
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)

            

            class StopAtAccuracy(Callback):
                def __init__(self, accuracy=0.8):
                    super(StopAtAccuracy, self).__init__()
                    self.accuracy = accuracy

                def on_epoch_end(self, epoch, logs=None):
                    if logs.get('val_accuracy') >= self.accuracy:
                        print(f"\nReached {self.accuracy*100}% validation accuracy, stopping training!")
                        self.model.stop_training = True
            stop_at_accuracy_callback = StopAtAccuracy(accuracy=0.91)

            history = model_cnn.fit(
                X_train_cnn, y_train,
                epochs=120,
                batch_size=60,
                validation_data=(X_test_cnn, y_test),
                shuffle=True,
                callbacks=[model_checkpoint_callback, stop_at_accuracy_callback]
            )
            model_cnn.predict(X_train_cnn)
            model_feat = Model(inputs = model_cnn.get_layer('convo_2d_1').input, outputs=model_cnn.get_layer('convo_2d_3').output)

            feat_train = model_feat.predict(X_train_cnn)
            feat_test = model_feat.predict(X_test_cnn)

            return feat_train,y_train,feat_test,y_test
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info(f'X_train: {X_train.shape}')
            X_train_cnn, y_train, X_test_cnn, y_test = self.cnn_feature_extraction(X_train, y_train, X_test, y_test)

            models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost Classifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params = {
    
    "Random Forest": {
        'n_estimators': [8, 16, 32],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    "Gradient Boosting": {
        'n_estimators': [8, 16],
        'learning_rate': [.1, .05, .01],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    "XGBoost Classifier": {
        'learning_rate': [.1, .05, .01],
        'n_estimators': [8, 16, 32],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
    },
    "CatBoost Classifier": {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Classifier": {
        'n_estimators': [8, 16, 32],
        'learning_rate': [.1, .05, .01],
        'algorithm': ['SAMME', 'SAMME.R']
    }
}

            model_report = evaluate_models(X_train_cnn.reshape(X_train_cnn.shape[0], -1), y_train, X_test_cnn.reshape(X_test_cnn.shape[0], -1), y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test.reshape(X_test.shape[0], -1))
            accuracy, f1, roc_auc = self.eval_metrics(y_test, predicted)

            logging.info(f"Accuracy: {accuracy}, F1 Score: {f1}, ROC AUC Score: {roc_auc}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
