Parkinson's Detection Using Gait Analysis
This repository contains the code and resources for detecting Parkinson's disease using gait analysis through a combination of Convolutional Neural Networks (CNN) and XGBoost, with an emphasis on Explainable AI (XAI) techniques.


Table of Contents
Introduction
Project Structure
Dataset
Model Architecture
Explainable AI
Results
Installation
Usage
Contributing
License
Introduction
Parkinson's disease is a neurodegenerative disorder that affects movement. Gait analysis, which studies the manner of walking, can provide valuable insights for detecting Parkinson's. This project leverages deep learning and machine learning techniques to analyze gait patterns and detect Parkinson's disease with high accuracy.

Project Structure
markdown
Copy code
parkinsons-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── cnn_model.py
│   ├── xgboost_model.py
│   └── explainable_ai.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_cnn_model.py
│   ├── test_xgboost_model.py
│   └── test_explainable_ai.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
Dataset
The dataset consists of gait recordings from individuals diagnosed with Parkinson's disease and healthy controls. The data is preprocessed and segmented to be used as input for the CNN.

Model Architecture
The model architecture comprises two main components:

Convolutional Neural Network (CNN):

The CNN is used to extract features from the gait data. It includes three convolutional layers followed by pooling layers.
XGBoost:

The extracted features from the CNN are used as input to the XGBoost classifier, which predicts whether an individual has Parkinson's disease.

Explainable AI
To ensure the model's predictions are interpretable, we employ Explainable AI techniques. This involves using SHAP (SHapley Additive exPlanations) to explain the impact of each feature on the model's output.


Results
The combined CNN and XGBoost model achieved an accuracy of 91% on the test set. The use of Explainable AI provided insights into which features were most influential in the detection of Parkinson's disease.


Installation
To install the required dependencies, run:

bash
Copy code
pip install -r requirements.txt
Usage
Data Preprocessing:

Prepare the dataset by running the preprocessing script.
bash
Copy code
python src/data_processing.py
Train the CNN Model:

Train the CNN model to extract features from the gait data.
bash
Copy code
python src/cnn_model.py
Train the XGBoost Model:

Train the XGBoost classifier using the features extracted by the CNN.
bash
Copy code
python src/xgboost_model.py
Explain the Model Predictions:

Use Explainable AI techniques to interpret the model's predictions.
bash
Copy code
python src/explainable_ai.py
Contributing
Contributions are welcome! Please read the CONTRIBUTING.md file for guidelines on how to contribute to this project.

License
This project is licensed under the MIT License. See the LICENSE file for more details.


