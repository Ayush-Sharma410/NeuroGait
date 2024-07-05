# Parkinson's Detection Using Gait Analysis

This repository contains the code and resources for detecting Parkinson's disease using gait analysis through a combination of Convolutional Neural Networks (CNN) and XGBoost, with an emphasis on Explainable AI (XAI) techniques.

![Project Banner](images\parkinsons-disease-torn-paper-concept.webp)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Explainable AI](#explainable-ai)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Parkinson's disease is a neurodegenerative disorder that affects movement. Gait analysis, which studies the manner of walking, can provide valuable insights for detecting Parkinson's. This project leverages deep learning and machine learning techniques to analyze gait patterns and detect Parkinson's disease with high accuracy.

## Project Structure

parkinsons-detection/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ └── analysis.ipynb
│
├── src/
│ ├── init.py
│ ├── data_processing.py
│ ├── cnn_model.py
│ ├── xgboost_model.py
│ └── explainable_ai.py
│
├── tests/
│ ├── init.py
│ ├── test_data_processing.py
│ ├── test_cnn_model.py
│ ├── test_xgboost_model.py
│ └── test_explainable_ai.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py


## Dataset

The dataset consists of gait recordings from individuals diagnosed with Parkinson's disease and healthy controls. The data is preprocessed and segmented to be used as input for the CNN.

## Model Architecture

The model architecture comprises two main components:

1. **Convolutional Neural Network (CNN)**:
   - The CNN is used to extract features from the gait data. It includes three convolutional layers followed by pooling layers.

2. **XGBoost**:
   - The extracted features from the CNN are used as input to the XGBoost classifier, which predicts whether an individual has Parkinson's disease.

![Model Architecture](path/to/your/model_architecture_image.jpg)


## Results

The combined CNN and XGBoost model achieved an accuracy of **91%** on the test set. The use of Explainable AI provided insights into which features were most influential in the detection of Parkinson's disease.

![Results](path/to/your/results_image.jpg)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
