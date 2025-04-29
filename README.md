## Parkinson's Disease Prediction Web App

This project is a **Machine Learning-powered Streamlit web application** that predicts the likelihood of Parkinson’s Disease using biomedical voice measurements. 
The model is trained using the [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Features

- Clean, interactive UI built using **Streamlit**
- Predict Parkinson’s using 22 voice measurement parameters
- Machine Learning model trained with **scikit-learn**
- Lightweight and easy to deploy anywhere

## Setup Instructions

Follow the steps below to run the project locally.

### Prerequisites

- Python 3.9
- Git

### Clone the Repository

### Install Dependencies
  It is recommended to create a virtual environment:
  python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Then install the required packages:
pip install -r requirements.txt

### Run the Application
streamlit run PP_WebApp.py
This will launch the app in your default browser (usually at http://localhost:8501/).


## Model Details
Trained using: Logistic Regression

Features Used: 22 biomedical voice measurements

Dataset: UCI Parkinson’s Disease Data Set

## Notes
Ensure trained_model.sav is placed in the root directory of the app.

You can retrain the model in a Colab notebook using the same dependency versions mentioned above and export the .sav file again if needed.


