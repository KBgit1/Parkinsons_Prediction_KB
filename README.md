## Parkinson's Disease Prediction Web App

This project is a **Machine Learning-powered Streamlit web application** that predicts the likelihood of Parkinson’s Disease using biomedical voice measurements. 
The model is trained using the [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons).

Deployed link- "https://parkinsonspredictionkb-jsfsnmkgmza3eeotzbcnnk.streamlit.app/"

## Features

### Sidebar Navigation
A collapsible sidebar menu powered by streamlit-option-menu for intuitive navigation.

Sections include:

Home – Introduction to the app.

Predictor – Parkinson’s disease diagnosis page.

Help – Guide and input explanation for users.

### User-Friendly Interface
Simple and elegant layout with labeled input fields for medical parameters.

Inputs are grouped logically for easier data entry.

Responsive design that works across devices.

### Disease Diagnosis
Allows users to enter medical voice measurements (e.g., MDVP features, jitter, shimmer, etc.).

Uses a pre-trained machine learning model to predict whether the patient has Parkinson’s.

Instant result shown after clicking the “Predict” button.

### Predict Button
Submits all entered values and triggers the backend model for prediction.

Displays result as “Parkinson’s Detected” or “Healthy” with visual cues.

### Help Section
Describes what each medical term means in simple language.

Aimed at both patients and clinicians unfamiliar with the technical terms.

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


