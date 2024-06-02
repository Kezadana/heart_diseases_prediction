import streamlit as st
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import os
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Join the current directory with the file name
file_path = os.path.join(current_directory, 'heart_disease_data.csv')

# Check if the file exists
if os.path.exists(file_path):
    # Read the CSV file
    heart_data = pd.read_csv(file_path)
else:
    print("File not found:", file_path)


# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Split the data into features and target variable
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

# Load the trained model (in real usage, you'd skip training and just load)
model = joblib.load('logistic_regression_model.pkl')

# Streamlit UI
st.title('Heart Disease Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=44)
sex = st.selectbox('Sex', [0, 1])
cp = st.number_input('Chest Pain Type', min_value=0, max_value=3, value=0)
trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)
chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, max_value=600, value=169)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=250, value=144)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=2.8)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
ca = st.number_input('Number of Major Vessels Colored by Flourosopy', min_value=0, max_value=4, value=0)
thal = st.selectbox('Thalassemia', [0, 1, 2, 3])

# Prediction
if st.button('Predict'):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        st.success('The Person does not have a Heart Disease')
    else:
        st.error('The Person has Heart Disease')
