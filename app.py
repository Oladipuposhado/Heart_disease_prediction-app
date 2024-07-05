# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:41:17 2024

@author: shadopc
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load the model
with open("heart_disease.pkl", "rb") as f:
   rf = pickle.load(f)
    
#Title 
st.title('Heart Attack prediction App')

#User inputs
def user_input_features():
    age =st.number_input('Age', value= None, min_value=29, max_value =54, placeholder='type a number')
    sex = st.selectbox('Sex',['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0
    cp = st.slider('Chest Pain Type', 0,1,2,3)
    trestbps = st.slider('Resting blood pressure', 94,200,130)
    chol = st.slider('Serum Cholesterol', 126,564,246)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True','False'])
    fbs = 0 if fbs == 'False' else 1
    restecg = st.slider('Resting Electrocardigraphic Results', 0,1,2) 
    thalach = st.number_input('Maximum Heart rate Achieved', 71, 202, 150)
    exang = st.slider('Exercise induced Angina', 0,1,2)
    oldpeak = st.slider('ST Depression Induvced by Exercise', 0.0,6.2,1.0)
    slope = st.slider('Slope of the Peak Exercise ST Segment', 0,1,2)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0,1,2)
    thal = st.slider('Thalassemia', 0,1,2)
    
    data = {'age':age,
            'sex': sex, 
            'cp':cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg, 
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak, 
            'slope': slope, 
            'ca': ca,
            'thal': thal}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

#prediction button
if st.button('Predict'):
     #Ensure the input data matches the model's expected features
    expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    if set(input_df.columns) != set(expected_features):
        st.error(f"Input features do not match expected features: {expected_features}")
    elif 'rf' in locals():
        try:
            prediction = rf.predict(input_df)
            st.subheader('Prediction')
            st.write('You are at a High Risk of Having a Heart Attack' if prediction[0] == 1 else 'You are at a Low Risk of Having a Heart Attack')
        except ValueError as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model is not loaded or input features are incorrect, prediction cannot be made.")
        

    