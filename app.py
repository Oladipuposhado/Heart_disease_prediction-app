# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:41:17 2024

@author: shadopc
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import logging
from pydantic import BaseModel, Field, ValidationError


logging.basicConfig(level=logging.DEBUG)

class HeartDiseaseInput(BaseModel):
    age: int = Field(ge=29, le=54)  # Age between 29 and 54
    sex: int = Field(ge=0, le=1, description= "1 for male 0 for female")
    cp: int = Field(ge=0, le=3) #Chest pain (0-3)
    trestbps: int = Field(ge=94, le=200) #Resting blood pressure (94-200)
    chol: int = Field(ge=126, le=246) #Cholesterol serum (126- 246)
    fbs: int = Field(ge=0, le=1,)  #'Fasting Blood Sugar > 120 mg/dl' (0-1)
    restecg: int = Field(ge=0, le=2) #Resting electrocardigraphic  results  (0-2
    thalach: int = Field(ge=71, le=150) #Maximum heart rate achieved (71-150) 
    exang: int = Field(ge=0, le=2) #Exercise induced angina(0-2)
    oldpeak: float = Field(ge=0.0, le=1.0) #depression induced by exercise
    slope: int = Field(ge=0, le=2) #Slope of the Peak Exercise ST Segment (0-2)
    ca: int = Field(ge=0, le=2) #Number of Major Vessels Colored by Fluoroscopy (0-2)
    thal: int = Field(ge=0, le=2) #Thalassemia (0-2)




# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open("heart_disease.pkl", "rb") as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        logging.exception("Error loading model")
        st.error(f"Error loading the model: {str(e)}")
        return None

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

#input_df = user_input_features()


# Main prediction function
def make_prediction(model, input_data):
    try:
        validated_input = HeartDiseaseInput(**input_data)
        input_df = pd.DataFrame([validated_input.dict()])

        # Ensure model is loaded
        if model is not None:
            prediction = model.predict(input_df)
            result = 'High Risk of Heart Attack' if prediction[0] == 1 else 'Low Risk of Heart Attack'
            st.subheader('Prediction Result')
            st.success(result)
        else:
            st.error("The model is not loaded, unable to make predictions.")
    except ValidationError as e:
        st.error(f"Input validation error: {e}")
    except Exception as e:
        logging.exception("Error during prediction")
        st.error(f"An error occurred during prediction: {str(e)}")

# Streamlit app interface
#st.title('Heart Attack Prediction App')

# Load the model once (cached)
rf_model = load_model()

# Get user input
user_input = user_input_features()

# Prediction button
if st.button('Predict'):
    make_prediction(rf_model, user_input)
# Prediction button
#if st.button('Predict'):
    #expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    #if set(input_df.columns) != set(expected_features):
        #st.error(f"Input features do not match expected features: {expected_features}")
    #elif 'rf' in locals() and isinstance(rf, joblib.base.DeserializedObject):
        #try:
            #prediction = rf.predict(input_df)
           # st.subheader('Prediction')
            #except Exception as e:
            #logging.exception("Error during prediction")
            #st.error(f"Prediction error: {str(e)}")
    #else:
        #st.error("Model is not loaded or input features are incorrect, prediction cannot be made.")

    
