import streamlit as st
import pandas as pd
import joblib
import logging
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.DEBUG)


class HeartDiseaseInput(BaseModel):
    age: int = Field(ge=29, le=54)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=94, le=200)
    chol: int = Field(ge=126, le=246)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=71, le=220)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0.0, le=6.2)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=2)
    thal: int = Field(ge=0, le=2)


@st.cache_resource
def load_model():
    try:
        with open("heart_disease.joblib", "rb") as f:
            model = joblib.load(f)
        return model
    except Exception as e:
        logging.exception("Error loading model")
        st.error(f"Error loading the model: {str(e)}")
        return None


def user_input_features():
    age = st.number_input('Age', value=29, min_value=29, max_value=54)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0
    cp = st.slider('Chest Pain Type', 0, 3, 0)
    trestbps = st.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.slider('Serum Cholesterol', 126, 246, 126)  # ✅ added default value
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['False', 'True'])
    fbs = 1 if fbs == 'True' else 0
    restecg = st.slider('Resting ECG Results', 0, 2, 0)  # ✅ was missing
    thalach = st.number_input('Maximum Heart Rate Achieved', 71, 220, 150)
    exang = st.selectbox('Exercise Induced Angina (0 = No, 1 = Yes)', [0, 1], index=0)
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 0.0, step=0.1)  # ✅ max was wrong (1.0 vs 6.2)
    slope = st.slider('Slope of the Peak Exercise ST Segment', 0, 2, 0)
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 2, 0)
    thal = st.slider('Thalassemia', 0, 2, 0)

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,  # ✅ now included
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
    }
    return pd.DataFrame(data, index=[0])


def make_prediction(model, input_data):
    try:
        # ✅ Fix: convert the DataFrame row to a dict of scalar values
        input_dict = {col: input_data[col].iloc[0] for col in input_data.columns}

        validated_input = HeartDiseaseInput(**input_dict)

        if hasattr(validated_input, "model_dump"):
            input_dict_validated = validated_input.model_dump()
        else:
            input_dict_validated = validated_input.dict()

        input_df = pd.DataFrame([input_dict_validated])

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


# ✅ Fix: load model and inputs BEFORE the button, and only ONE predict button
st.title('Heart Attack Prediction App')

rf_model = load_model()
user_input = user_input_features()

if st.button('Predict Heart Attack Risk'):
    make_prediction(rf_model, user_input)

