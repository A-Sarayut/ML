import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
 
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
model = pickle.load(open('treemodel.sav', 'rb'))
st.sidebar.header('prpare data')
st.title('Stroke Prediction')

def prepare_info():
    gender = st.sidebar.radio("What's your gender ?",('Male', 'Female', 'Other'))
    age =st.sidebar.slider("age",16,100,16)
    hypertension =st.sidebar.radio("hypertension ",("No","Yes"))
    heart_disease=st.sidebar.radio("heart_disease ",("No","Yes"))
    ever_married =st.sidebar.radio("ever_married ",("No","Yes"))
    work_type=st.sidebar.radio("work_type ",("Private","Self-employed ","children","Never_worked"))
    Residence_type  =st.sidebar.radio("Residence_type ",("Urban","Rural"))
    avg_glucose_level=st.sidebar.slider("avg_glucose_level ",50,250,50)
    bmi =st.sidebar.slider("bmi ",15,40,15)
    smoking_status  =st.sidebar.radio("smoking_status ",("never smoked","Unknown","formerly smoked","smokes"))


    info = {
        "gender":gender,
        "age":age,
        "hypertension":hypertension,
        "heart_di":heart_disease,
        "ever_married":ever_married,
        "work_type":work_type,
        "Residence_type":Residence_type,
        "avg_glucose_level":avg_glucose_level,
        "bmi":bmi,
        "smoking_status":smoking_status
    }
    init_info = pd.DataFrame(info, index=[0])
    le = LabelEncoder()
    init_info['gender'] = le.fit_transform(init_info['gender'])
    init_info['ever_married'] = le.fit_transform(init_info['ever_married'])
    init_info['work_type'] = le.fit_transform(init_info['work_type'])
    init_info['Residence_type'] = le.fit_transform(init_info['Residence_type'])
    init_info['smoking_status'] = le.fit_transform(init_info['smoking_status'])
        
    return init_info

user_info = prepare_info()
st.write(user_info)