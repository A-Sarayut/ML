from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


model_dtree = pickle.load(open('treemodel.sav', 'rb'))
#model_knn = pickle.load(open('knn_model.pkl', 'rb'))
model_rfr = pickle.load(open('rfr_model.sav', 'rb'))
st.sidebar.header('prpare data')
st.title('Stroke Prediction')


def prepare_info():
    gender = st.sidebar.radio("What's your gender ?",
                              ('Male', 'Female', 'Other'))
    age = st.sidebar.slider("age", 16, 100, 16)
    hypertension = st.sidebar.radio("hypertension ", (0, 1))
    heart_disease = st.sidebar.radio("heart_disease ", (0, 1))
    ever_married = st.sidebar.radio("ever_married ", ("No", "Yes"))
    work_type = st.sidebar.radio(
        "work_type ", ("Private", "Self-employed ", "children", "Never_worked"))
    Residence_type = st.sidebar.radio("Residence_type ", ("Urban", "Rural"))
    avg_glucose_level = st.sidebar.slider("avg_glucose_level ", 50, 250, 50)
    bmi = st.sidebar.slider("bmi ", 15, 40, 15)
    smoking_status = st.sidebar.radio(
        "smoking_status ", ("never smoked", "Unknown", "formerly smoked", "smokes"))

    info = {
        "id": 99999,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    init_info = pd.DataFrame(info, index=[0])
    return init_info


def label_encoding(df):
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['ever_married'] = le.fit_transform(df['ever_married'])
    df['work_type'] = le.fit_transform(df['work_type'])
    df['Residence_type'] = le.fit_transform(df['Residence_type'])
    df['smoking_status'] = le.fit_transform(df['smoking_status'])
    return df



data = pd.read_csv("./healthcare-dataset-stroke-data.csv")
data.bmi.replace(to_replace=np.nan, value=data.bmi.mean(), inplace=True)
userData = prepare_info()
data = data.append(userData, ignore_index=True)
# Select some rows using st.multiselect. This will break down when you have >1000 rows.
st.write('### Full Dataset', data)

st.write('### User Dataset', userData)
selected_indices = st.multiselect('Select rows:', data.index)
selected_rows = data.loc[selected_indices].iloc[:, 1:-1]
target = data.loc[selected_indices].iloc[:, -1]
st.write('### Selected Row X : ', len(selected_indices), selected_rows)
st.write('### Selected Row Y : ', len(selected_indices), target)


model_selected = st.radio("Select Model ", ("KNN", "RFR","DTREE"))
if model_selected == "RFR":
    model =  model_rfr
elif model_selected == "DTREE":
    model =  model_dtree

encode_df = label_encoding(selected_rows)
prediction = model.predict(encode_df)
st.write('### predict : ', len(selected_indices), prediction)
