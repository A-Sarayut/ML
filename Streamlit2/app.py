from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')


model_dtree = pickle.load(open('treemodel.sav', 'rb'))
model_knn = pickle.load(open('knn_model.sav', 'rb'))
model_rfr = pickle.load(open('rfr_model.sav', 'rb'))
st.sidebar.header('prpare data')
st.title('Stroke Classification Web App')
st.write('>  Develope By : Sarayut Aree , Aphisit Thupsaeng')


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

#userData = prepare_info()
##data = data.append(userData, ignore_index=True)
# st.write('### Input Data', userData)


data = pd.read_csv("./healthcare-dataset-stroke-data.csv")
data.bmi.replace(to_replace=np.nan, value=data.bmi.mean(), inplace=True)

st.sidebar.write('### Full Dataset', data)
st.sidebar.subheader("Choose Classifier")
model_selected = st.sidebar.selectbox(
    "Classifier", ("K-Nearest Neighbors (KNN)", "Random Forest (RF)", "Decision Tree"))
if model_selected == "Random Forest (RF)":
    model = model_rfr
elif model_selected == "Decision Tree":
    model = model_dtree
elif model_selected == "K-Nearest Neighbors (KNN)":
    model = model_knn
selected_indices = st.sidebar.multiselect('Select rows:', data.index)

selection_row = data.loc[selected_indices]
input = data.loc[selected_indices].iloc[:, 1:-1]
target = data.loc[selected_indices].iloc[:, -1]
st.write('### Selecting Row : ', len(selected_indices), selection_row,)

# MODEL SECTION
if selected_indices:
    encode_df = label_encoding(input)
    prediction = model.predict(encode_df)

    st.subheader("Model Evaluation")
    st.write("Accuracy : ", accuracy_score(target, prediction).round(2))
    st.write("Precision : ", precision_score(target, prediction).round(2))
    st.write("Recall : ", recall_score(target, prediction).round(2))

    st.write('### prediction : ', len(selected_indices), prediction)
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(model, input, target, display_labels=["yes", "no"])
    st.pyplot()
