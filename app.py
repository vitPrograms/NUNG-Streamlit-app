import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Кешування для оптимізації
@st.cache_resource
def load_model_and_resources():
    model = load_model("obesity_model.h5")
    preprocessor = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, preprocessor, columns

model, preprocessor, columns = load_model_and_resources()

# Інтерфейс користувача
st.title("🏋️ Obesity Level Prediction")
st.write("Enter your details to predict obesity level.")

# Розбиття на стовпці для основних параметрів
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Characteristics")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=25, step=1, key="age")
    height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.7, step=0.01, key="height")
    weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, step=1, key="weight")

with col2:
    st.subheader("Health History")
    family_history = st.selectbox("Family history with overweight (yes/no)", ["yes", "no"], key="family_history")
    smoke = st.selectbox("Do you smoke (yes/no)", ["yes", "no"], key="smoke")

# Додаткові ознаки у розгортаючому вікні
with st.expander("Additional Parameters (optional)", expanded=False):
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Dietary Habits")
        favc = st.selectbox("Frequently consume high-calorie food (yes/no)", ["yes", "no"], key="favc")
        fcvc = st.slider("Frequency of vegetable consumption (1-3)", min_value=1, max_value=3, value=2, key="fcvc")
        ncp = st.slider("Number of main meals per day (1-3)", min_value=1, max_value=3, value=2, key="ncp")
        caec = st.selectbox("Food between meals (Never, Sometimes, Frequently, Always)", ["Never", "Sometimes", "Frequently", "Always"], key="caec")
        ch2o = st.slider("Daily water intake (1-3)", min_value=1, max_value=3, value=2, key="ch2o")
        scc = st.selectbox("Monitor calorie intake (yes/no)", ["yes", "no"], key="scc")

    with col4:
        st.subheader("Activity & Lifestyle")
        faf = st.slider("Physical activity frequency (0-3)", min_value=0, max_value=3, value=1, key="faf")
        tue = st.slider("Time using technology (0-3)", min_value=0, max_value=3, value=1, key="tue")
        calc = st.selectbox("Alcohol consumption (Never, Sometimes, Frequently, Always)", ["Never", "Sometimes", "Frequently", "Always"], key="calc")
        mtrans = st.selectbox("Main transportation (Automobile, Bike, Motorbike, Public Transportation, Walking)", ["Automobile", "Bike", "Motorbike", "Public Transportation", "Walking"], key="mtrans")

# Попередня обробка
def preprocess_input(gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans, preprocessor, columns):
    df = pd.DataFrame({
        "Gender": [gender],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": [family_history],
        "FAVC": [favc],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": [caec],
        "SMOKE": [smoke],
        "CH2O": [ch2o],
        "SCC": [scc],
        "FAF": [faf],
        "TUE": [tue],
        "CALC": [calc],
        "MTRANS": [mtrans]
    })

    # Перетворення через ColumnTransformer
    X_processed = preprocessor.transform(df)

    # Перевірка розміру
    if X_processed.shape[1] != len(columns):
        st.error(f"Помилка: Кількість стовпців {X_processed.shape[1]} не збігається з очікуваною {len(columns)}. Оновіть columns.pkl.")
        return None

    X_processed_df = pd.DataFrame(X_processed, columns=columns)
    return X_processed_df

# Передбачення
if st.button("Predict Obesity Level"):
    input_data = preprocess_input(gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans, preprocessor, columns)
    if input_data is not None:
        pred_proba = model.predict(input_data)
        pred_class = np.argmax(pred_proba, axis=1)
        class_names = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II', 'Obesity Type I', 'Obesity Type II', 'Obesity Type III']
        predicted_label = class_names[pred_class[0]]
        st.success(f"Predicted Obesity Level: **{predicted_label}**")
        st.write("Probabilities:", {class_names[i]: f"{prob:.2%}" for i, prob in enumerate(pred_proba[0])})

# Удосконалення інтерфейсу
col1, col2 = st.columns(2)
with col1:
    st.header("Input Parameters")
    st.write("Adjust the sliders and dropdowns to input your data.")
with col2:
    st.header("Result")
    if 'predicted_label' in locals():
        st.write(f"**Your obesity level is: {predicted_label}**")

# Додатковий контент
st.image("https://via.placeholder.com/150", caption="Obesity Awareness", use_column_width=True)
st.write("Developed by [Your Name] for AI Innovations course.")