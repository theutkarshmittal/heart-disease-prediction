import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# Load Model, Scaler and Columns
# ==========================================================

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.markdown("### AI Powered Clinical Risk Assessment")

st.write(
    "Fill in the patient's clinical information below and click **Predict** "
    "to estimate the likelihood of heart disease."
)

st.divider()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.header("📌 Project Information")

    st.write("**Model Used:** Support Vector Machine (SVM)")
    st.write("**Cross Validation Accuracy:** 87.6%")
    st.write("**Author:** Utkarsh Mittal")

    st.divider()

    st.info(
        "⚠ This application is for educational purposes only "
        "and should not be used as a medical diagnosis."
    )

# ==========================================================
# User Input
# ==========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "LVH", "ST"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        6.5,
        1.0,
        0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# ==========================================================
# Prediction
# ==========================================================

if st.button("🔍 Predict Heart Disease"):

    sex = 1 if sex == "Male" else 0
    exercise_angina = 1 if exercise_angina == "Yes" else 0

    input_data = {
        "Age": age,
        "Sex": sex,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak
    }

    # One Hot Encoding
    chest_columns = [
        "ChestPainType_ASY",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA"
    ]

    ecg_columns = [
        "RestingECG_LVH",
        "RestingECG_Normal",
        "RestingECG_ST"
    ]

    slope_columns = [
        "ST_Slope_Down",
        "ST_Slope_Flat",
        "ST_Slope_Up"
    ]

    for col in chest_columns:
        input_data[col] = 0

    input_data["ChestPainType_" + chest_pain] = 1

    for col in ecg_columns:
        input_data[col] = 0

    input_data["RestingECG_" + resting_ecg] = 1

    for col in slope_columns:
        input_data[col] = 0

    input_data["ST_Slope_" + st_slope] = 1

    input_df = pd.DataFrame([input_data])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale only numerical columns
    numeric_columns = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

    prediction = model.predict(input_df)[0]

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error("⚠ High Risk of Heart Disease")

        st.write(
            "The model predicts that the patient is likely to have heart disease."
        )

        st.warning(
            "Please consult a qualified healthcare professional for further evaluation."
        )

    else:

        st.success("✅ Low Risk of Heart Disease")

        st.write(
            "The model predicts that the patient is unlikely to have heart disease."
        )

        st.info(
            "Maintain a healthy lifestyle and continue regular health check-ups."
        )

st.divider()

st.caption(
    "Made with ❤️ using Python, Scikit-Learn and Streamlit"
)