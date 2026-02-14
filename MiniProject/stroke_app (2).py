import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and features
model = pickle.load(open('stroke_lr_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("🩺 Stroke Prediction App")

# --- User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Enter age")
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=None, placeholder="Enter glucose level")
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=None, placeholder="Enter BMI")
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])


# --- ERROR HANDLING ---
def validate_inputs():
    if age is None or age == 0:
        st.error("❌ Age is required!")
        return False

    if avg_glucose_level is None or avg_glucose_level == 0:
        st.error("❌ Average Glucose Level is required!")
        return False

    if bmi is None or bmi == 0:
        st.error("❌ BMI is required!")
        return False

    return True


# --- Build Input Dict ---
input_dict = {
    "gender": 1 if gender == "Male" else (0 if gender == "Female" else 2),
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": 1 if ever_married == "Yes" else 0,
    "Residence_type": 1 if residence_type == "Urban" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,

    # One-hot encoding
    "work_type_Govt_job": 1 if work_type == "Govt_job" else 0,
    "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
    "work_type_Private": 1 if work_type == "Private" else 0,
    "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
    "work_type_children": 1 if work_type == "children" else 0,
    "smoking_status_Unknown": 1 if smoking_status == "Unknown" else 0,
    "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
    "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
    "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
}

# --- Create DF ---
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=features, fill_value=0)


# --- Prediction ---
if st.button("Predict"):
    if validate_inputs():   # ✅ Check before prediction
        features_scaled = scaler.transform(input_df)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            st.error("⚠️ High chance of Stroke")
        else:
            st.success("✅ Low chance of Stroke")
