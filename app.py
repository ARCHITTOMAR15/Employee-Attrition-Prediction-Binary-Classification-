import streamlit as st
import pandas as pd
import joblib
import numpy as np

# LOAD EMPLOYEE ATTRITION MODEL
Employe_attrition_rf_model = joblib.load('employee_attrition_rf_model.pkl')

# Set page config
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

# Add light background color and style input fields
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f8ff;
        }
        
        /* Style input containers */
        .stSlider, .stNumberInput, .stSelectbox {
            background-color: #e8f4f8 !important;
            border: 2px solid #4a90e2 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* Style slider track */
        div[data-baseweb="slider"] {
            background-color: #e8f4f8 !important;
            border: 2px solid #4a90e2 !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        /* Style input boxes */
        input[type="number"], select, input[type="text"] {
            background-color: #e8f4f8 !important;
            border: 2px solid #4a90e2 !important;
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Add sidebar with app details
with st.sidebar:
    st.header("Information")
    st.write("""
    **Employee Attrition Prediction App**
    
    This machine learning application predicts whether an employee is likely to leave the company based on various factors.
    
    **How it works:**
    - Input employee details including satisfaction level, tenure, work accidents, promotions, and more
    - The app uses a Random Forest Machine Learning model trained on historical employee data
    - Provides instant predictions with high accuracy
    
    **Features:**
    - Easy-to-use interface
    - Real-time predictions
    - Considers multiple employee factors
    
    **Use Cases:**
    - HR departments for employee retention strategies
    - Management planning
    - Identifying at-risk employees
    """)

# Title
st.title("Employee Attrition Prediction")

# Display local image resized to a smaller height so it fits the title width


# Center the subheader
st.markdown("<h2 style='text-align: center;'>Enter Employee Details</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2) # enforce equal column widths

with col1:
    satisfaction_level = st.slider("Satisfaction Level (1,10)", 1.0, 10.0, 0.5)
    time_spend_company = st.number_input("Time Spent at Company (years)", min_value=0, max_value=40, value=5)
    work_accident = st.selectbox("Work Accident", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    

with col2:
    department = st.selectbox("Department", ["Sales", "RandD", "HR", "Management"])
    salary_level = st.selectbox("Salary Level", ["Low", "Medium", "High"])
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("Predict Attrition", key="predict_btn"):
    # Prepare binary columns for department
    department_RandD = 1 if department == "RandD" else 0
    department_hr = 1 if department == "HR" else 0
    department_management = 1 if department == "Management" else 0
    
    # Prepare binary columns for salary
    salary_high = 1 if salary_level == "High" else 0
    salary_low = 1 if salary_level == "Low" else 0
    salary_medium = 1 if salary_level == "Medium" else 0
    
    # Prepare input data in the correct order
    input_data = np.array([[
        satisfaction_level,
        time_spend_company,
        work_accident,
        promotion_last_5years,
        department_RandD,
        department_hr,
        department_management,
        salary_high,
        salary_low,
        salary_medium
    ]])
    
    # Make prediction
    prediction = Employe_attrition_rf_model.predict(input_data)[0]
    
    # Display result
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("⚠️ **Employee is likely to leave the company**")
    else:
        st.success("✅ **Employee is likely to stay in the company**")
