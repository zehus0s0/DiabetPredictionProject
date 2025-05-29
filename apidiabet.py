import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        width: 100%;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .diabetic {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .non-diabetic {
        background-color: #e8f5e9;
        border: 2px solid #4CAF50;
    }
    h1 {
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("diabetes_scaler.pkl")
    medians = joblib.load("diabetes_medians.pkl")
    return model, scaler, medians

try:
    model, scaler, medians = load_model_components()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model components: {e}")
    model_loaded = False

# App title and description
st.title("Diabetes Prediction Tool")
st.markdown("Enter patient information below to predict diabetes risk")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120, help="Plasma glucose concentration after 2 hours in an oral glucose tolerance test")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=850, value=80, help="2-Hour serum insulin")
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, help="Body mass index")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01, help="Diabetes pedigree function (genetic influence)")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=33, help="Age in years")

# Predict button
predict_button = st.button("Predict Diabetes Risk")

if predict_button and model_loaded:
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create a DataFrame from the input data
    progress_bar.progress(20)
    
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    # Handle zero values
    progress_bar.progress(40)
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    df = pd.DataFrame([user_data])
    
    for col in zero_columns:
        df[col] = np.where(df[col] == 0, np.nan, df[col])
        df[col].fillna(medians[col], inplace=True)
    
    # Scale the data
    progress_bar.progress(60)
    df_scaled = scaler.transform(df)
    
    # Make prediction
    progress_bar.progress(80)
    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)[:, 1]
    
    # Complete the progress bar
    progress_bar.progress(100)
    
    # Show prediction result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    confidence = f"{probability[0]*100:.2f}%" if prediction[0] == 1 else f"{(1-probability[0])*100:.2f}%"
    
    # Display the prediction with conditional formatting
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="prediction-box diabetic">
            <h2>Prediction: Diabetic</h2>
            <h3>Confidence: {probability[0]*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-box non-diabetic">
            <h2>Prediction: Non-Diabetic</h2>
            <h3>Confidence: {(1-probability[0])*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance section
    st.subheader("Key Factors")
    
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        importances = model.feature_importances_
        
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display top 4 important features
        top_features = importance_df.head(4)
        
        # Create bar chart
        st.bar_chart(top_features.set_index('Feature'))
        
        # Health insights based on top features
        st.subheader("Health Insights")
        
        # Glucose level insights
        if 'Glucose' in top_features['Feature'].values:
            if glucose > 140:
                st.warning("⚠️ Glucose level is elevated. Consider consulting with a healthcare provider.")
            elif glucose < 70:
                st.warning("⚠️ Glucose level is below normal range. Consider consulting with a healthcare provider.")
            else:
                st.success("✅ Glucose level is within normal range.")
        
        # BMI insights
        if 'BMI' in top_features['Feature'].values:
            if bmi < 18.5:
                st.warning("⚠️ BMI indicates underweight status.")
            elif 18.5 <= bmi < 25:
                st.success("✅ BMI is within healthy range.")
            elif 25 <= bmi < 30:
                st.warning("⚠️ BMI indicates overweight status.")
            else:
                st.warning("⚠️ BMI indicates obesity. This is a risk factor for diabetes.")
        
        # Blood pressure insights
        if 'BloodPressure' in top_features['Feature'].values:
            if blood_pressure > 80:
                st.warning("⚠️ Diastolic blood pressure is elevated.")
            elif blood_pressure < 60:
                st.warning("⚠️ Diastolic blood pressure is low.")
            else:
                st.success("✅ Blood pressure is within normal range.")
    
    # Disclaimer
    st.markdown("""
    ---
    **Disclaimer**: This tool provides an estimate only and should not replace professional medical advice. 
    Please consult with a healthcare provider for accurate diagnosis and treatment.
    """)

# Show placeholder content when app first loads
if not predict_button and model_loaded:
    st.info("Enter patient information and click 'Predict Diabetes Risk' to get a prediction.")
    
    # Add information about diabetes
    st.markdown("""
    ### About Diabetes
    Diabetes is a chronic health condition that affects how your body turns food into energy. 
    
    Early detection and management can significantly reduce complications.
    """)
    
    # Add information about risk factors
    st.markdown("""
    ### Key Risk Factors
    - **High blood glucose**
    - **Family history of diabetes**
    - **BMI over 25**
    - **Age over 45**
    - **Physical inactivity**
    - **High blood pressure**
    """)