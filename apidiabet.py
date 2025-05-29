# import numpy as np
# import pandas as pd
# import joblib
# import streamlit as st

# # Load model
# diabet_loaded_rf_model = joblib.load("diabet_rf_model.pkl")

# # Load the column names used during training (expecting 8 features)
# with open("diabet_training_columns.pkl", "rb") as f:
#     diabet_training_columns = joblib.load(f)

# # Define the input form for diabetes prediction
# st.title("Diabetes Prediction")

# # Collect input data based on your diabetes dataset columns
# pregnancies = st.number_input("Pregnancies", min_value=0)
# glucose = st.number_input("Glucose Level", min_value=0)
# blood_pressure = st.number_input("Blood Pressure", min_value=0)
# skin_thickness = st.number_input("Skin Thickness", min_value=0)
# insulin = st.number_input("Insulin", min_value=0)
# bmi = st.number_input("BMI", min_value=0.0)
# diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
# age = st.number_input("Age", min_value=0)

# # Create a DataFrame from the input data
# diabet_data = pd.DataFrame({
#     "Pregnancies": [pregnancies],
#     "Glucose": [glucose],
#     "BloodPressure": [blood_pressure],
#     "SkinThickness": [skin_thickness],
#     "Insulin": [insulin],
#     "BMI": [bmi],
#     "DiabetesPedigreeFunction": [diabetes_pedigree],
#     "Age": [age]
# })

# # # Yeni özellikler oluşturma
# # diabet_data["NEW_AGE_CAT"] = np.where((diabet_data["Age"] >= 50), "senior", "mature")
# # diabet_data["NEW_BMI"] = pd.cut(x=diabet_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
# # diabet_data["NEW_GLUCOSE"] = pd.cut(x=diabet_data["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
# # diabet_data["NEW_INSULIN_SCORE"] = diabet_data.apply(lambda row: "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal", axis=1)
# # diabet_data["NEW_GLUCOSE*Insulin"] = diabet_data["Glucose"] * diabet_data["Insulin"]
# # diabet_data["NEW_GLUCOSE*Pregnancies"] = diabet_data["Glucose"] * diabet_data["Pregnancies"]

# # diabet_data = pd.get_dummies(diabet_data)

# # Align with training columns
# diabet_data = diabet_data.reindex(columns=diabet_training_columns, fill_value=0)

# # Ensure the input data aligns with the model's expected features
# diabet_data = diabet_data[diabet_training_columns]  # Use only the expected columns,

# # Print columns for debugging
# st.write("Data Columns:", diabet_data.columns.tolist())
# st.write("Training Columns:", diabet_training_columns)

# # Predict using the Random Forest model
# if st.button("Predict Diabetes"):
#     prediction_rf = diabet_loaded_rf_model.predict(diabet_data)

#     st.write(f"Random Forest Prediction: {'Diabetic' if prediction_rf[0] else 'Not Diabetic'}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import streamlit as st
warnings.simplefilter(action="ignore")

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Function to load the model and necessary data
@st.cache_resource
def load_model():
    model = joblib.load("diabet_rf_model.pkl")
    try:
        with open("diabet_training_columns.pkl", "rb") as f:
            training_columns = joblib.load(f)
    except:
        # If columns file doesn't exist, use these default columns
        training_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    return model, training_columns

# Normal ranges for each parameter (based on medical guidelines)
NORMAL_RANGES = {
    'Pregnancies': (0, 17),  # Max from dataset
    'Glucose': (70, 99),     # Normal fasting glucose
    'BloodPressure': (90, 120),  # Normal systolic
    'SkinThickness': (10, 30),   # Normal triceps skinfold in mm
    'Insulin': (16, 166),    # Normal fasting insulin μU/mL
    'BMI': (18.5, 24.9),     # Normal BMI range
    'DiabetesPedigreeFunction': (0.1, 1.0),  # Typical range
    'Age': (21, 90)          # Adult age range
}

# Function for feature engineering
def process_features(input_data):
    # Create a copy to avoid modifying the original data
    data = input_data.copy()
    
    # AGE categories
    data.loc[(data["Age"] >= 21) & (data["Age"] < 50), "NEW_AGE_CAT"] = "mature"
    data.loc[(data["Age"] >= 50), "NEW_AGE_CAT"] = "senior"
    
    # BMI categories
    data['NEW_BMI'] = pd.cut(x=data['BMI'], 
                            bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=["Underweight", "Healthy", "Overweight", "Obese"])
    
    # Glucose categories
    data["NEW_GLUCOSE"] = pd.cut(x=data["Glucose"], 
                                bins=[0, 140, 200, 300], 
                                labels=["Normal", "Prediabetes", "Diabetes"])
    
    # Age and BMI combined categories
    data.loc[(data["BMI"] < 18.5) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
    data.loc[(data["BMI"] < 18.5) & (data["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
    data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
    data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & (data["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
    data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
    data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & (data["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
    data.loc[(data["BMI"] > 30) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
    data.loc[(data["BMI"] > 30) & (data["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
    
    # Age and Glucose combined categories
    data.loc[(data["Glucose"] < 70) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
    data.loc[(data["Glucose"] < 70) & (data["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
    data.loc[((data["Glucose"] >= 70) & (data["Glucose"] < 100)) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
    data.loc[((data["Glucose"] >= 70) & (data["Glucose"] < 100)) & (data["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
    data.loc[((data["Glucose"] >= 100) & (data["Glucose"] <= 125)) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
    data.loc[((data["Glucose"] >= 100) & (data["Glucose"] <= 125)) & (data["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
    data.loc[(data["Glucose"] > 125) & ((data["Age"] >= 21) & (data["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
    data.loc[(data["Glucose"] > 125) & (data["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"
    
    # Insulin categories
    data["NEW_INSULIN_SCORE"] = data.apply(lambda x: "Normal" if 16 <= x["Insulin"] <= 166 else "Abnormal", axis=1)
    
    # Interaction terms
    data["NEW_GLUCOSE*Insulin"] = data["Glucose"] * data["Insulin"]
    data["NEW_GLUCOSE*Pregnancies"] = data["Glucose"] * (1 + data["Pregnancies"])
    
    return data

# Function to encode categorical variables
def encode_features(data):
    # Get categorical columns
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    
    # Label encoding for binary categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        if data[col].nunique() == 2:
            data[col] = le.fit_transform(data[col])
        else:
            # One-hot encoding for non-binary categorical columns
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
    
    return data

# Function to scale numerical features
def scale_features(data, num_cols):
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

# Prepare a single input for prediction
def prepare_input(input_dict, training_columns):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Apply feature engineering
    processed_df = process_features(input_df)
    
    # Encode categorical features
    encoded_df = encode_features(processed_df)
    
    # Get numerical columns
    num_cols = [col for col in encoded_df.columns if encoded_df[col].dtype != 'object' and col in input_dict.keys()]
    
    # Scale numerical features
    scaled_df = scale_features(encoded_df, num_cols)
    
    # Ensure all training columns are present
    for col in training_columns:
        if col not in scaled_df.columns:
            scaled_df[col] = 0
    
    # Select only the columns used during training
    final_df = scaled_df[training_columns]
    
    return final_df

# Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.write("""
    ### Enter Your Health Information
    This app predicts whether a person has diabetes based on medical data.
    The input sliders show normal ranges in green.
    """)
    
    # Load model and training columns
    model, training_columns = load_model()
    
    # Create sidebar with input parameters
    st.sidebar.header('User Input Parameters')
    
    input_data = {}
    
    # Create sliders for each input parameter with normal ranges highlighted
    for feature, (min_normal, max_normal) in NORMAL_RANGES.items():
        # Determine min and max for slider
        min_val = 0
        max_val = min_normal * 3 if min_normal > 0 else max_normal * 2
        
        # Calculate step size
        step = (max_val - min_val) / 100
        step = max(step, 0.1)  # Ensure step is at least 0.1
        
        # Create the slider
        value = st.sidebar.slider(
            f"{feature} (Normal: {min_normal}-{max_normal})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_normal + max_normal) / 2),  # Default to middle of normal range
            step=step
        )
        
        # Add visual indication if value is outside normal range
        if value < min_normal or value > max_normal:
            st.sidebar.warning(f"{feature} is outside normal range!")
        
        input_data[feature] = value
    
    # Create prediction button
    if st.sidebar.button('Predict'):
        # Prepare input for prediction
        processed_input = prepare_input(input_data, training_columns)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        
        # Display prediction
        st.subheader('Prediction Results')
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Diagnosis:")
            if prediction == 1:
                st.error("**Positive for Diabetes**")
            else:
                st.success("**Negative for Diabetes**")
        
        with col2:
            st.write("#### Probability:")
            st.write(f"Diabetes Risk: {prediction_proba[1]:.2%}")
            st.write(f"No Diabetes Risk: {prediction_proba[0]:.2%}")
        
        # Visualization of the prediction
        st.subheader('Risk Visualization')
        
        # Create a gauge chart for diabetes risk
        fig, ax = plt.subplots(figsize=(8, 3))
        
        # Gradient colors for risk levels
        colors = ['green', 'yellowgreen', 'orange', 'red']
        
        # Create gradient background
        for i in range(100):
            ax.axvspan(i, i+1, alpha=0.5, color=colors[min(3, i // 25)])
        
        # Add pointer for the risk probability
        risk = prediction_proba[1] * 100
        ax.scatter(risk, 0.5, s=300, color='black', zorder=5, marker='^')
        
        # Annotate risk levels
        ax.text(12.5, 0.8, 'Low Risk', fontsize=10, ha='center')
        ax.text(37.5, 0.8, 'Moderate Risk', fontsize=10, ha='center')
        ax.text(62.5, 0.8, 'High Risk', fontsize=10, ha='center')
        ax.text(87.5, 0.8, 'Very High Risk', fontsize=10, ha='center')
        
        # Add risk percentage
        ax.text(risk, 0.2, f"{risk:.1f}%", fontsize=12, ha='center', fontweight='bold')
        
        # Set up the plot
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_yticks([])
        ax.set_title('Diabetes Risk Level')
        
        # Display the plot
        st.pyplot(fig)
        
        # Insights based on input values
        st.subheader('Health Insights')
        
        # Check which values are outside normal range
        outside_normal = []
        for feature, value in input_data.items():
            min_normal, max_normal = NORMAL_RANGES[feature]
            if value < min_normal or value > max_normal:
                outside_normal.append((feature, value, min_normal, max_normal))
        
        if outside_normal:
            st.warning("The following values are outside normal range:")
            for feature, value, min_normal, max_normal in outside_normal:
                st.write(f"- **{feature}**: {value:.1f} (Normal range: {min_normal}-{max_normal})")
                
                # Provide specific advice for each feature
                if feature == "Glucose" and value > max_normal:
                    st.write("   ℹ️ High glucose levels may indicate prediabetes or diabetes. Consider consulting with a healthcare provider.")
                elif feature == "BMI" and value > max_normal:
                    st.write("   ℹ️ A BMI above the normal range may contribute to diabetes risk. Consider weight management strategies.")
                elif feature == "BloodPressure" and value > max_normal:
                    st.write("   ℹ️ Elevated blood pressure may indicate hypertension. Consider lifestyle modifications.")
                elif feature == "Insulin" and (value < min_normal or value > max_normal):
                    st.write("   ℹ️ Abnormal insulin levels may indicate insulin resistance or other metabolic issues.")
        else:
            st.success("All your values are within normal ranges. Keep up the good health practices!")
        
        # Feature importance visualization
        st.subheader('What Factors Influenced This Prediction?')
        
        # Get feature importances
        feature_importances = model.feature_importances_
        feature_names = training_columns
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_importances = feature_importances[sorted_indices]
        
        # Show top 5 features
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        bars = ax.barh(sorted_names[:5], sorted_importances[:5], color=colors[:5])
        
        # Add importance percentage to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2%}", 
                   va='center', fontsize=10)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 5 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Disclaimer
        st.info("""
        **Disclaimer**: This prediction is based on a machine learning model and should not be used for medical diagnosis. 
        Always consult with a healthcare professional for proper medical advice and diagnosis.
        """)

if __name__ == "__main__":
    main()