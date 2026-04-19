import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cleveland Heart Disease Predictor", page_icon="❤️", layout="wide")

# App Styles
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
h1 {
    color: #ff4b4b;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pretrained_model():
    # Load previously trained ML model
    model = joblib.load('heart_model.pkl')
    # Load metadata associated with this model (accuracy, confusion matrix, col names, test_df)
    accuracy, conf_matrix, feature_names, df = joblib.load('model_metadata.pkl')
    return model, accuracy, conf_matrix, feature_names, df

def main():
    st.title("❤️ Heart Disease Prediction App")
    st.write("Using the renowned UCI Cleveland Dataset and a pre-trained offline Machine Learning model to predict heart disease risk.")
    
    with st.spinner("Loading pre-trained model and weights..."):
        try:
            model, accuracy, conf_matrix, feature_names, df = load_pretrained_model()
        except FileNotFoundError:
            st.error("Model files not found! Please ensure you have run `train.py` first to generate `heart_model.pkl` and `model_metadata.pkl`.")
            st.stop()
            
    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", ["Heart Risk Assessment", "Model Evaluation & Data"])
    
    if menu == "Heart Risk Assessment":
        st.header("Patient Data Input")
        st.write("Please provide the following clinical details to assess the risk of heart disease.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
            cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], 
                              help="1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic")
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[1, 0], format_func=lambda x: "True" if x==1 else "False")
            restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2],
                                   help="0: Normal, 1: ST-T wave abnormality, 2: Probable/definite left ventricular hypertrophy")
            thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
            exang = st.selectbox("Exercise Induced Angina (exang)", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
            
        with col3:
            oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[1, 2, 3],
                                 help="1: Upsloping, 2: Flat, 3: Downsloping")
            ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (ca)", options=[0, 1, 2, 3])
            thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7],
                                help="3 = Normal, 6 = Fixed defect, 7 = Reversable defect")
            
        predict_btn = st.button("Predict Risk 🚀", use_container_width=True)
        
        if predict_btn:
            input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                      columns=feature_names)
            
            with st.spinner("Analyzing..."):
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
            st.markdown("---")
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Heart Disease Detected!")
                st.write(f"The model predicts the presence of heart disease with **{proba[1]*100:.1f}% confidence**.")
                st.warning("Please consult with a healthcare professional for a comprehensive diagnosis.")
            else:
                st.success(f"### ✅ Low Risk of Heart Disease")
                st.write(f"The model predicts the absence of heart disease with **{proba[0]*100:.1f}% confidence**.")
                st.info("Maintaining a healthy lifestyle is still recommended.")

    elif menu == "Model Evaluation & Data":
        st.header("📊 Model Evaluation & Accuracy")
        st.write("We used a previously trained offline Machine Learning model (Gradient Boosting Classifier).")
        
        st.markdown(f"### Validation Accuracy: **{accuracy*100:.2f}%**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd', cbar=False,
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'])
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
            
        with col2:
            st.subheader("Dataset Overview (Sample)")
            st.dataframe(df.head(10))
            st.write(f"Total entries: {len(df)}")
            st.write(f"Features trained on: {len(feature_names)}")

if __name__ == "__main__":
    main()
