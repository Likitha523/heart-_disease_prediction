import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="❤️")

@st.cache_resource
def load_models():
    try:
        import os
        model_type = 'pkl'
        if os.path.exists('model_type.txt'):
            with open('model_type.txt', 'r') as f:
                model_type = f.read().strip()
                
        if model_type == 'keras':
            from tensorflow.keras.models import load_model
            model = load_model('heart_model.keras')
        else:
            model = joblib.load('heart_model.pkl')
            
        preprocessor = joblib.load('preprocessor.pkl')
        return model, preprocessor, model_type
    except Exception as e:
        st.error(f"Model files not found or error loading: {e}. Please run train_model.py first.")
        st.stop()

model, preprocessor, model_type = load_models()
scaler = preprocessor['scaler']
feature_cols = preprocessor['features']

st.title("❤️ AI Heart Disease Prediction System")
st.markdown("Predict the risk of Cardiovascular Disease using Machine Learning.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=130, max_value=220, value=170)
    weight = st.number_input("Weight (kg)", min_value=40, max_value=200, value=70)

with col2:
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=80, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=50, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])
    gluc = st.selectbox("Glucose", options=[1, 2, 3], format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])

with col3:
    smoke = st.selectbox("Smoker?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    alco = st.selectbox("Alcohol Intake?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    active = st.selectbox("Physical Activity?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")

if st.button("Predict Risk", type="primary"):
    
    # Feature Engineering
    age_years = age
    bmi = weight / ((height / 100) ** 2)
    
    pulse_pressure = ap_hi - ap_lo
    map_val = ap_lo + (pulse_pressure / 3)
    
    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 0 # Normal
        elif 120 <= ap_hi <= 129 and ap_lo < 80:
            return 1 # Elevated
        elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
            return 2 # Stage 1 HBP
        elif ap_hi >= 140 or ap_lo >= 90:
            return 3 # Stage 2 HBP
        else:
            return 0
    bp_risk = categorize_bp(ap_hi, ap_lo)
    
    def categorize_age(age_val):
        if age_val < 40: return 0
        elif 40 <= age_val < 50: return 1
        elif 50 <= age_val < 55: return 2
        elif 55 <= age_val < 60: return 3
        else: return 4
    age_group = categorize_age(age_years)
    gender_val = 2 if gender == "Male" else 1
    
    input_data = {
        'gender': [gender_val],
        'height': [height],
        'weight': [weight],
        'ap_hi': [ap_hi],
        'ap_lo': [ap_lo],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active],
        'age_years': [age_years],
        'bmi': [bmi],
        'pulse_pressure': [pulse_pressure],
        'map': [map_val],
        'bp_risk': [bp_risk],
        'age_group': [age_group]
    }
    
    # Ensure column order matches training
    input_df = pd.DataFrame(input_data)
    input_df = input_df[feature_cols]
    
    # Check for Polynomial Features
    poly = preprocessor.get('poly', None)
    if poly is not None:
        input_transformed = poly.transform(input_df)
    else:
        input_transformed = input_df
        
    # Scale input
    input_scaled = scaler.transform(input_transformed)
    
    # Predict
    if model_type == 'keras':
        prob = float(model.predict(input_scaled, verbose=0)[0][0])
    else:
        prob = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if prob >= 0.5 else 0
    confidence = max(prob, 1-prob) * 100
    
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 **HIGH RISK** of Cardiovascular Disease")
        else:
            st.success(f"✅ **LOW RISK** of Cardiovascular Disease")
            
        st.metric("Probability Score", f"{prob*100:.2f}%")
        st.metric("Confidence Level", f"{confidence:.2f}%")
        
    with res_col2:
        st.subheader("Explainability (SHAP)")
        st.markdown("Feature impact on this specific prediction:")
        
        try:
            if model_type == 'keras':
                st.info("SHAP feature explainability is currently optimized for Tree-based models. Detailed SHAP plot is hidden for the Deep Learning Neural Network.")
            else:
                # We attempt SHAP for Tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                
                # Matplotlib for SHAP
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Since SHAP behavior varies by library version, we plot our own simplified horizontal bar chart 
                # showing feature contributions
                
                # For tree explainers, shap_values might be a list (multiclass) or array
                if isinstance(shap_values, list):
                    sv = shap_values[1][0] # take class 1
                elif len(shap_values.shape) == 3: # Some lightgbm/xgboost
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values[0]
                    
                # Determine correct feature names depending on Polynomial expansion
                actual_feature_names = feature_cols
                if 'poly' in locals() and poly is not None:
                    actual_feature_names = poly.get_feature_names_out(feature_cols)
                    
                contribution_df = pd.DataFrame({
                    'Feature': actual_feature_names,
                    'Contribution': sv
                })
                contribution_df['Abs_Contrib'] = np.abs(contribution_df['Contribution'])
                
                # Filter down to top 10 most impactful features for clean readability
                contribution_df = contribution_df.sort_values(by='Abs_Contrib', ascending=False).head(10)
                contribution_df = contribution_df.sort_values(by='Abs_Contrib', ascending=True)
                
                colors = ['red' if x > 0 else 'green' for x in contribution_df['Contribution']]
                
                ax.barh(contribution_df['Feature'], contribution_df['Contribution'], color=colors)
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Top 10 Feature Impacts for Current Prediction')
                
                st.pyplot(fig)
                st.caption("🔴 Red shifts prediction higher, 🟢 Green shifts prediction lower")
        except Exception as e:
            # Fallback to general feature importance plotting if SHAP fails
            try:
                importances = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                
                if importances is not None:
                    actual_feature_names = feature_cols
                    if 'poly' in locals() and poly is not None:
                        actual_feature_names = poly.get_feature_names_out(feature_cols)
                        
                    feat_df = pd.DataFrame({'Feature': actual_feature_names, 'Importance': importances})
                    feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10).sort_values(by='Importance', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
                    ax.set_xlabel('Global Feature Importance')
                    ax.set_title('Global Risk Predictors (Fallback)')
                    st.pyplot(fig)
                    st.caption("These constitute the general top risk factors for your generated profile (SHAP local insights disabled).")
                else:
                    st.warning("Visualized impact is unsupported for this underlying framework combination.")
            except:
                st.info("Feature impact plots are currently unavailable for this specific model architecture.")
            
    st.markdown("---")
    st.caption("Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.")
