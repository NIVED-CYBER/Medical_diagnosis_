import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .disease-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: white;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive-result {
        background-color: #ff6b6b;
        color: white;
    }
    .negative-result {
        background-color: #51cf66;
        color: white;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    """Load all disease prediction models"""
    models = {}
    model_files = {
        'diabetes': 'Models/diabetes_model.sav',
        'heart_disease': 'Models/heart_disease_model.sav',
        'parkinsons': 'Models/parkinsons_model.sav',
        'lung_cancer': 'Models/lungs_disease_model.sav',
        'thyroid': 'Models/Thyroid_model.sav'
    }
    for disease, filepath in model_files.items():
        try:
            models[disease] = pickle.load(open(filepath, 'rb'))
        except FileNotFoundError:
            st.error(f"Model file not found: {filepath}")
            models[disease] = None
        except Exception as e:
            st.error(f"Error loading {disease} model: {str(e)}")
            models[disease] = None
    return models

# Disease information database
DISEASE_INFO = {
    'Diabetes Prediction': {
        'description': 'Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).',
        'risk_factors': ['Family history', 'Obesity', 'Sedentary lifestyle', 'Age over 45'],
        'prevention': ['Regular exercise', 'Healthy diet', 'Weight management', 'Regular check-ups'],
        'icon': '🩺'
    },
    'Heart Disease Prediction': {
        'description': 'Heart disease refers to several types of heart conditions that affect heart function.',
        'risk_factors': ['High blood pressure', 'High cholesterol', 'Smoking', 'Diabetes'],
        'prevention': ['Heart-healthy diet', 'Regular exercise', 'No smoking', 'Stress management'],
        'icon': '❤️'
    },
    'Parkinsons Prediction': {
        'description': 'Parkinson\'s disease is a progressive nervous system disorder that affects movement.',
        'risk_factors': ['Age', 'Family history', 'Gender (men more likely)', 'Environmental toxins'],
        'prevention': ['Regular exercise', 'Balanced diet', 'Avoid toxins', 'Stay mentally active'],
        'icon': '🧠'
    },
    'Lung Cancer Prediction': {
        'description': 'Lung cancer is a type of cancer that begins in the lungs and can spread to other parts of the body.',
        'risk_factors': ['Smoking', 'Secondhand smoke', 'Radon exposure', 'Family history'],
        'prevention': ['No smoking', 'Avoid secondhand smoke', 'Test home for radon', 'Healthy diet'],
        'icon': '🫁'
    },
    'Hypo-Thyroid Prediction': {
        'description': 'Hypothyroidism is a condition where the thyroid gland doesn\'t produce enough thyroid hormone.',
        'risk_factors': ['Autoimmune disease', 'Family history', 'Age over 60', 'Gender (women more likely)'],
        'prevention': ['Regular screening', 'Balanced diet with iodine', 'Stress management', 'Avoid excessive soy'],
        'icon': '🦋'
    }
}

def create_input_field(label, help_text, key, input_type="number", min_val=0, max_val=None, step=1):
    """Create standardized input fields with validation and type consistency"""
    col1, col2 = st.columns([3, 1])
    with col1:
        if input_type == "number":
            # Ensure type consistency between step and max_val
            if isinstance(step, float):
                min_val = float(min_val)
                if max_val is not None:
                    max_val = float(max_val)
            value = st.number_input(
                label,
                min_value=min_val,
                max_value=max_val,
                step=step,
                key=key,
                help=help_text
            )
        else:
            value = st.text_input(label, key=key, help=help_text)
    with col2:
        if input_type == "number":
            st.info(f"Range: {min_val}-{max_val if max_val else '∞'}")
    return value

def display_disease_info(disease_name):
    """Display disease information card"""
    info = DISEASE_INFO.get(disease_name, {})
    st.markdown(f"""
    <div class="disease-card">
        <h3>{info.get('icon', '🏥')} {disease_name}</h3>
        <p>{info.get('description', 'No description available.')}</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚨 Risk Factors")
        for factor in info.get('risk_factors', []):
            st.write(f"• {factor}")
    with col2:
        st.subheader("🛡️ Prevention Tips")
        for tip in info.get('prevention', []):
            st.write(f"• {tip}")

def display_prediction_result(prediction, disease_name):
    """Display prediction results with styling"""
    is_positive = prediction[0] == 1
    if is_positive:
        result_class = "positive-result"
        result_text = f"⚠️ High Risk: The model predicts a higher likelihood of {disease_name.lower()}"
        recommendation = "Please consult with a healthcare professional for proper diagnosis and treatment."
    else:
        result_class = "negative-result"
        result_text = f"✅ Low Risk: The model predicts a lower likelihood of {disease_name.lower()}"
        recommendation = "Continue maintaining a healthy lifestyle and regular check-ups."

    st.markdown(f"""
    <div class="prediction-result {result_class}">
        {result_text}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <strong>Recommendation:</strong> {recommendation}
        <br><br>
        <strong>Disclaimer:</strong> This prediction is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

def save_prediction_history(disease, inputs, prediction, timestamp):
    """Save prediction to session state history"""
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    history_entry = {
        'disease': disease,
        'inputs': inputs,
        'prediction': int(prediction[0]),
        'timestamp': timestamp
    }
    st.session_state.prediction_history.append(history_entry)

def display_prediction_history():
    """Display prediction history"""
    if 'prediction_history' not in st.session_state or not st.session_state.prediction_history:
        st.info("No previous predictions found.")
        return

    st.subheader("📊 Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['result'] = history_df['prediction'].map({1: 'Positive', 0: 'Negative'})

    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(history_df))
    with col2:
        positive_count = (history_df['prediction'] == 1).sum()
        st.metric("Positive Results", positive_count)
    with col3:
        negative_count = (history_df['prediction'] == 0).sum()
        st.metric("Negative Results", negative_count)

    # Display history table
    display_df = history_df[['timestamp', 'disease', 'result']].copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(display_df, use_container_width=True)

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Main App
def main():
    # App header
    st.markdown('<h1 class="main-header">🏥 AI Disease Prediction System</h1>', unsafe_allow_html=True)

    # Load models
    models = load_models()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/health-checkup.png", width=80)
        st.title("Navigation")
        selected = st.selectbox(
            'Select Disease Prediction',
            [
                'Diabetes Prediction',
                'Heart Disease Prediction',
                'Parkinsons Prediction',
                'Lung Cancer Prediction',
                'Hypo-Thyroid Prediction'
            ]
        )
        st.markdown("---")
        # Quick stats
        st.subheader("📈 Quick Stats")
        if 'prediction_history' in st.session_state:
            total_predictions = len(st.session_state.prediction_history)
            st.metric("Total Predictions", total_predictions)
        else:
            st.metric("Total Predictions", 0)
        st.markdown("---")
        # Show history toggle
        show_history = st.checkbox("📊 Show Prediction History")

    # Main content area
    if show_history:
        display_prediction_history()
    else:
        # Display disease information
        display_disease_info(selected)
        st.markdown("---")

        # Disease-specific prediction forms
        if selected == 'Diabetes Prediction' and models['diabetes']:
            diabetes_prediction(models['diabetes'])
        elif selected == 'Heart Disease Prediction' and models['heart_disease']:
            heart_disease_prediction(models['heart_disease'])
        elif selected == 'Parkinsons Prediction' and models['parkinsons']:
            parkinsons_prediction(models['parkinsons'])
        elif selected == 'Lung Cancer Prediction' and models['lung_cancer']:
            lung_cancer_prediction(models['lung_cancer'])
        elif selected == 'Hypo-Thyroid Prediction' and models['thyroid']:
            thyroid_prediction(models['thyroid'])
        else:
            st.error(f"Model for {selected} is not available. Please check the model file.")

def diabetes_prediction(model):
    st.subheader("🩺 Diabetes Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = create_input_field('Number of Pregnancies', 'Number of times pregnant', 'pregnancies', max_val=20)
        glucose = create_input_field('Glucose Level (mg/dL)', 'Plasma glucose concentration', 'glucose', max_val=300)
        blood_pressure = create_input_field('Blood Pressure (mm Hg)', 'Diastolic blood pressure', 'blood_pressure', max_val=200)
        skin_thickness = create_input_field('Skin Thickness (mm)', 'Triceps skinfold thickness', 'skin_thickness', max_val=100)
    with col2:
        insulin = create_input_field('Insulin Level (μU/mL)', '2-Hour serum insulin', 'insulin', max_val=1000)
        bmi = create_input_field('BMI', 'Body mass index (weight in kg/(height in m)^2)', 'bmi', max_val=70.0, step=0.1)  # Fixed type
        dpf = create_input_field('Diabetes Pedigree Function', 'Diabetes pedigree function score', 'dpf', max_val=5.0, step=0.01)
        age = create_input_field('Age (years)', 'Age of the person', 'age', max_val=120)

    if st.button('🔍 Predict Diabetes Risk', type="primary"):
        try:
            input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
            if any(val is None or val < 0 for val in input_data):
                st.error("Please fill in all fields with valid non-negative values.")
                return
            prediction = model.predict([input_data])
            timestamp = datetime.now().isoformat()
            save_prediction_history('Diabetes', input_data, prediction, timestamp)
            display_prediction_result(prediction, 'Diabetes')
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def heart_disease_prediction(model):
    st.subheader("❤️ Heart Disease Risk Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = create_input_field('Age (years)', 'Age of the person', 'hd_age', max_val=120)
        sex = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', key='hd_sex')
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3],
                         format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x],
                         key='hd_cp')
        trestbps = create_input_field('Resting Blood Pressure (mm Hg)', 'Resting blood pressure', 'hd_trestbps', max_val=250)
        chol = create_input_field('Cholesterol (mg/dL)', 'Serum cholesterol', 'hd_chol', max_val=600)
    with col2:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1],
                          format_func=lambda x: 'No' if x == 0 else 'Yes', key='hd_fbs')
        restecg = st.selectbox('Resting ECG', [0, 1, 2],
                              format_func=lambda x: ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][x],
                              key='hd_restecg')
        thalach = create_input_field('Max Heart Rate', 'Maximum heart rate achieved', 'hd_thalach', max_val=250)
        exang = st.selectbox('Exercise Induced Angina', [0, 1],
                            format_func=lambda x: 'No' if x == 0 else 'Yes', key='hd_exang')
    with col3:
        oldpeak = create_input_field('ST Depression', 'ST depression induced by exercise', 'hd_oldpeak', max_val=10.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST', [0, 1, 2],
                            format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x],
                            key='hd_slope')
        ca = create_input_field('Major Vessels (0-3)', 'Number of major vessels colored by fluoroscopy', 'hd_ca', max_val=3)
        thal = st.selectbox('Thalassemia', [0, 1, 2],
                           format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect'][x],
                           key='hd_thal')

    if st.button('🔍 Predict Heart Disease Risk', type="primary"):
        try:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction = model.predict([input_data])
            timestamp = datetime.now().isoformat()
            save_prediction_history('Heart Disease', input_data, prediction, timestamp)
            display_prediction_result(prediction, 'Heart Disease')
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def parkinsons_prediction(model):
    st.subheader("🧠 Parkinson's Disease Risk Assessment")
    st.info("This assessment uses voice measurement data. Please consult with a healthcare professional for proper voice analysis.")

    with st.expander("📊 Fundamental Frequency Measures", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = create_input_field('MDVP:Fo(Hz)', 'Average vocal fundamental frequency', 'pk_fo', max_val=300.0, step=0.01)
        with col2:
            fhi = create_input_field('MDVP:Fhi(Hz)', 'Maximum vocal fundamental frequency', 'pk_fhi', max_val=600.0, step=0.01)
        with col3:
            flo = create_input_field('MDVP:Flo(Hz)', 'Minimum vocal fundamental frequency', 'pk_flo', max_val=300.0, step=0.01)

    with st.expander("📈 Jitter Measures"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            jitter_percent = create_input_field('Jitter (%)', 'Jitter percentage', 'pk_jitter_percent', max_val=10.0, step=0.001)
        with col2:
            jitter_abs = create_input_field('Jitter (Abs)', 'Absolute jitter', 'pk_jitter_abs', max_val=1.0, step=0.00001)
        with col3:
            rap = create_input_field('RAP', 'Relative average perturbation', 'pk_rap', max_val=1.0, step=0.001)
        with col4:
            ppq = create_input_field('PPQ', 'Period perturbation quotient', 'pk_ppq', max_val=1.0, step=0.001)
    ddp = create_input_field('Jitter:DDP', 'Jitter DDP', 'pk_ddp', max_val=1.0, step=0.001)

    with st.expander("📉 Shimmer Measures"):
        col1, col2, col3 = st.columns(3)
        with col1:
            shimmer = create_input_field('Shimmer', 'Shimmer', 'pk_shimmer', max_val=1.0, step=0.001)
            shimmer_db = create_input_field('Shimmer (dB)', 'Shimmer in dB', 'pk_shimmer_db', max_val=10.0, step=0.01)
        with col2:
            apq3 = create_input_field('APQ3', 'Amplitude perturbation quotient 3', 'pk_apq3', max_val=1.0, step=0.001)
            apq5 = create_input_field('APQ5', 'Amplitude perturbation quotient 5', 'pk_apq5', max_val=1.0, step=0.001)
        with col3:
            apq = create_input_field('APQ', 'Amplitude perturbation quotient', 'pk_apq', max_val=1.0, step=0.001)
            dda = create_input_field('DDA', 'Difference of differences of amplitudes', 'pk_dda', max_val=1.0, step=0.001)

    with st.expander("🎵 Harmonic Measures"):
        col1, col2 = st.columns(2)
        with col1:
            nhr = create_input_field('NHR', 'Noise-to-harmonics ratio', 'pk_nhr', max_val=1.0, step=0.001)
        with col2:
            hnr = create_input_field('HNR', 'Harmonics-to-noise ratio', 'pk_hnr', max_val=50.0, step=0.01)

    with st.expander("🔬 Nonlinear Dynamical Complexity Measures"):
        col1, col2, col3 = st.columns(3)
        with col1:
            rpde = create_input_field('RPDE', 'Recurrence period density entropy', 'pk_rpde', max_val=1.0, step=0.001)
            dfa = create_input_field('DFA', 'Detrended fluctuation analysis', 'pk_dfa', max_val=1.0, step=0.001)
        with col2:
            spread1 = create_input_field('Spread1', 'Nonlinear measure of fundamental frequency variation', 'pk_spread1', min_val=-10.0, max_val=0.0, step=0.001)
            spread2 = create_input_field('Spread2', 'Nonlinear measure of fundamental frequency variation', 'pk_spread2', max_val=1.0, step=0.001)
        with col3:
            d2 = create_input_field('D2', 'Correlation dimension', 'pk_d2', max_val=10.0, step=0.001)
            ppe = create_input_field('PPE', 'Pitch period entropy', 'pk_ppe', max_val=1.0, step=0.001)

    if st.button('🔍 Predict Parkinson\'s Risk', type="primary"):
        try:
            input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                         apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
            prediction = model.predict([input_data])
            timestamp = datetime.now().isoformat()
            save_prediction_history('Parkinsons', input_data, prediction, timestamp)
            display_prediction_result(prediction, 'Parkinson\'s Disease')
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def lung_cancer_prediction(model):
    st.subheader("🫁 Lung Cancer Risk Assessment")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', key='lc_gender')
        age = create_input_field('Age (years)', 'Age of the person', 'lc_age', max_val=120)
        smoking = st.selectbox('Smoking History', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_smoking')
        yellow_fingers = st.selectbox('Yellow Fingers', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_yellow_fingers')
        anxiety = st.selectbox('Anxiety', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_anxiety')
    with col2:
        peer_pressure = st.selectbox('Peer Pressure', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_peer_pressure')
        chronic_disease = st.selectbox('Chronic Disease', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_chronic_disease')
        fatigue = st.selectbox('Fatigue', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_fatigue')
        allergy = st.selectbox('Allergy', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_allergy')
        wheezing = st.selectbox('Wheezing', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_wheezing')
    with col3:
        alcohol_consuming = st.selectbox('Alcohol Consumption', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_alcohol')
        coughing = st.selectbox('Coughing', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_coughing')
        shortness_of_breath = st.selectbox('Shortness of Breath', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_shortness')
        swallowing_difficulty = st.selectbox('Swallowing Difficulty', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_swallowing')
        chest_pain = st.selectbox('Chest Pain', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='lc_chest_pain')

    if st.button('🔍 Predict Lung Cancer Risk', type="primary"):
        try:
            input_data = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                         fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath,
                         swallowing_difficulty, chest_pain]
            prediction = model.predict([input_data])
            timestamp = datetime.now().isoformat()
            save_prediction_history('Lung Cancer', input_data, prediction, timestamp)
            display_prediction_result(prediction, 'Lung Cancer')
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

def thyroid_prediction(model):
    st.subheader("🦋 Thyroid Disease Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        age = create_input_field('Age (years)', 'Age of the person', 'th_age', max_val=120)
        sex = st.selectbox('Gender', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male', key='th_sex')
        on_thyroxine = st.selectbox('On Thyroxine Medication', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='th_on_thyroxine')
        tsh = create_input_field('TSH Level (mU/L)', 'Thyroid Stimulating Hormone level', 'th_tsh', max_val=100.0, step=0.01)
    with col2:
        t3_measured = st.selectbox('T3 Measured', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', key='th_t3_measured')
        t3 = create_input_field('T3 Level (ng/dL)', 'Triiodothyronine level', 'th_t3', max_val=10.0, step=0.01)
        tt4 = create_input_field('TT4 Level (μg/dL)', 'Total thyroxine level', 'th_tt4', max_val=30.0, step=0.1)

    if st.button('🔍 Predict Thyroid Disease Risk', type="primary"):
        try:
            input_data = [age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]
            if any(val is None or val < 0 for val in input_data):
                st.error("Please fill in all fields with valid non-negative values.")
                return
            prediction = model.predict([input_data])
            timestamp = datetime.now().isoformat()
            save_prediction_history('Hypo-Thyroid', input_data, prediction, timestamp)
            display_prediction_result(prediction, 'Hypothyroidism')
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
