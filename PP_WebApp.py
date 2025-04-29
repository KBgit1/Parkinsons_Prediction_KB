import numpy as np
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu

# Load the model and the scaler from the .sav file
loaded_model, loaded_scaler = pickle.load(open('trained_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Parkinsons Disease Prediction System using ML',
                           ['Home Page',
                            'Diagnosis',
                            'Help'],
                           icons=['hospital-fill', 'clipboard2-pulse-fill', 'compass'],
                           default_index=0)

# Home Page
if selected == 'Home Page':
    st.title('What is Parkinson’s Disease?')

    st.image("https://img.freepik.com/free-vector/gradient-parkinson-infographic_52683-81811.jpg?ga=GA1.1.1217230711.1745908830&semt=ais_hybrid&w=740", 
             caption="Parkinson’s Disease Illustration", use_container_width=True)

    st.markdown("""
    ### Overview
    Parkinson's Disease is a chronic and progressive neurological disorder that primarily affects movement. 
    It develops gradually, sometimes starting with a barely noticeable tremor in just one hand. 
    But while tremors are the most well-known sign of Parkinson's disease, the disorder also commonly causes stiffness or slowing of movement.

    ### Common Symptoms:
    - Tremor, often in hands, arms, legs, jaw, or head
    - Bradykinesia (slowness of movement)
    - Muscle rigidity
    - Postural instability and balance issues
    - Speech and writing changes

    ### Causes:
    Parkinson's Disease is caused by the degeneration of neurons in a specific area of the brain called the *substantia nigra*, 
    which leads to a drop in dopamine levels.

    ### Diagnosis:
    Parkinson's is typically diagnosed based on medical history, symptoms, and neurological and physical exams. 
    Machine learning models trained on voice and movement data can assist in early prediction and monitoring.

    > This web application utilizes a Machine Learning model to predict the likelihood of Parkinson's Disease based on key vocal measurements.
    """)

    st.image("https://www.mdpi.com/diagnostics/diagnostics-11-01892/article_deploy/html/images/diagnostics-11-01892-g001.png", use_container_width=True)

# Diagnosis Page
elif selected == 'Diagnosis':
    st.title('Fill the entries for Testing')

    col1, col2, col3 = st.columns(3)

    # Input fields
    with col1:
            MDVP_Fo_Hz = st.text_input('Avg. pitch freq')
    with col2:
            MDVP_Fhi_Hz = st.text_input('Max pitch freq')
    with col3:
            MDVP_Flo_Hz = st.text_input('Min pitch freq')
    with col1:
            MDVP_Jitter_percent = st.text_input('Pitch var (%)')
    with col2:
            MDVP_Jitter_Abs = st.text_input('Pitch var (sec)')
    with col3:
            MDVP_RAP = st.text_input('Short-term jitter')
    with col1:
            MDVP_PPQ = st.text_input('Smooth pitch var')
    with col2:
            Jitter_DDP = st.text_input('RAP diff avg')
    with col3:
            MDVP_Shimmer = st.text_input('Amp var (%)')
    with col1:
            MDVP_Shimmer_dB = st.text_input('Amp var (dB)')
    with col2:
            Shimmer_APQ3 = st.text_input('3-pt amp var')
    with col3:
            Shimmer_APQ5 = st.text_input('5-pt amp var')
    with col1:
            MDVP_APQ = st.text_input('Avg amp var')
    with col2:
            Shimmer_DDA = st.text_input('DDA (amp diff)')
    with col3:
            NHR = st.text_input('Noise/harmonics')
    with col1:
            HNR = st.text_input('Harmonics/noise')
    with col2:
            RPDE = st.text_input('Signal irregularity')
    with col3:
            DFA = st.text_input('Fractal analysis')
    with col1:
            spread1 = st.text_input('Freq spread')
    with col2:
            spread2 = st.text_input('Amp spread')
    with col3:
            D2 = st.text_input('Signal complexity')
    with col1:
            PPE = st.text_input('Pitch entropy')

    # Prediction function
    def parkinsons_prediction(input_list):
        input_array = np.asarray(input_list, dtype=np.float64).reshape(1, -1)
        std_data = loaded_scaler.transform(input_array)
        prediction = loaded_model.predict(std_data)
        return 'The person has Parkinson’s Disease' if prediction[0] == 1 else 'The person is healthy'

    # Button
    if st.button('Test Result'):
        try:
            user_input = [
                MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz,
                MDVP_Jitter_percent, MDVP_Jitter_Abs, MDVP_RAP,
                MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
                NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
            ]
            # Convert all input to float
            user_input = [float(i) for i in user_input]
            diagnosis = parkinsons_prediction(user_input)
            st.success(diagnosis)
        except ValueError:
            st.error("Please make sure all fields are filled with valid numbers.")

# Help Page
elif selected == 'Help':
    st.title('Need Help?')
    st.markdown("""
    ### How to Use This App:
    1. Navigate to the **Diagnosis** tab.
    2. Fill in the numeric values for the 22 voice-related parameters.
    3. Click **Test Result** to see if the person is predicted to have Parkinson's Disease.

    ### Where to Get Parameters:
    These values are usually derived from vocal recordings using specialized audio analysis tools.
    If you're not a clinician or don't have these tools, please consult a medical professional for accurate testing.

    ### Data Source:
    The model used here is trained on the **Parkinson’s Disease dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

    ### About the Model:
    This prediction system uses a trained **Machine Learning model** (like Random Forest, SVM, or Logistic Regression) 
    that has learned patterns from real-world patient data. The results are **indicative only**, and not a substitute for professional medical advice.

    ### Contact:
    - 📧 Email: support@parkinsons-predict.ai
    - 📞 Helpline: +1-800-PARKINSON
    """)
