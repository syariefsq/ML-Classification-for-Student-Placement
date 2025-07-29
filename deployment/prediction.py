import pandas as pd
import streamlit as st
import pickle
import os

# Load Model
MODEL_PATH = './src/model_xgb.pkl'
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()
with open(MODEL_PATH, 'rb') as file_1:
    model_xgb = pickle.load(file_1)

# Median values for each numeric field (based on EDA or typical dataset medians)
MEDIANS = {
    "Secondary_Percentage": 67.0,
    "Higher_Secondary_Percentage": 65.0,
    "Degree_Percentage": 66.0,
    "Entrance_Test_Percentage": 60.0,
    "Mba_Percentage": 62.0,
    "Salary": 200000.0
}

def run():
    st.markdown("""
    <style>
    .section-card {
        background: #f8fafd;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        padding: 1.5em 1.5em 1em 1.5em;
        margin-bottom: 1.5em;
        border: 1px solid #e3e8ee;
    }
    .predict-btn button {
        background: linear-gradient(90deg, #1f77b4 0%, #4f8cff 100%);
        color: white !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.7em 2em;
        font-size: 1.1em;
        margin-top: 1em;
        box-shadow: 0 2px 8px rgba(31,119,180,0.08);
    }
    .predict-btn button:hover {
        background: linear-gradient(90deg, #18609c 0%, #3576b8 100%);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>div>input {
        border-radius: 6px;
        border: 1px solid #cfd8dc;
        background: #fafdff;
        padding: 0.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Placement Prediction App")
    st.markdown("""
    Use this tool to estimate the placement outcome for a student based on their academic, demographic, and experience profile. Fill in the details below and click <b>Predict Placement</b> to see the result.
    """, unsafe_allow_html=True)
    st.image("https://media.tenor.com/gs0NfYUZMpIAAAAC/baby-yoda.gif", use_column_width=True)

    with st.form("placement_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Demographic Information")
        gender = st.selectbox("Gender", ["M", "F"], index=0)  # 'M' more likely placed
        work_experience = st.selectbox("Work Experience", ["Yes", "No"], index=0)  # 'Yes' more likely placed
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Secondary Education")
        secondary_percentage = st.slider("Secondary Percentage", min_value=0.0, max_value=100.0, value=85.0, step=0.01)  # high value
        secondary_board = st.selectbox("Secondary Board", ["Central", "Others"], index=0)  # 'Central' more likely placed
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Higher Secondary Education")
        higher_secondary_percentage = st.slider("Higher Secondary Percentage", min_value=0.0, max_value=100.0, value=85.0, step=0.01)  # high value
        higher_secondary_board = st.selectbox("Higher Secondary Board", ["Central", "Others"], index=0)
        higher_secondary_specialization = st.selectbox("Higher Secondary Specialization", ["Science", "Commerce", "Arts"], index=0)  # 'Science' more likely placed
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Degree Information")
        degree_percentage = st.slider("Degree Percentage", min_value=0.0, max_value=100.0, value=85.0, step=0.01)  # high value
        degree_type = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"], index=0)  # 'Sci&Tech' more likely placed
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Entrance Test & MBA")
        entrance_test_percentage = st.slider("Entrance Test Percentage", min_value=0.0, max_value=100.0, value=85.0, step=0.01)  # high value
        specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"], index=0)  # 'Mkt&HR' more likely placed
        mba_percentage = st.slider("MBA Percentage", min_value=0.0, max_value=100.0, value=85.0, step=0.01)  # high value
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Placement Details")
        # Placement_Status and Salary are not features for prediction, so we remove them from the form
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Placement")
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            data_inf = pd.DataFrame([{
                "Gender": gender,
                "Secondary_Percentage": secondary_percentage,
                "Secondary_Board": secondary_board,
                "Higher_Secondary_Percentage": higher_secondary_percentage,
                "Higher_Secondary_Board": higher_secondary_board,
                "Higher_Secondary_Specialization": higher_secondary_specialization,
                "Degree_Percentage": degree_percentage,
                "Degree_Type": degree_type,
                "Work_Experience": work_experience,
                "Entrance_Test_Percentage": entrance_test_percentage,
                "Specialisation": specialisation,
                "Mba_Percentage": mba_percentage
            }])

            st.markdown('---')
            st.markdown("#### Input Data Summary")
            st.write(data_inf)

            # Make prediction (model returns 1 for Placed, 0 for Not Placed)
            y_pred = model_xgb.predict(data_inf)[0]
            y_pred_label = 'Placed' if y_pred == 1 else 'Not Placed'
            prediction_proba = model_xgb.predict_proba(data_inf)[0]
            proba = prediction_proba[1] if y_pred == 1 else prediction_proba[0]

            if y_pred_label == "Placed":
                result_str = "🎉 Congratulations! The model predicts this student is likely to be <b>Placed</b>."
                why_str = "This result is influenced by strong academic performance, relevant degree, and/or prior work experience, all of which are key factors identified in the EDA."
                gif_url = "https://media.tenor.com/_r-UUCjuC9MAAAAC/congratulations-congrats.gif"  # celebratory, job offer
            else:
                result_str = "🙁 Unfortunately, the model predicts this student is <b>Not Placed</b>."
                why_str = "This may be due to lower academic scores, less relevant specialization, or lack of work experience, as highlighted in the EDA. Consider focusing on these areas to improve placement chances."
                gif_url = "https://media.tenor.com/Zd8r8YbaLVUAAAAC/youre-not-gonna-get-the-job-ronnie.gif"  # motivational, try again

            st.markdown('---')
            st.subheader("Prediction Result")
            st.markdown(f"{result_str}", unsafe_allow_html=True)
            st.markdown(f"<b>Why?</b> {why_str}", unsafe_allow_html=True)
            st.markdown(f"Probability: {proba:.2%}")
            st.markdown("<i>The probability reflects the model's confidence in its prediction, based on the input features. A value closer to 100% means higher confidence.</i>", unsafe_allow_html=True)
            st.image(gif_url, use_column_width=True)

if __name__ == "__main__":
    run()