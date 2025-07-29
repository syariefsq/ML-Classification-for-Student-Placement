import streamlit as st
import eda
import prediction

# membuat page lebih lebar
st.set_page_config(
    page_title='University Job Placement', 
    initial_sidebar_state='expanded'
)


page = st.sidebar.selectbox('Pilih Halaman', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()

else:
    prediction.run()