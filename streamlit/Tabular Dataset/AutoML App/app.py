from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('GUI/streamlit/AutoML/Data/dataset.csv'):
    df = pd.read_csv('GUI/streamlit/AutoML/Data/dataset.csv', index_col=None)

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('AutoML')
    choice = st.radio(
        'Navigation', ['Upload', 'Profiling', 'Modelling', 'Download'])
    st.info('This project application helps you build and explore your data.')

if choice == 'Upload':
    st.title('Upload your Dataset')
    file = st.file_uploader('Upload')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('GUI/streamlit/AutoML/Data/dataset.csv', index=None)
        st.dataframe(df)

if choice == 'Profiling':
    st.title('Exploratory Data Analysis')
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == 'Modelling':
    chosen_target = st.selectbox('Choose the target column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'GUI/streamlit/AutoML/best_model')

if choice == 'Download':
    with open('GUI/streamlit/AutoML/best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name='best_model.pkl')
