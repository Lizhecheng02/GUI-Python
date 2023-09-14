from river.compose import Pipeline
from river.feature_extraction import BagOfWords, TFIDF
from river.naive_bayes import MultinomialNB
from datetime import datetime
import altair as alt
import seaborn as sns
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

data = [("my unit test failed", "Software"),
        ("tried the program, but it was buggy", "Software"),
        ("i need a new power supply", "Hardware"),
        ("the drive has a 2TB capacity", "Hardware"),
        ("unit-tests", "Software"),
        ("program", "Software"),
        ("power supply", "Hardware"),
        ("drive", "Hardware"),
        ("it needs more memory", "Hardware"),
        ("check the API", "Software"),
        ("design the API", "Software"),
        ("they need more CPU", "Hardware"),
        ("code", "Software"),
        ("i found some bugs in the code", "Software"),
        ("i swapped the memory", "Hardware"),
        ("i tested the code", "Software")]


model = Pipeline(('vectorizer', BagOfWords(
    lowercase=True)), ('nv', MultinomialNB))

for X, y in data:
    model = model.learn_one(X, y)

conn = sqlite3.connect('GUI\streamlit\Database\Text Classifier App\data.db')
c = conn.cursor()


def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT, prediction TEXT, probability NUMBER, software_proba NUMBER, hardware_proba NUMBER, postdate DATE)')


def add_data(message, prediction, probability, software_proba, hardware_proba, postdate):
    c.execute('INSERT INTO predictionTable(message, prediction, probability, software_proba, hardware_proba, postdate) VALUES (?, ?, ?, ?, ?, ?)',
              (message, prediction, probability, software_proba, hardware_proba, postdate))
    conn.commit()


def view_all_data():
    c.execute('SELECT * FROM predictionTable')
    data = c.fetchall()
    return data


def main():
    menu = ['Home', 'Manage', 'About']
    create_table()

    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.subheader('Home')
        with st.form(key='mlform'):
            col1, col2 = st.columns([2, 1])
            with col1:
                message = st.text_area('Message')
                submit_message = st.form_submit_button(label='Predict')
            with col2:
                st.write('Online Incremental ML')
                st.info('Predict Text as Software or Hardware Related')

        if submit_message:
            prediction = model.predict_one(message)
            prediction_proba = model.predict_proba_one(message)
            probability = max(prediction_proba.values())
            postdate = datetime.now()
            add_data(message, prediction, probability,
                     prediction_proba['Software'], prediction_proba['Hardware'], postdate)
            st.success('Data Submitted !')

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info('Original Text')
                st.write(message)
                st.success('Prediction')
                st.write(prediction)
            with res_col2:
                st.info('Probability')
                st.write(prediction_proba)
                df_proba = pd.DataFrame({
                    'label': prediction_proba.keys(),
                    'probability': prediction_proba.values()
                })
                fig = alt.Chart(df_proba).mark_bar().encode(
                    x='label', y='probability')
                st.altair_chart(fig, use_container_width=True)

    elif choice == 'Manage':
        st.subheader('Manage & Monitor Results')
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data, columns=[
                              'message', 'prediction', 'probability', 'software_proba', 'hardware_proba', 'postdate'])
        st.dataframe(new_df)
        new_df['postdate'] = pd.to_datetime(new_df['postdate'])

        c = alt.Chart(new_df).mark_line().encode(x='postdate', y='probability')
        st.altair_chart(c)

        c_software_proba = alt.Chart(new_df['software_proba'].reset_index(
        )).mark_line().encode(x='software_proba', y='index')
        c_hardware_proba = alt.Chart(new_df['hardware_proba'].reset_index(
        )).mark_line().encode(x='hardware_proba', y='index')

        c1, c2 = st.columns(2)
        with c1:
            with st.expander('Software Probability'):
                st.altair_chart(c_software_proba, use_container_width=True)
        with c2:
            with st.expander('Hardware Probability'):
                st.altair_chart(c_hardware_proba, use_container_width=True)

        with st.expander('Prediction Distribution'):
            fig2 = plt.figure()
            sns.countplot(x='probability', data=new_df)
            st.pyplot(fig2)

    else:
        st.subheader('About')
        st.success('This App is Created By Zhecheng Li')


if __name__ == '__main__':
    main()
