import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance


def main():
    st.title('Iris EDA App')
    st.subheader('EDA Web App with Streamlit')

    dataset = 'GUI\streamlit\Iris_App\iris.csv'

    @st.cache_data(persist=True)
    def explore_data(file_path):
        df = pd.read_csv(file_path)
        return df

    data = explore_data(dataset)

    if st.checkbox('Preview Dataset'):
        if st.button('Head'):
            st.write(data.head())
        if st.button('Tail'):
            st.write(data.tail())
        # else:
        #     st.write(data.head(2))

    if st.checkbox('Show all DataFrame'):
        st.dataframe(data)

    if st.checkbox('Show all Column names'):
        st.text('Columns: ')
        st.write(data.columns.to_list())

    data_dim = st.radio(
        'Which dimension do you want to show:', ('Rows', 'Columns'))
    if data_dim == 'Rows':
        st.text('Showing Length of Rows')
        st.write(len(data))
    if data_dim == 'Columns':
        st.text('Showing Length of Columns')
        st.write(data.shape[1])

    if st.checkbox('Show summary of Dataset'):
        st.write(data.describe().T)

    species_option = st.selectbox(
        'Select Columns:', ('sepal_length', 'sepal_width', 'petal_length', 'petal_width'))
    if species_option == 'sepal_length':
        st.write(data['sepal_length'])
    elif species_option == 'sepal_width':
        st.write(data['sepal_width'])
    elif species_option == 'petal_length':
        st.write(data['petal_length'])
    elif species_option == 'petal_width':
        st.write(data['petal_width'])
    else:
        st.write('Select a column !!!')

    if st.checkbox('Simple hist plot with matplotlib'):
        data.plot(kind='hist')
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    if st.checkbox('Simple correlation plot with matplotlib'):
        plt.matshow(data.corr())
        st.pyplot()

    if st.checkbox('Simple correlation plot with seaborn'):
        st.write(sns.heatmap(data.corr(), annot=True, fmt='.2g'))
        st.pyplot()

    # if st.checkbox('Bar plot of groups or counts'):
    #     data = explore_data(dataset)
    #     v_counts = data.groupby('species')
    #     st.bar_chart(v_counts)

    @st.cache_data
    def load_image(img_path):
        img = Image.open(img_path)
        return img

    species_type = st.radio('Which species in Iris dataset that you want to see?',
                            ('Setosa', 'Versicolor', 'Virginica'))
    if species_type == 'Setosa':
        st.text('Showing Setosa Species')
        st.image(load_image('GUI\streamlit\Iris_App\images\iris_setosa.jpg'))
    elif species_type == 'Versicolor':
        st.text('Showing Versicolor Species')
        st.image(load_image('GUI\streamlit\Iris_App\images\iris_versicolor.jpg'))
    elif species_type == 'Virginica':
        st.text('Showing Virginica Species')
        st.image(load_image('GUI\streamlit\Iris_App\images\iris_virginica.jpg'))

    if st.checkbox('Show Image / Hide Image'):
        my_image = load_image('GUI\streamlit\Iris_App\images\iris_setosa.jpg')
        enh = ImageEnhance.Contrast(my_image)
        num = st.slider('Set your contrast number', 1.0, 3.0)
        img_width = st.slider('Set image width', 300, 500)
        st.image(enh.enhance(num), width=img_width)

    if st.button('About App'):
        st.subheader('Iris dataset EDA App')
        st.text('Built with Streamlit')
        st.text('Thanks to the Streamlit Team Amazing Work')

    if st.checkbox('By'):
        st.text('Zhecheng Li')


if __name__ == '__main__':
    main()
