from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
import os
import glob
import pandas as pd
import shutil
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from zipfile import ZipFile

matplotlib.use('Agg')


def main():
    st.title('Common ML Data Explorer')
    st.subheader('Simple ML App with Streamlit')

    html_temp = """
	<div style='background-color:tomato;'>
    <p style='color:white;font-size:45px;'>Streamlit is Awesome</p>
    </div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    def file_selector(folder_path='GUI\streamlit\ML Dataset App\Data'):
        file_names = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', file_names)
        return os.path.join(folder_path, selected_filename)

    file_name = file_selector()
    st.write(f'You select {file_name}')
    df = pd.read_csv(file_name)

    if st.checkbox('Show Dataset'):
        number = st.number_input(
            'Number of Rows to View', step=1, min_value=1, max_value=len(df)
        )
        st.dataframe(df.head(number))

    if st.button('Columns Names'):
        st.write(df.columns.to_list())

    if st.checkbox('Shape of Dataset'):
        st.write(df.shape)
        data_dim = st.radio('Show dimension by', ('Rows', 'Columns'))
        if data_dim == 'Rows':
            st.text('Number of Rows')
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text('Number of Columns')
            st.write(df.shape[1])

    if st.checkbox('Select Column to Show'):
        all_columns = df.columns.to_list()
        selected_columns = st.multiselect('Select', all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    if st.button('Data Types'):
        st.write(df.dtypes)

    if st.checkbox('Value Counts'):
        st.text('Value Counts By Target / Class')
        st.write(df.iloc[:, -1].value_counts())

    if st.checkbox('Summary'):
        st.write(df.describe().T)

    st.subheader('Data Visualization')
    if st.checkbox('Correlation Plot [Matplotlib]'):
        plt.matshow(df.corr())
        st.pyplot()

    if st.checkbox('Correlation Plot with Annotation [Seaborn]'):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    if st.checkbox('Plot of Value Counts'):
        st.text('Value Counts By Target / Class')
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox(
            'Select Primary Column to GroupBy', all_columns_names)
        selected_column_names = st.multiselect(
            'Select Columns', all_columns_names)
        if st.button('Plot'):
            st.text('Generating Plot For: {} and {}'.format(
                primary_col, selected_column_names))
            if selected_column_names:
                vc_plot = df.groupby(primary_col)[
                    selected_column_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

    if st.checkbox('Pie Plot'):
        all_columns_names = df.columns.tolist()
        if st.button('Generate Pie Plot'):
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct='%1.1f%%',
                     labeldistance=0.6, pctdistance=0.3))
            st.pyplot()

    if st.checkbox('Barh Plot'):
        all_columns_names = df.columns.tolist()
        st.info('Please Choose the X and Y Column')
        X_column = st.selectbox(
            'Select X column for Barh Plot', all_columns_names)
        Y_column = st.selectbox(
            'Select Y column for Barh Plot', all_columns_names)
        barh_plot = df.plot.barh(x=X_column, y=Y_column, figsize=(10, 10))
        if st.button('Generate Barh Plot'):
            st.write(barh_plot)
            st.pyplot()

    st.subheader('Customizable Plots')
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox('Select the Type of Plot', [
                                'area', 'bar', 'line', 'hist', 'box', 'kde'])
    selected_column_names = st.multiselect(
        'Select Columns to Plot', all_columns_names)
    plot_fig_height = st.number_input(
        'Choose Fig Size for Height', min_value=1, max_value=15, step=1)
    plot_fig_width = st.number_input(
        'Choose Fig Size for Width', min_value=1, max_value=15, step=1)
    plot_fig_size = (plot_fig_height, plot_fig_width)
    cust_target = df.iloc[:, -1].name

    if st.button('Generate Various Plot'):
        st.success('Generating a customizable Plot of : {} for :: {}'.format(
            type_of_plot, selected_column_names))

        if type_of_plot == 'area':
            cust_data = df[selected_column_names]
            # st.area_chart(cust_data, width=plot_fig_width,
            #               height=plot_fig_height)
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':
            cust_data = df[selected_column_names]
            st.bar_chart(cust_data)
            # st.bar_chart(cust_data, width=plot_fig_width,
            #               height=plot_fig_height)
        elif type_of_plot == 'line':
            cust_data = df[selected_column_names]
            st.line_chart(cust_data)
            # st.line_chart(cust_data, width=plot_fig_width,
            #               height=plot_fig_height)
        else:
            custom_plot = df[selected_column_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()

    st.subheader('Our Features and Targets')

    if st.checkbox('Show Features'):
        all_features = df.iloc[:, 0:1]
        st.text('Feature names: {}'.format(all_features.columns[0:-1]))
        st.dataframe(all_features.head(10))

    if st.checkbox('Show Target'):
        target = df.iloc[:, -1]
        st.text('Target / Class Name: {}'.format(target))
        st.dataframe(target.head(10))

    # st.markdown("""[Dataset Website](iris.zip)""")

    # def makezipfile(data):
    #     output_filename = '{}_zipped.zip'.format(data)
    #     with ZipFile(output_filename, 'w') as z:
    #         z.write(data)
    #     return output_filename

    # if st.button('Download File'):
    #     DOWNLOAD_TPL = f'[{file_name}]({makezipfile(file_name)})'
    #     st.text(DOWNLOAD_TPL)
    #     st.markdown(DOWNLOAD_TPL)


if __name__ == '__main__':
    main()
