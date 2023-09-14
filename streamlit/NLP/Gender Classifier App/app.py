import streamlit as st
# import joblib
import time
import pickle
# import sklearn
# from sklearn.externals import joblib
from PIL import Image

# gender_vectorizer = open(
#     'GUI/streamlit/Gender Classifier/Models/gender_vectorizer.pkl', 'rb')
# gender_cv = joblib.load(gender_vectorizer)

# gender_cv = joblib.load(
#     'GUI/streamlit/Gender Classifier/Models/gender_vectorizer.pkl')

gender_cv = pickle.load(
    open('GUI/streamlit/Gender Classifier/Models/gender_vectorizer.pkl', 'rb'))

# gender_nv_model = open(
#     'GUI/streamlit/Gender Classifier/Models/naivebayesgendermodel.pkl', 'rb')
# gender_clf = joblib.load(gender_nv_model)

# gender_clf = joblib.load(
#     'GUI/streamlit/Gender Classifier/Models/naivebayesgendermodel.pkl')

gender_clf = pickle.load(
    open('GUI/streamlit/Gender Classifier/Models/naivebayesgendermodel.pkl', 'rb'))


def predict_gender(data):
    vec = gender_cv.transform(data).toarray()
    result = gender_clf.predict(vec)
    return result


def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name),
                unsafe_allow_html=True)


def load_image(file_name):
    img = Image.open(file_name)
    return st.image(img, width=100)


def main():
    st.title('Gender Classifier')
    html_temp = """
	<div style='background-color:tomato;padding:10px'>
	<h2 style='color:white;text-align:center;'>Streamlit ML App</h2>
	</div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    load_css('GUI\streamlit\Gender Classifier\Data\icon.css')
    load_icon('people')

    name = st.text_input('Enter Name', 'Type Here')
    if st.button('Predict'):
        result = predict_gender([name])
        if result[0] == 0:
            prediction = 'Female'
            c_img = 'GUI\streamlit\Gender Classifier\Data\female.png'
        elif result[0] == 1:
            prediction = 'Male'
            c_img = 'GUI\streamlit\Gender Classifier\Data\male.png'

        st.success('Name: {} is classifiered as {}'.format(
            name.title(), prediction))
        load_image(c_img)

    if st.button('About'):
        st.text('Zhecheng Li')
        st.text('Built with Streamlit')


if __name__ == '__main__':
    main()
