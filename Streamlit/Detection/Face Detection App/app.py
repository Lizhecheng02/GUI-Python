import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance


@st.cache
def load_image(img_path):
    img = Image.open(img_path)
    return img


face_cascade = cv2.CascadeClassifier(
    'GUI\streamlit\Detection\Face Detection App\Models\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'GUI\streamlit\Detection\Face Detection App\Models\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(
    'GUI\streamlit\Detection\Face Detection App\Models\haarcascade_smile.xml')


def detect_faces(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces


def detext_eyes(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def detext_smiles(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def cartonize_image(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def cannize_image(image):
    new_img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


def main():
    st.title('Face Detection App')
    st.text('Build with Streamlit and OpenCV')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader(
            'Upload Image', type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            image = Image.open(image_file)
            st.text('Original Image')
            st.image(image)

            enhance_type = st.sidebar.radio(
                'Enhance Type', ['Original', 'GrayScale', 'Contrast', 'Brightness', 'Blurring'])
            state = st.button('Enhance Image')
            if enhance_type == 'GrayScale':
                if state:
                    new_img = np.array(image.convert('RGB'))
                    img = cv2.cvtColor(new_img, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    st.image(gray)
            elif enhance_type == 'Contrast':
                if state:
                    c_rate = st.sidebar.slider('Contrast', 0.5, 3.5)
                    enhancer = ImageEnhance.Contrast(image)
                    img_output = enhancer.enhance(c_rate)
                    st.image(img_output)
            elif enhance_type == 'Brightness':
                if state:
                    c_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
                    enhancer = ImageEnhance.Brightness(image)
                    img_output = enhancer.enhance(c_rate)
                    st.image(img_output)
            elif enhance_type == 'Blurring':
                if state:
                    new_img = np.array(image.convert('RGB'))
                    blur_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
                    img = cv2.cvtColor(new_img, 1)
                    blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
                    st.image(blur_img)
            elif enhance_type == 'Original':
                if state:
                    st.image(image, width=300)
            else:
                if state:
                    st.image(image, width=300)

        task = ['Face', 'Smile', 'Eye', 'Cannize', 'Cartonize']
        feature_choice = st.sidebar.selectbox('Find Features', task)
        if st.button('Detect Features or Process'):
            if feature_choice == 'Face':
                result_img, result_faces = detect_faces(image)
                st.image(result_img)
                st.success('Found {} Faces'.format(len(result_faces)))
            elif feature_choice == 'Smile':
                result_img = detext_smiles(image)
                st.image(result_img)
            elif feature_choice == 'Eye':
                result_img = detext_eyes(image)
                st.image(result_img)
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(image)
                st.image(result_img)
            elif feature_choice == 'Cannize':
                result_img = cannize_image(image)
                st.image(result_img)

    elif choice == 'About':
        st.subheader('About Face Detection App')
        st.markdown(
            'Built with Streamlit by [Zhecheng Li](https://www.kaggle.com/lizhecheng)')
        st.text('Zhecheng Li')
        st.success('From University of California')


if __name__ == '__main__':
    main()
