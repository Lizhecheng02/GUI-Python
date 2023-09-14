import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.set_page_config(layout="wide")

processor = ViltProcessor.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)
model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)


def get_answer(image, text):
    try:
        img = Image.open(BytesIO(image)).convert("RGB")
        encoding = processor(
            images=img,
            text=text,
            return_tensors="pt"
        )
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer

    except Exception as e:
        return str(e)


st.title("Visual Question Answering")
st.info("Upload an Image and Enter a question to Get an Answer.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        st.image(
            uploaded_file,
            use_column_width=True
        )
    else:
        st.error("Please upload an Image !!!!!!")

with col2:
    question = st.text_input("Question")
    if uploaded_file and question is not None:
        if st.button("Get the answer"):
            image = Image.open(uploaded_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format="JPEG")
            image_bytes = image_byte_array.getvalue()

            answer = get_answer(
                image_bytes,
                question
            )

            st.success(f"Answer: {answer}")
