from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import openai
import base64
from PIL import Image
from io import BytesIO
import streamlit as st

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(
    n=1,
    temperature=0.5,
    model="gpt-3.5-turbo-0301",
    use_cache=True
)

menu = st.sidebar.selectbox(
    "Choose an option",
    ["Summarize", "Question"]
)

if menu == "Summarize":
    st.subheader("Summarization of Your Data")
    file_uploader = st.file_uploader("Upload Your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize(
            "filename1.csv",
            summary_method="default",
            textgen_config=textgen_config
        )
        st.write(summary)
        goals = lida.goals(
            summary=summary,
            n=2,
            textgen_config=textgen_config
        )
        for goal in goals:
            st.write(goal)
        i = 0
        library = "seaborn"
        textgen_config = TextGenerationConfig(
            n=1,
            temperature=0.2,
            use_cache=True
        )
        charts = lida.visualize(
            summary=summary,
            goal=goals[i],
            textgen_config=textgen_config,
            library=library
        )
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)

elif menu == "Question":
    st.subheader("Query Your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload Your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename2.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Enter Your Query", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen=llm("openai"))
                textgen_config = TextGenerationConfig(
                    n=1,
                    temperature=0.2,
                    use_cache=True
                )
                summary = lida.summarize(
                    "filename2.csv",
                    summary_method="default",
                    textgen_config=textgen_config
                )
                user_query = text_area
                charts = lida.visualize(
                    summary=summary,
                    goal=user_query,
                    textgen_config=textgen_config
                )
                img_base64_string = charts[0].raster
                img = base64_to_image(img_base64_string)
                st.image(img)
