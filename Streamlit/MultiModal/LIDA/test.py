from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import openai
import base64
from PIL import Image
from io import BytesIO

load_dotenv()


def base64_to_image(base64_string):
    byte_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(byte_data))


def save_image(base64_str, save_path):
    img = base64_to_image(base64_str)
    img.save(save_path)
    print(f"Image saved at {save_path}")


openai.api_key = os.getenv("OPENAI_API_KEY")

lida = Manager(text_gen=llm("openai"))

textgen_config = TextGenerationConfig(
    n=1,
    temperature=0.5,
    model="gpt-3.5-turbo-0301",
    use_cache=True
)
summary = lida.summarize(
    "2019.csv",
    summary_method="default",
    textgen_config=textgen_config
)
print(summary)

goals = lida.goals(
    summary=summary,
    n=2,
    textgen_config=textgen_config
)

for goal in goals:
    print(goal)

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
image_base64 = charts[0].raster

save_image(image_base64, "filename2.png")
