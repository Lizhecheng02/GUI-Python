import gradio as gr
import torch
import requests
import cv2
from torchvision import transforms
from numpy import asarray

model = torch.hub.load(
    'pytorch/vision:v0.6.0',
    'resnet50',
    pretrained=True
).eval()
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def predict(inp):
    inp = cv2.resize(asarray(inp), (224, 224))
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
    return confidences


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    theme='huggingface'
)

demo.launch(share=True)
