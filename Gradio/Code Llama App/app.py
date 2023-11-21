from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import os
import io
import gradio as gr  # Using gradio 3.41.2
import time

custom_prompt_template = """
    You are an AI Coding Assitant and your task is to solve coding problems and return code snippets based on given user's query. Below is the user's query.
    Query: {query}
    You just need to return the helpful code.
    Helpful Answer:
"""


def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["query"]
    )
    return prompt


def load_model():
    # Download model from https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGML
    llm = CTransformers(
        model="codellama-7b-instruct.ggmlv3.Q4_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.2,
        repetition_penalty=1.0
    )
    return llm


def chain_pipeline():
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa_chain = LLMChain(
        prompt=qa_prompt,
        llm=llm
    )
    return qa_chain


llm_chain = chain_pipeline()


def bot(query):
    llm_response = llm_chain.run({"query": query})
    return llm_response


with gr.Blocks(title="Code Llama Demo") as demo:
    gr.Markdown("## Code Llama Demo")

    # Chatbot doc https://www.gradio.app/docs/chatbot
    chatbot = gr.Chatbot([], elem_id="chatbot", height=500)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = bot(message)
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()
