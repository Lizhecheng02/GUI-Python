import gradio as gr


def BMI(name, height, weight):
    result = weight / ((height / 100.0) ** 2)
    str = f'Dear {name}, your BMI number is {result}'
    evaluation = 'Your health condition performs well ! ! !'
    if result >= 24.0:
        evaluation = 'Try to control your eating habit'
    elif result <= 18.5:
        evaluation = 'Maybe you should eat more'
    return str, evaluation


demo = gr.Interface(
    fn=BMI,
    inputs=[gr.Textbox(lines=1, label="Input your name"),
            gr.Slider(145, 230, label='Height (cm)'),
            gr.Slider(35, 150, label='Weight (kg)')],
    outputs=[gr.Textbox(label="Result"),
             gr.Textbox(label="Hint")],
    theme='huggingface',
)

demo.launch(share=True, auth=('gradio', '123'))
