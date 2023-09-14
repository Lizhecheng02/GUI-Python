"""
This is just an example of how to inplement gradio with easy machine learning algorithm.
For the dataset, we need to consider the meaning of 'capital gain', because in the dataset they are all 0.
Maybe we should not use it to predict, so this method is not accurate, you can change the code to get more accurate prediction
"""

import random
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
import gradio as gr
from datasets import load_dataset

dataset = load_dataset('scikit-learn/adult-census-income')
dataset

"""
The dataset is unbalanced, in order to make the prediction more accurate,
we need to sample both 7800 rows for positive and negative
"""

X_train = dataset['train'].to_pandas()
X_train_let50k = X_train[X_train['income'] == "<=50K"].sample(7800)
X_train_gt50k = X_train[X_train['income'] == ">50K"].sample(7800)
X_train = pd.concat([X_train_let50k, X_train_gt50k])
X_train = X_train.reset_index(drop=True)
X_train.drop(columns=['fnlwgt'], axis=1, inplace=True)
y_train = X_train.pop('income')
y_train = (y_train == '>50K').astype(int)
print(X_train.head())
print(y_train.value_counts())


categorical_columns = [
    'workclass', 'education', 'marital.status',
    'occupation', 'relationship', 'race', 'sex',
    'native.country'
]
X_train = X_train.astype({col: 'category' for col in categorical_columns})


data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
model = xgb.train(params={'objective': 'binary:logistic'}, dtrain=data)
explainer = shap.TreeExplainer(model)


def predict(*args):
    df = pd.DataFrame([args], columns=X_train.columns)
    df = df.astype({col: 'category' for col in categorical_columns})
    pos_pred = model.predict(xgb.DMatrix(df, enable_categorical=True))
    return {'>50K': float(pos_pred[0]), '<=50K': 1 - float(pos_pred[0])}


def interpret(*args):
    df = pd.DataFrame([args], columns=X_train.columns)
    df = df.astype({col: 'category' for col in categorical_columns})
    shap_values = explainer.shap_values(
        xgb.DMatrix(df, enable_categorical=True)
    )
    scores_desc = list(zip(shap_values[0], X_train.columns))
    scores_desc = sorted(scores_desc)
    fig_m = plt.figure(tight_layout=True)
    plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
    plt.title('Feature Shap Values')
    plt.ylabel('Shap Value')
    plt.xlabel('Feature')
    plt.tight_layout()
    return fig_m


unique_class = sorted(X_train['workclass'].unique())
unique_education = sorted(X_train['education'].unique())
unique_marital_status = sorted(X_train['marital.status'].unique())
unique_relationship = sorted(X_train['relationship'].unique())
unique_occupation = sorted(X_train['occupation'].unique())
unique_race = sorted(X_train['race'].unique())
unique_sex = sorted(X_train['sex'].unique())
unique_country = sorted(X_train['native.country'].unique())

"""
Here, we need to notice that if we use lambda function to initialize gr.Dropdown value.
Unless we refresh the gradio link to enter new input, the output would not change no matter
how we change the input and click the button
"""

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            age = gr.Slider(
                label='Age',
                minimum=17,
                maximum=90,
                step=1,
                randomize=True
            )

            work_class = gr.Dropdown(
                label='Workclass',
                choices=unique_class,
                # value=lambda: random.choice(unique_class)
            )

            education = gr.Dropdown(
                label='Education Level',
                choices=unique_education,
                # value=lambda: random.choice(unique_education)
            )

            years = gr.Slider(
                label='Schooling Years',
                minimum=1,
                maximum=16,
                step=1,
                randomize=True
            )

            marital_status = gr.Dropdown(
                label='Marital Status',
                choices=unique_marital_status,
                # value=lambda: random.choice(unique_marital_status)
            )

            occupation = gr.Dropdown(
                label='Occupation',
                choices=unique_occupation,
                # value=lambda: random.choice(unique_occupation)
            )

            relationship = gr.Dropdown(
                label='Relationship Status',
                choices=unique_relationship,
                # value=lambda: random.choice(unique_relationship)
            )

            race = gr.Dropdown(
                label='Race',
                choices=unique_race,
                # value=lambda: random.choice(unique_race)
            )

            sex = gr.Dropdown(
                label='Sex',
                choices=unique_sex,
                # value=lambda: random.choice(unique_sex)
            )

            capital_gain = gr.Slider(
                label='Capital Gain',
                minimum=0,
                maximum=100000,
                step=100,
                randomize=True
            )

            capital_loss = gr.Slider(
                label='Capital Loss',
                minimum=0,
                maximum=10000,
                step=100,
                randomize=True
            )

            hours_per_week = gr.Slider(
                label='Hours Per Week Worked',
                minimum=1,
                maximum=99,
                step=1
            )

            country = gr.Dropdown(
                label='Native Country',
                choices=unique_country,
                # value=lambda: random.choice(unique_country)
            )
        with gr.Column():
            label = gr.Label()
            plot = gr.Plot()
            with gr.Row():
                predict_btn = gr.Button(value='Predict')
                interpret_btn = gr.Button(value='Interpret')
                predict_btn.click(
                    predict,
                    inputs=[age, work_class, education, years,
                            marital_status, occupation,
                            relationship, race, sex, capital_gain,
                            capital_loss, hours_per_week, country],
                    outputs=[label]
                )
                interpret_btn.click(
                    interpret,
                    inputs=[age, work_class, education, years,
                            marital_status, occupation,
                            relationship, race, sex, capital_gain,
                            capital_loss, hours_per_week, country],
                    outputs=[plot]
                )


demo.launch(share=True, auth=('Admin', '123456'))
