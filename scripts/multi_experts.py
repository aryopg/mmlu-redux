import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = "Question: {question}\n{choices}\nAnswer: "


def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return [
        {
            "role": "system",
            "content": "Given a question and its choices, determine the most correct answer (A, B, C, or D). ONLY RESPOND WITH ONE LETTER.",
        },
        {
            "role": "user",
            "content": QUESTION_VERBALISER.format(
                question=question, choices=verbalised_choices, answer=choices[answer]
            ),
        },
    ]


def predict(client, model_name, prompt, generation_configs):
    response = client.chat.completions.create(
        model=model_name, messages=prompt, **generation_configs
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""
    return prediction


def main():
    dataset = load_dataset("cais/mmlu", "virology", split="test")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    model_name = "gpt-4-turbo"
    generation_temp00_configs = {
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1,
    }
    generation_temp07_configs = {
        "temperature": 0.7,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1,
    }
    generation_temp1_configs = {
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1,
    }

    pred_df = pd.DataFrame(
        columns=[
            "question",
            "choices",
            "answer",
            "prediction_0",
            "prediction_1",
            "prediction_2",
        ]
    )
    for i in range(len(dataset)):
        verbalised_text = verbaliser(
            dataset[i]["question"], dataset[i]["choices"], dataset[i]["answer"]
        )
        prediction_0 = predict(
            client, model_name, verbalised_text, generation_temp00_configs
        )
        prediction_1 = predict(
            client, model_name, verbalised_text, generation_temp07_configs
        )
        prediction_2 = predict(
            client, model_name, verbalised_text, generation_temp1_configs
        )

        pred_df.loc[i] = [
            dataset[i]["question"],
            dataset[i]["choices"],
            dataset[i]["answer"],
            prediction_0,
            prediction_1,
            prediction_2,
        ]
    pred_df.to_csv(f"{model_name}_mmlu_multi_experts.csv", index=False)


if __name__ == "__main__":
    main()
