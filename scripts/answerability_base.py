import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

INSTRUCTION = (
    "# Task:\n"
    "Given a pair consisting of a multiple choice question and its corresponding ground truth answer, your task is to determine whether the ground truth answer is correct or not.\n\n"
    "# Instructions:\n"
    "- Read the data provided along with its ground truth answer.\n"
    "- Evaluate the correctness of the provided answer based on your understanding of the data.\n"
    "- If you believe the answer accurately answers the question, output 'Correct'.\n"
    "- If you believe the answer does not accurately answer the question, output 'Incorrect'.\n"
    "- If you believe the question-answer pair lacks clarity in terms of its presentation, output 'Unclear'.\n"
    "- If you believe the question-answer pair is clear, but you require outside information to answer it, output 'Expert'.\n"
    "- Provide your response clearly and concisely.\n\n"
)
CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = (
    "Question: {question}\n{choices}\nGround Truth Answer: {answer}\nYour Response: "
)


def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return [
        {"role": "system", "content": INSTRUCTION},
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

    print(prediction)
    return prediction


def main():
    dataset = load_dataset("cais/mmlu", "virology", split="test")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    model_name = "gpt-3.5-turbo"
    generation_configs = {
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 32,
    }

    pred_df = pd.DataFrame(columns=["question", "choices", "answer", "prediction"])
    for i in range(len(dataset)):
        verbalised_text = verbaliser(
            dataset[i]["question"], dataset[i]["choices"], dataset[i]["answer"]
        )
        print(verbalised_text)
        pred_df.loc[i] = [
            dataset[i]["question"],
            dataset[i]["choices"],
            dataset[i]["answer"],
            predict(client, model_name, verbalised_text, generation_configs),
        ]
    pred_df.to_csv(f"{model_name}_mmlu_answerability_base.csv", index=False)


if __name__ == "__main__":
    main()
