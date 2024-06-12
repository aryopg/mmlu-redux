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
    "1. Question Presentation: Is the question well-presented? Assess clarity, grammar, and sufficiency of information.\n"
    "1.1 If Yes, assess the presentation of the MC options.\n"
    "1.2 If No, classify the issue as Bad Question Clarity.\n"
    "2. MC Options Presentation: Are the MC options well-presented? Check if the options are clear, distinct, and relevant to the question.\n"
    "2.1 If Yes, determine if there is one potentially correct answer.\n"
    "2.2 If No, classify the issue as Bad Options Clarity.\n"
    "3. Answer Evaluation: Is there one, more than one, or no potentially correct answer in the options list?\n"
    "3.1 If one, continue to Ground Truth Answer Evaluation.\n"
    "3.2 If more than one, classify the issue as Multiple Correct Answer.\n"
    "3.3 If no correct answer, classify the issue as No Correct Answer.\n"
    "4. Ground Truth Answer Evaluation: Is the ground truth answer correct?\n"
    "4.1. If Yes, classify as OK.\n"
    "4.2. If No, classify as Wrong Groundtruth.\n"
    "Provide your assesment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification', 'Answer'. "
    "The 'classification' is either OK, Wrong Groundtruth, No Correct Answer, Multiple Correct Answer, Bad Options Clarity, or Bad Question Clarity. "
    "If you classify the issue as Wrong Groundtruth, provide your own answer to the question with key 'answer', else return empty string in key 'answer'.\n\n"
    "# Example Answer:\n"
    "{'Question Presentation': 'OK', 'MC Options Presentation': 'OK', 'Answer Evaluation': 'One', 'Ground Truth Answer Evaluation': 'Wrong Groundtruth', 'classification': 'Wrong Groundtruth', 'answer': 'C'}"
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
        "max_tokens": 200,
    }

    pred_df = pd.DataFrame(columns=["question", "choices", "answer", "prediction"])
    for i in range(len(dataset)):
        verbalised_text = verbaliser(
            dataset[i]["question"], dataset[i]["choices"], dataset[i]["answer"]
        )
        prediction = predict(client, model_name, verbalised_text, generation_configs)
        pred_df.loc[i] = [
            dataset[i]["question"],
            dataset[i]["choices"],
            dataset[i]["answer"],
            prediction,
        ]
    pred_df.to_csv(f"{model_name}_mmlu_answerability_taxonomy_raw.csv", index=False)


if __name__ == "__main__":
    main()
