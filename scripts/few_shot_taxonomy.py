import argparse
import json
import logging
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
    "Given a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to determine whether the ground truth answer is correct or not.\n\n"
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
    "4.1. If Yes, classify as Correct.\n"
    "4.2. If No, classify as Wrong Groundtruth.\n"
    "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification', 'Answer'. "
    "The 'classification' is either Correct, Wrong Groundtruth, No Correct Answer, Multiple Correct Answer, Bad Options Clarity, or Bad Question Clarity. "
    "If you classify the issue as Wrong Groundtruth, provide your own answer to the question with key 'answer', else return empty string in key 'answer'.\n\n"
    "# Example Answer:\n"
    "{'Question Presentation': 'OK', 'MC Options Presentation': 'OK', 'Answer Evaluation': 'One', 'Ground Truth Answer Evaluation': 'Wrong Groundtruth', 'classification': 'Wrong Groundtruth', 'answer': 'C'}"
)

CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = "Question: {question}\nChoices:\n{choices}\nGround Truth Answer: {answer}\nYour Response: "

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Correct",
            "Classification": "Correct",
            "Answer": "",
        },
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
        "answer": 1,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Correct",
            "Classification": "Correct",
            "Answer": "",
        },
    },
    {
        "question": "What is the largest ocean on Earth?",
        "choices": ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Pacific Ocean"],
        "answer": 1,  # Incorrect ground truth
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "OK",
            "Answer Evaluation": "One",
            "Ground Truth Answer Evaluation": "Wrong Groundtruth",
            "Classification": "Wrong Groundtruth",
            "Answer": "D",
        },
    },
    {
        "question": "Which of the following is a fruit?",
        "choices": ["Carrot", "Potato", "Tomato", "Cucumber"],
        "answer": 2,
        "response": {
            "Question Presentation": "OK",
            "MC Options Presentation": "Bad Options Clarity",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "Bad Options Clarity",
            "Answer": "",
        },
    },
    {
        "question": "What is the meaning of life?",
        "choices": ["42", "Happiness", "Success", "Love"],
        "answer": 0,
        "response": {
            "Question Presentation": "Bad Question Clarity",
            "MC Options Presentation": "N/A",
            "Answer Evaluation": "N/A",
            "Ground Truth Answer Evaluation": "N/A",
            "Classification": "Bad Question Clarity",
            "Answer": "",
        },
    },
]


def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return QUESTION_VERBALISER.format(
        question=question, choices=verbalised_choices, answer=choices[answer]
    )


def few_shot_prompt(examples, test_question, test_choices, test_answer):
    example_prompts = []
    for example in examples:
        example_prompt = verbaliser(
            example["question"], example["choices"], example["answer"]
        )
        example_prompts.append(f"{example_prompt}\n{json.dumps(example['response'])}")

    # Add the test question and choices to the prompt
    test_prompt = verbaliser(test_question, test_choices, test_answer)
    example_prompts.append(test_prompt)

    return "\n".join(example_prompts)


def predict_gpt3(client, model_name, prompt, generation_configs):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        **generation_configs,
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""

    return prediction


def compute_metrics(pred_df):
    exact_match = (pred_df["predicted_error_type"] == pred_df["error_type"]).mean()
    return {"exact_match": exact_match}


def main(args):
    dataset = load_dataset("edinburgh-dawg/mini-mmlu", args.config, split="test")

    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    gpt3_model_name = "gpt-3.5-turbo"
    gpt3_generation_configs = {
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 200,
    }

    pred_df = pd.DataFrame(
        columns=[
            "question",
            "choices",
            "answer",
            "error_type",
            "model_answer",
            "predicted_error_type",
            "predicted_answer",
        ]
    )

    for i in range(len(dataset)):
        question = dataset[i]["question"]
        choices = eval(dataset[i]["choices"])
        answer = dataset[i]["answer"]

        verbalised_text = few_shot_prompt(FEW_SHOT_EXAMPLES, question, choices, answer)

        prediction = predict_gpt3(
            openai_client, gpt3_model_name, verbalised_text, gpt3_generation_configs
        )

        try:
            model_answer = prediction.split("Your Response: ")[-1].strip()
            if model_answer.startswith("{") and model_answer.endswith("}"):
                prediction_json = json.loads(model_answer)
                predicted_error_type = prediction_json["Classification"]
                predicted_answer = prediction_json.get("Answer", "")
            else:
                model_answer = prediction
                predicted_error_type = "Invalid Prediction"
                predicted_answer = ""
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            model_answer = prediction
            predicted_error_type = "Invalid Prediction"
            predicted_answer = ""
            print(f"Error parsing prediction for instance {i}: {str(e)}")
            print(f"Model answer: {model_answer}")

        pred_df.loc[i] = [
            question,
            choices,
            answer,
            dataset[i]["error_type"],
            model_answer,
            predicted_error_type,
            predicted_answer,
        ]

    metrics = compute_metrics(pred_df)
    print(f"Metrics for {args.config}:")
    print(metrics)

    pred_df.to_csv(f"mmlu_groundtruth_correctnessss_{args.config}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration of the mini-mmlu dataset to use",
    )
    args = parser.parse_args()
    main(args)
