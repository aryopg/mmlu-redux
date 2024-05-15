import logging
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from anthropic import Claude
from transformers import LlamaTokenizerFast
from transformers import LlamaForCausalLM
from huggingface_hub import login

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
    "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification'. "
    "The 'classification' is either Correct, Wrong Groundtruth, No Correct Answer, Multiple Correct Answer, Bad Options Clarity, or Bad Question Clarity.\n\n"
    "# Example Answer:\n"
    "{'Question Presentation': 'OK', 'MC Options Presentation': 'OK', 'Answer Evaluation': 'One', 'Ground Truth Answer Evaluation': 'Correct', 'classification': 'Correct'}"
)
CHOICES_DELIMITER = "\n"
QUESTION_VERBALISER = (
    "Question: {question}\nChoices:\n{choices}\nGround Truth Answer: {answer}\nYour Response: "
)


def verbaliser(question, choices, answer):
    verbalised_choices = CHOICES_DELIMITER.join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    return QUESTION_VERBALISER.format(
        question=question, choices=verbalised_choices, answer=choices[answer]
    )


def predict_gpt3(client, model_name, prompt, generation_configs):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": prompt}],
        **generation_configs
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""

    return prediction


def predict_llama(model, tokenizer, prompt, max_length):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    return prediction


def predict_claude(client, prompt):
    response = client.completions.create(
        model="claude-3",
        prompt=prompt,
        max_tokens_to_sample=200,
        stop_sequences=["\n\nHuman:"]
    )
    prediction = response.completion
    return prediction



def main():
    dataset = load_dataset("cais/mmlu", "virology", split="test")

    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    
    # Set up Hugging Face authorization token
    login("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")
    llm_path = "meta-llama/Meta-Llama-3-8B"
    llama_tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
    llama_model = LlamaForCausalLM.from_pretrained(llm_path)
    
    claude_client = Claude(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    gpt3_model_name = "gpt-3.5-turbo"
    gpt3_generation_configs = {
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 200,
    }
    
    llama_max_length = 200

    pred_df = pd.DataFrame(columns=["question", "choices", "answer", "gpt3_prediction", "llama_prediction", "claude_prediction"])
    for i in range(len(dataset)):
        verbalised_text = verbaliser(
            dataset[i]["question"], dataset[i]["choices"], dataset[i]["answer"]
        )
        
        gpt3_prediction = predict_gpt3(openai_client, gpt3_model_name, verbalised_text, gpt3_generation_configs)
        llama_prediction = predict_llama(llama_model, llama_tokenizer, INSTRUCTION + "\n\n" + verbalised_text, llama_max_length)
        claude_prediction = predict_claude(claude_client, INSTRUCTION + "\n\n" + verbalised_text)
        
        pred_df.loc[i] = [
            dataset[i]["question"],
            dataset[i]["choices"],
            dataset[i]["answer"],
            gpt3_prediction,
            llama_prediction,
            claude_prediction,
        ]
    pred_df.to_csv("mmlu_groundtruth_correctness_raw.csv", index=False)


if __name__ == "__main__":
    main()
