import os
import torch
from openai import OpenAI
import anthropic

INSTRUCTION = (
    "# Task:\n"
    "Given a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to classify the triple based on these error types ['Bad options clarity', 'Multiple correct answers', 'Wrong groundtruth', 'No correct answer', 'ok', 'Bad question clarity'].\n\n"
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
    "4.1. If Yes, classify as ok.\n"
    "4.2. If No, classify as Wrong Groundtruth.\n"
    "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification'. "
    "The 'classification' is either OK, Wrong Groundtruth, No Correct Answer, Multiple Correct Answers, Bad Options Clarity, or Bad Question Clarity.\n\n"
    "FOLLOW THE EXACT EXAMPLE ANSWER FORMAT ALL IN ONE LINE WITHOUT PROVIDING EXPLANATION"
    "# Example Answer:\n"
    "{\"Question Presentation\": \"OK\", \"MC Options Presentation\": \"OK\", \"Answer Evaluation\": \"One\", \"Ground Truth Answer Evaluation\": \"Correct\", \"Classification\": \"OK\"}"
)

def predict_gpt4(client, model_name, prompt, generation_configs):
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

def fewshot_predict_gpt4(client, model_name, messages, generation_configs):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **generation_configs
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""
    return prediction


def predict_llama(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id

    output = model.generate(
        input_ids, 
        attention_mask=attention_mask, 
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens, 
        num_return_sequences=1,
        do_sample = False,
        temperature = 0.0
    )
    prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return prediction

def predict_claude(client, prompt):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        temperature=0.0,
        system=INSTRUCTION,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    prediction = response.content[0].text
    return prediction