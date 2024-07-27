import torch

INSTRUCTION = (
    "# Task:\n"
    "Given the context (some of which might be irrelevant) and a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to classify the triple's correctness to either 'ok' or 'not ok'.\n\n"
    "# Instructions:\n"
    "For each question, let's review step by step:\n\n"
    "## Step 1: Question Presentation\n"
    "1. Is the question well-presented? Assess the clarity and sufficiency of information reference to all the necessary context to answer the question.\n"
    "   - If Yes, proceed to Step 2.\n"
    "   - If No, classify the issue as 'not ok' and stop further evaluation.\n\n"
    "## Step 2: MC Options Presentation\n"
    "2. Are the multiple choice (MC) options well-presented? Check if the options are complete, clear, distinct, and relevant to the question.\n"
    "   - If Yes, proceed to Step 3.\n"
    "   - If No, classify the issue as 'not ok' and stop further evaluation.\n\n"
    "## Step 3: Answer Evaluation\n"
    "3. Is there one, more than one, or no potentially correct answer in the options list?\n"
    "   - If one, proceed to Step 4.\n"
    "   - If more than one, classify the issue as 'not ok' and stop further evaluation.\n"
    "   - If no correct answer, classify the issue as 'not ok' and stop further evaluation.\n\n"
    "## Step 4: Ground Truth Answer Evaluation\n"
    "4. Is the ground truth answer correct?\n"
    "   - If Yes, classify as 'ok'.\n"
    "   - If No, classify as 'not ok'.\n\n"
    "## Summary\n"
    "Provide the summary of the assessments in JSON format with the keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', and 'Classification'.\n"
    "The 'Classification' is either 'ok', or 'not ok'. \n"
    "ANSWER STRICTLY USING ONLY 'OK' OR 'NOT OK'\n\n"
    "## Example Summary Format\n"
    "FOLLOW THE EXACT EXAMPLE FINAL ANSWER FORMAT BELOW AS A FINAL SUMMARY\n"
    "# Final Answer:\n"
    '{"Question Presentation": "OK", "MC Options Presentation": "OK", "Answer Evaluation": "One", "Ground Truth Answer Evaluation": "Correct", "Classification": "OK"}'
)


def predict_gpt4(client, model_name, prompt, generation_configs):
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
        do_sample=False,
        temperature=0.0,
    )
    prediction = tokenizer.decode(
        output[0, input_ids.shape[1] :], skip_special_tokens=True
    )
    return prediction


def predict_claude(client, prompt):
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=200,
        temperature=0.0,
        system=INSTRUCTION,
        messages=[{"role": "user", "content": prompt}],
    )
    prediction = response.content[0].text
    return prediction


if __name__ == "__main__":
    print(INSTRUCTION)
