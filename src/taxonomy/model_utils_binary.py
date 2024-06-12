import torch

INSTRUCTION = (
    "# Task:\n"
    "Given a question, its choices, and the ground truth answer, classify the question as either 'oK' or 'not ok'.\n"
    "- 'ok' means that the question and the choices are understandable, and the ground truth answer is correct.\n"
    "- 'not ok' means that the ground truth answer is incorrect, or the question and the choices are not well presented.\n"
    "Classify with 'ok' or 'not ok' WITHOUT PROVIDING ANY REASONING"
)


def predict_gpt4(client, model_name, prompt, generation_configs, messages=None):
    response = client.chat.completions.create(
        model=model_name,
        messages=(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": prompt},
            ]
            if not messages
            else messages
        ),
        **generation_configs,
    )
    if response and response.choices:
        prediction = response.choices[0].message.content
    else:
        prediction = ""

    return prediction


def predict_llama(model, tokenizer, prompt, max_new_tokens, device):
    input_text = f"{system_prompt}\n\n{prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    pad_token_id = tokenizer.pad_token_id
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        temperature=0.0
    )
    prediction = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return prediction

def predict_claude(client, messages):
    system_message = None
    formatted_messages = []

    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            formatted_messages.append(
                {"role": message["role"], "content": message["content"]}
            )

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=700,
        temperature=0.0,
        system=system_message,
        messages=formatted_messages,
    )

    prediction = response.content[0].text
    return prediction
