import os
import sys

sys.path.append(os.getcwd())
import json

import pandas as pd
import requests


def get_data(data):
    question = data["instance"]["input"]["text"]
    choices = list(data["output_mapping"].values())
    model_answer = data["result"]["completions"][0]["text"]
    return question, choices, model_answer


def main():
    # Create directories
    models_apiversion_mapping = {
        "anthropic_claude-3-opus-20240229": "1.0.0",
        "openai_gpt-4o-2024-05-13": "1.3.0",
        "openai_gpt-4-0613": "1.0.0",
        "google_gemini-1.5-pro-preview-0409": "1.2.0",
        "openai_gpt-4-1106-preview": "1.0.0",
        "meta_llama-3-70b": "1.1.0",
        "writer_palmyra-x-v3": "1.3.0",
        "google_text-unicorn@001": "1.0.0",
        "mistralai_mixtral-8x22b": "1.1.0",
        "google_gemini-1.5-flash-preview-0514": "1.3.0",
    }

    subjects = [
        "college_chemistry",
        "college_mathematics",
        "econometrics",
        "formal_logic",
        "global_facts",
        "high_school_physics",
        "machine_learning",
        "professional_law",
        "public_relations",
        "virology",
    ]

    outputs_dir = "outputs/original_helm"

    # Download files
    url_template = "https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/v{api_version}/mmlu:subject={subject},method=multiple_choice_joint,model={model},eval_split=test,groups=mmlu_{subject}/scenario_state.json"

    for model, api_version in models_apiversion_mapping.items():
        for subject in subjects:
            url = url_template.format(
                api_version=api_version, subject=subject, model=model
            )
            response = requests.get(url)
            if response.status_code == 200:
                model_dir = os.path.join(outputs_dir, model)
                os.makedirs(model_dir, exist_ok=True)
                with open(os.path.join(model_dir, f"{subject}.json"), "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {url}")

    print("Download complete.")

    subject_model_request_states = {}
    for subject in subjects:
        model_request_states = {}
        for model in models_apiversion_mapping.keys():
            with open(os.path.join(outputs_dir, model, f"{subject}.json"), "r") as f:
                data = json.load(f)
                question_list, choices_list, model_answer_list = [], [], []
                for request_state in data["request_states"]:
                    question, choices, model_answer = get_data(request_state)
                    question_list += [question]
                    choices_list += [choices]
                    model_answer_list += [model_answer]
                model_request_states[model] = {
                    "question": question_list,
                    "choices": choices_list,
                    "model_answer": model_answer_list,
                }
        subject_model_request_states[subject] = model_request_states

    combined_df_dir = "outputs/original_helm_combined"
    os.makedirs(combined_df_dir, exist_ok=True)

    # Convert subject_model_request_states to a dataframe, making sure that each row is the same question
    for subject, model_request_states in subject_model_request_states.items():
        subject_model_request_states_df = {}
        for model, request_states in model_request_states.items():
            for i in range(len(request_states["question"])):
                question = request_states["question"][i]
                choices = request_states["choices"][i]
                model_answer = request_states["model_answer"][i]
                if (
                    question + " " + " ".join(choices)
                    not in subject_model_request_states_df
                ):
                    subject_model_request_states_df[
                        question + " " + " ".join(choices)
                    ] = {
                        "question": question,
                        "choices": choices,
                        model: model_answer,
                    }
                else:
                    subject_model_request_states_df[question + " " + " ".join(choices)][
                        model
                    ] = model_answer
        subject_model_request_states_df = pd.DataFrame(
            subject_model_request_states_df.values()
        )
        subject_model_request_states_df.to_csv(
            os.path.join(combined_df_dir, f"{subject}.csv"), index=False
        )


if __name__ == "__main__":
    main()
