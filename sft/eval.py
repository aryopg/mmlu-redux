from datasets import load_dataset, concatenate_datasets
import transformers
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import json

from utils import NestedKeyDataset


def load_model():
    lora_model_id = "lora_lr/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
    lora_model_id = "lora_full/meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
    lora_model_id = (
        "edinburgh-dawg/mmlu-error-detection-3classes-balanced-step4096-llama3"
    )
    lora_model_id = (
        "lora_binary_uniform_2048//meta-llama/Meta-Llama-3-8B-Instruct/unaligned/"
    )

    print(lora_model_id)

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    # tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lora_model_id, padding_side="left")

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id,
    )

    return pipeline


def test1():
    ds_id = "edinburgh-dawg/mmlu-redux"
    ds_id = "edinburgh-dawg/mini-mmlu"
    ds_id = "edinburgh-dawg/labelchaos"
    # DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad options clarity\n2. bad questions clarity\n3. clean\n4. multiple correct answers\n5. no correct answer\n6. wrong groundtruth"
    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad presentation\n2. clean\n3. wrong groundtruth"
    DEFAULT_INSTRUCTION = (
        "# Task:\n"
        "Given a triple consisting of a multiple choice question, its choices, and the corresponding ground truth answer, your task is to classify the triple into 'ok' or 'not ok'.\n\n"
        "# Instructions:\n"
        "1. Question Presentation: Is the question well-presented? Assess clarity, grammar, and sufficiency of information.\n"
        "1.1 If Yes, assess the presentation of the MC options.\n"
        "1.2 If No, classify the issue as 'not ok'.\n"
        "2. MC Options Presentation: Are the MC options well-presented? Check if the options are clear, distinct, and relevant to the question.\n"
        "2.1 If Yes, determine if there is one potentially correct answer.\n"
        "2.2 If No, classify the issue as 'not ok'.\n"
        "3. Answer Evaluation: Is there one, more than one, or no potentially correct answer in the options list?\n"
        "3.1 If one, continue to Ground Truth Answer Evaluation.\n"
        "3.2 If more than one, classify the issue as 'not ok'.\n"
        "3.3 If no correct answer, classify the issue as 'not ok'.\n"
        "4. Ground Truth Answer Evaluation: Is the ground truth answer correct?\n"
        "4.1. If Yes, classify as ok.\n"
        "4.2. If No, classify as 'not ok'.\n"
        "Provide your assessment in JSON format with keys 'Question Presentation', 'MC Options Presentation', 'Answer Evaluation', 'Ground Truth Answer Evaluation', 'Classification'. "
        "The 'classification' is either ok, or not ok. \n\n"
        "FOLLOW THE EXACT EXAMPLE ANSWER FORMAT WITHOUT PROVIDING EXPLANATION"
        "# Example Answer:\n"
        '{"Question Presentation": "ok", "MC Options Presentation": "ok", "Answer Evaluation": "ok", "Ground Truth Answer Evaluation": "ok", "Classification": "ok"}'
    )

    def create_conversation(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION},
            {"role": "user", "content": example["input"] + "\nYour response:"},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"messages": messages}

    CHOICES_DELIMITER = "\n"
    QUESTION_VERBALISER = "{question}\n{choices}\nAnswer: {answer}"

    def verbaliser(question, choices, answer):
        verbalised_choices = CHOICES_DELIMITER.join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        return QUESTION_VERBALISER.format(
            question=question,
            choices=verbalised_choices,
            answer=f"{chr(65+answer)}. {choices[answer]}",
        )

    # subset_lst = "anatomy astronomy business_ethics clinical_knowledge college_chemistry college_computer_science college_mathematics college_medicine college_physics conceptual_physics econometrics electrical_engineering formal_logic global_facts high_school_chemistry high_school_geography high_school_macroeconomics high_school_mathematics high_school_physics high_school_statistics high_school_us_history human_aging logical_fallacies machine_learning miscellaneous philosophy professional_accounting professional_law public_relations virology".split()

    subset_lst = [
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
    subset_lst = ["small"]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                # output_lst += [(entry["error_type"].replace("_", " "))]
                output_lst += [
                    (
                        entry["corruptions"].replace("_", " ")
                        if entry["corruptions"]
                        else "clean"
                    )
                ]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    all_lst = [entry for entry in subset_lst]
    test_ds = concatenate_datasets([subset_id_to_ds[name]["test"] for name in all_lst])

    test_ds = test_ds.map(
        create_conversation,
        remove_columns=test_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )
    pipeline = load_model()
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = []
    for i in tqdm(range(len(test_ds))):
        # print(test_ds[i])
        # print(test_ds[i]["messages"][:-1])

        outputs = pipeline(
            # prompt,
            test_ds[i]["messages"][:-1],
            max_new_tokens=4,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            # stop_sequence="<end_of_turn>"
            eos_token_id=terminators,
            # stop_sequence="</s>"
            # stop_sequence="assistant"
            # stop_sequence=pipeline.tokenizer.eos_token
        )
        # print(sent)
        # breakpoint()
        # print(outputs[0]["generated_text"][-2]["content"])#[len(prompt):])
        # print(outputs[0]["generated_text"][-1]["content"])#[len(prompt):])
        pred = outputs[0]["generated_text"][-1]["content"]  # [len(prompt):])
        # outputs.append({"input": test_ds[i], "pred": pred})
        print(json.dumps({"input": test_ds[i], "pred": pred}))
        # print(outputs[0]["generated_text"])
        # print("="*20)

        # if i > 10:
        #    break


def test():
    ds_id = "edinburgh-dawg/mini-mmlu"
    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me whether the provided answer is correct."

    def create_conversation(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION},
            {"role": "user", "content": example["input"] + "\\nYour response:"},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"messages": messages}

    CHOICES_DELIMITER = "\n"
    QUESTION_VERBALISER = "{question}\n{choices}\nAnswer: {answer}"

    def verbaliser(question, choices, answer):
        verbalised_choices = CHOICES_DELIMITER.join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        return QUESTION_VERBALISER.format(
            question=question,
            choices=verbalised_choices,
            answer=f"{chr(65+answer)}. {choices[answer]}",
        )

    subset_lst = [
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
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                output_lst += [("yes" if "ok" == entry["error_type"].strip() else "no")]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    all_lst = [entry for entry in subset_lst]
    test_ds = concatenate_datasets([subset_id_to_ds[name]["test"] for name in all_lst])

    test_ds = test_ds.map(
        create_conversation,
        remove_columns=test_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )
    pipeline = load_model()
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = []
    for i in tqdm(range(len(test_ds))):
        # print(test_ds[i])
        # print(test_ds[i]["messages"][:-1])

        outputs = pipeline(
            # prompt,
            test_ds[i]["messages"][:-1],
            max_new_tokens=4,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            # stop_sequence="<end_of_turn>"
            eos_token_id=terminators,
            # stop_sequence="</s>"
            # stop_sequence="assistant"
            # stop_sequence=pipeline.tokenizer.eos_token
        )
        # print(sent)
        # breakpoint()
        # print(outputs[0]["generated_text"][-2]["content"])#[len(prompt):])
        # print(outputs[0]["generated_text"][-1]["content"])#[len(prompt):])
        pred = outputs[0]["generated_text"][-1]["content"]  # [len(prompt):])
        # outputs.append({"input": test_ds[i], "pred": pred})
        print(json.dumps({"input": test_ds[i], "pred": pred}))
        # print(outputs[0]["generated_text"])
        # print("="*20)

        # if i > 10:
        #    break


def main():
    ds_id = "edinburgh-dawg/labelchaos"

    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me whether the provided answer is correct."

    def create_conversation(example):
        messages = [
            {"role": "system", "content": DEFAULT_INSTRUCTION},
            {"role": "user", "content": example["input"] + "\\nYour response:"},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"messages": messages}

    CHOICES_DELIMITER = "\n"
    QUESTION_VERBALISER = "{question}\n{choices}\nAnswer: {answer}"

    def verbaliser(question, choices, answer):
        verbalised_choices = CHOICES_DELIMITER.join(
            [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
        )
        return QUESTION_VERBALISER.format(
            question=question,
            choices=verbalised_choices,
            answer=f"{chr(65+answer)}. {choices[answer]}",
        )

    subset_lst = [
        "bad_options_clarity",
        "bad_questions_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                output_lst += [("yes" if "clean" in subset_id else "no")]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    all_lst = [entry for entry in subset_lst]
    test_ds = concatenate_datasets([subset_id_to_ds[name]["test"] for name in all_lst])

    test_ds = test_ds.map(
        create_conversation,
        remove_columns=test_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    pipeline = load_model()
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    batch_size = 8
    for out in pipeline(
        NestedKeyDataset(test_ds, "messages", pipeline.tokenizer),
        batch_size=batch_size,
        max_new_tokens=4,
        do_sample=False,
        eos_token_id=terminators,
    ):
        print(out)
        break

    # outputs = []
    # for i in tqdm(range(len(test_ds))):
    #    #print(test_ds[i])
    #    #print(test_ds[i]["messages"][:-1])

    #    outputs = pipeline(
    #        #prompt,
    #        test_ds[i]["messages"][:-1],
    #        max_new_tokens=4,
    #        do_sample=False,
    #        temperature=0.7,
    #        top_p=0.9,
    #        #stop_sequence="<end_of_turn>"
    #        eos_token_id=terminators,
    #        #stop_sequence="</s>"
    #        #stop_sequence="assistant"
    #        #stop_sequence=pipeline.tokenizer.eos_token
    #    )
    #    #print(sent)
    #    #breakpoint()
    #    #print(outputs[0]["generated_text"][-2]["content"])#[len(prompt):])
    #    #print(outputs[0]["generated_text"][-1]["content"])#[len(prompt):])
    #    pred = outputs[0]["generated_text"][-1]["content"] #[len(prompt):])
    #    #outputs.append({"input": test_ds[i], "pred": pred})
    #    print(json.dumps({"input": test_ds[i], "pred": pred}))
    #    #print(outputs[0]["generated_text"])
    #    #print("="*20)

    #    #if i > 10:
    #    #    break


if __name__ == "__main__":
    # main()
    test1()
