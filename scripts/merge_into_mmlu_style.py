from datasets import load_dataset, concatenate_datasets, DatasetDict
import sys
import os

import numpy as np

np.random.seed(42)

MAX_SAMPLES_PER_DATASET = 10000


"""
Here we merge six clean manually annotated datasets into a single one having MMLU format. 
We use:

- OpenBookQA (general)
- ARC-Challenge (general)
- ARC-Easy (general)
- TruthfulQA (mix)
- MedQA (medical)
- MathQA (math)

The script will download the datasets locally to yout hf cache, process them and save a final dataset to output folder.


Usage: 
    python merge_clean_dataset.py <path_to_output_folder>

"""

###########################################################################################################
# Helper functions
###########################################################################################################


def shorten_options(example):
    if len(example["choices"]) > 4:
        if example["answer"] < 4:
            example["choices"] = example["choices"][:4]
        else:
            correct_answer = example["choices"][example["answer"]]
            example["choices"] = example["choices"][:4]
            new_rand_idx = np.random.randint(4)
            example["choices"][new_rand_idx] = correct_answer
            example["answer"] = new_rand_idx
    return example


def undersample(dataset, n_samples):
    """
    Keep only the first n_samples of each split in the dataset.
    """
    for split in dataset.keys():
        if len(dataset[split]) > n_samples:
            # shuffle
            dataset[split] = dataset[split].shuffle(seed=42)
            # select n_samples
            dataset[split] = dataset[split].select(range(n_samples))
        else:
            print(f"Warning: {split} has less than {n_samples} samples.")
    return dataset


def main(output_folder):
    ###########################################################################################################
    # ARC-Easy, ARC-Challenge, OpenbookQA. These ones have a very similar schema so we process them together.
    ###########################################################################################################

    allenai_datasets = {
        "ARC-Easy": ("allenai/ai2_arc", "ARC-Easy"),
        "ARC-Challenge": ("allenai/ai2_arc", "ARC-Challenge"),
        "OpenbookQA": ("allenai/openbookqa", "main"),
    }
    processed_allenai_datasets = []

    def fix_allenai(example, original_dataset="unknown"):
        example["answer"] = (
            ord(example["answer"]) - ord("A")
            if not example["answer"].isnumeric()
            else int(example["answer"]) - 1
        )

        # we keep this just to be consistent with MMLU format
        example["subject"] = "n/a"

        # we keep track of the original dataset
        example["original_dataset"] = original_dataset

        # shorten options
        example = shorten_options(example)
        return example

    for original_name, hf_name in allenai_datasets.items():
        print("Processing:", hf_name)
        dataset = load_dataset(*hf_name)
        dataset = (
            dataset.flatten()
            .remove_columns(["choices.label", "id"])
            .rename_columns({"choices.text": "choices", "answerKey": "answer"})
        )
        if "question_stem" in dataset["train"].column_names:
            dataset = dataset.rename_columns({"question_stem": "question"})

        dataset = dataset.map(
            fix_allenai, fn_kwargs={"original_dataset": original_name}
        )
        dataset = dataset.filter(lambda x: len(x["choices"]) >= 4)
        dataset = undersample(dataset, MAX_SAMPLES_PER_DATASET)
        processed_allenai_datasets.append(dataset)

    ###########################################################################################################
    # TruthfulQA
    ###########################################################################################################

    print("Processing TruthfulQA...")

    truthful_qa = load_dataset(
        path="truthful_qa", data_dir="multiple_choice", split="validation"
    )

    truthful_qa = (
        truthful_qa.remove_columns(["mc2_targets"])
        .flatten()
        .rename_columns(
            {"mc1_targets.choices": "choices", "mc1_targets.labels": "answer"}
        )
    )

    def fix_truthfulqa(example):
        # replace anwer
        example["answer"] = example["answer"].index(1)

        example["subject"] = "n/a"
        example["original_dataset"] = "truthful_qa"

        # shorten options
        example = shorten_options(example)

        return example

    truthful_qa = truthful_qa.map(fix_truthfulqa)
    truthful_qa = truthful_qa.filter(lambda x: len(x["choices"]) >= 4)

    # create train val test splits
    truthful_qa_processed = truthful_qa.train_test_split(test_size=0.4, seed=42)
    truthful_qa_processed["test"] = truthful_qa_processed["test"].train_test_split(
        test_size=0.5, seed=42
    )

    truthful_qa_processed["validation"] = truthful_qa_processed["test"]["train"]
    truthful_qa_processed["test"] = truthful_qa_processed["test"]["test"]

    truthful_qa_processed = undersample(truthful_qa_processed, MAX_SAMPLES_PER_DATASET)

    ###########################################################################################################
    # MathQA
    ###########################################################################################################

    print("Processing MathQA...")

    math_qa = load_dataset("math_qa")
    math_qa = math_qa.remove_columns(
        ["Rationale", "annotated_formula", "linear_formula", "category"]
    ).rename_columns({"Problem": "question", "options": "choices", "correct": "answer"})

    def fix_mathqa(example):
        example["answer"] = ord(example["answer"]) - ord("a")
        example["subject"] = "n/a"
        example["original_dataset"] = "math_qa"
        idcs = [example["choices"].find(char + " )") for char in "abcde"]
        choices = [
            example["choices"][start + 3 : end].strip().strip(",")
            for start, end in zip(idcs, idcs[1:] + [len(example["choices"])])
        ]
        example["choices"] = choices

        # shorten options
        example = shorten_options(example)

        return example

    math_qa_processed = math_qa.map(fix_mathqa)
    math_qa_processed = math_qa_processed.filter(lambda x: len(x["choices"]) >= 4)
    math_qa_processed = undersample(math_qa_processed, MAX_SAMPLES_PER_DATASET)

    ###########################################################################################################
    # MedQA
    ###########################################################################################################
    print("Processing MedQA...")

    med_qa = load_dataset(
        path="GBaker/MedQA-USMLE-4-options-hf",
        data_files={
            "train": "train.json",
            "test": "test.json",
            "validation": "dev.json",
        },
    )

    def fix_med_qa(row):
        """
        With this funtion we transform the data according to the mmlu format.
        We consider the mc1_targets field as answers, since it contains a single correct answer
        """
        return {
            "subject": "n/a",
            "choices": [row["ending0"], row["ending1"], row["ending2"], row["ending3"]],
            "original_dataset": "med_qa",
        }

    med_qa = med_qa.rename_columns({"sent1": "question", "label": "answer"})
    med_qa = med_qa.map(fix_med_qa)
    med_qa_processed = med_qa.remove_columns(
        ["id", "sent2", "ending0", "ending1", "ending2", "ending3"]
    )
    med_qa_processed = med_qa_processed.filter(lambda x: len(x["choices"]) >= 4)
    med_qa_processed = undersample(med_qa_processed, MAX_SAMPLES_PER_DATASET)

    ###########################################################################################################
    ### Merge final dataset
    ###########################################################################################################

    print("Merging datasets into final one...")

    all_clean_datasets = (
        processed_allenai_datasets
        + [math_qa_processed]
        + [truthful_qa_processed]
        + [med_qa_processed]
    )

    # concatenate the train, test and validation split

    data_dict = {"train": None, "test": None, "validation": None}

    for split in data_dict.keys():
        split_datasets = [d[split] for d in all_clean_datasets]
        concatenated = concatenate_datasets(split_datasets)
        data_dict[split] = concatenated
        # concatenated.save_to_disk(f'allenai_{split}')

    data_dict = DatasetDict(data_dict)

    # count how many examples for each type
    # double check
    from collections import Counter

    for split in ["train", "test", "validation"]:
        print(Counter(data_dict[split]["original_dataset"]))

    # save to disk
    print("Saving HF dataset to: ", output_folder)
    data_dict.save_to_disk(output_folder)

    for split in ["train", "test", "validation"]:
        save_to = os.path.join(output_folder, split + ".csv")
        print("Saving CSV dataset to: ", save_to)
        data_dict[split].to_csv(save_to)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python merge_clean_dataset.py <path_to_output_folder>")
    else:
        main(sys.argv[1])
