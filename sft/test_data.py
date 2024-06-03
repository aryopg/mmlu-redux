from datasets import load_dataset, interleave_datasets, concatenate_datasets


def main():
    ds_id = "edinburgh-dawg/labelchaos"

    subset_lst = [
        "bad_options_clarity",
        "bad_questions_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad options clarity\n2. bad questions clarity\n3. clean\n4. multiple correct answers\n5. no correct answer\n6. wrong groundtruth"
    DEFAULT_INSTRUCTION = "Analyze the following multiple-choice question and the corresponding answer carefully, and tell me which category it falls in:\n\n1. bad presentation\n2. clean\n3. wrong groundtruth"

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

    subset_lst = [
        "bad_options_clarity",
        "bad_questions_clarity",
        "clean",
        "multiple_correct_answers",
        "no_correct_answer",
        "wrong_groundtruth",
    ]
    subset_id_to_ds = {k: load_dataset(ds_id, k) for k in subset_lst}

    remapping = {
        "clean": "clean",
        "bad_questions_clarity": "bad presentation",
        "bad_options_clarity": "bad presentation",
        "no_correct_answer": "wrong groundtruth",
        "multiple_correct_answers": "wrong groundtruth",
        "wrong_groundtruth": "wrong groundtruth",
    }

    for subset_id, ds in subset_id_to_ds.items():
        for split_name in ["train", "validation", "test"]:
            input_lst, output_lst = [], []

            for entry in ds[split_name]:
                input_str = verbaliser(
                    entry["question"], entry["choices"], entry["answer"]
                )
                input_lst += [input_str]
                # output_lst += [(subset_id.replace("_", " "))]
                output_lst += [remapping[subset_id]]

            ds[split_name] = ds[split_name].add_column("input", input_lst)
            ds[split_name] = ds[split_name].add_column("output", output_lst)

    ok_train_ds = subset_id_to_ds["clean"]["train"]

    not_ok_lst = [entry for entry in subset_lst if "clean" not in entry]
    not_ok_train_ds = concatenate_datasets(
        [subset_id_to_ds[name]["train"] for name in not_ok_lst]
    )

    ok_train_ds = ok_train_ds.map(
        create_conversation,
        remove_columns=ok_train_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    not_ok_train_ds = not_ok_train_ds.map(
        create_conversation,
        remove_columns=not_ok_train_ds.features,
        batched=False,
        desc="Generating No conversations",
    )

    train_ds = interleave_datasets(
        [ok_train_ds, not_ok_train_ds], probabilities=[0.5, 0.5], seed=42
    )

    ok_dev_ds = subset_id_to_ds["clean"]["validation"]
    not_ok_dev_ds = concatenate_datasets(
        [subset_id_to_ds[name]["validation"] for name in not_ok_lst]
    )

    ok_dev_ds = ok_dev_ds.map(
        create_conversation,
        remove_columns=ok_dev_ds.features,
        batched=False,
        desc="Generating Yes conversations",
    )

    not_ok_dev_ds = not_ok_dev_ds.map(
        create_conversation,
        remove_columns=not_ok_dev_ds.features,
        batched=False,
        desc="Generating No conversations",
    )

    dev_ds = interleave_datasets(
        [ok_dev_ds, not_ok_dev_ds], probabilities=[0.5, 0.5], seed=42
    )

    for i in range(len(dev_ds)):
        print(train_ds[i]["messages"][0]["content"])
        print(train_ds[i]["messages"][1]["content"])
        print(train_ds[i]["messages"][2]["content"])
        print("=" * 10)
        # print(dev_ds[i])
        # print("="*10)
        if i > 20:
            break


if __name__ == "__main__":
    main()
