from datasets import load_dataset, concatenate_datasets, DatasetDict
import sys, os

"""
Here we merge six clean manually annotated datasets into a single one having MMLU format. 
We use:

- OpenBookQA (general)
- ARC-Challenge (general)
- ARC-Easy (general)
- TruthfulQA (mix)
- MedQA (medical)
- MathQA (math)

The script will download the datasets locally to your hf cache, process them and save a final dataset to output folder.


Usage: 
    python merge_clean_dataset.py <path_to_output_folder>

"""


def process_allenai(dataset):
    """
    Transform a dataset formatting from allenai-style into MMLU-style
    :param dataset: tuple containing the dataset name and the parameters required from the HF's load_dataset function
    :return: the input dataset formatted in MMLU-style
    """

    def fix_allenai(example, original_dataset='unknown'):
        """
        Tranform a single row from allenai formatting style into MMLU one
        :param example: row
        :param original_dataset: name of the dataset
        :return: the same row formatted in MMLU style
        """
        example["answer"] = ord(example["answer"]) - ord('A') if not example["answer"].isnumeric() else int(
            example["answer"]) - 1

        # we keep this just to be consistent with MMLU format
        example["subject"] = 'n/a'

        # we keep track of the original dataset
        example["original_dataset"] = original_dataset
        return example

    original_name, hf_params = dataset

    print(f'Processing: {original_name}...')
    dataset = load_dataset(**hf_params)
    dataset = dataset.flatten().remove_columns(['choices.label', 'id']).rename_columns(
        {'choices.text': 'choices', 'answerKey': 'answer'})
    if 'question_stem' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'question_stem': 'question'})
    processed_dataset = dataset.map(fix_allenai, fn_kwargs={'original_dataset': original_name})
    return processed_dataset


def process_truthful(dataset):
    """
    Transform a dataset formatting from truthful-style into MMLU-style
    :param dataset: tuple containing the dataset name and the parameters required from the HF's load_dataset function
    :return: the input dataset formatted in MMLU-style
    """

    def fix_truthfulqa(example):
        """
        Tranform a single row from truthful formatting style into MMLU one
        :param example: row
        :return: the same row formatted in MMLU style
        """
        # replace anwer
        example['answer'] = example['answer'].index(1)

        example['subject'] = 'n/a'
        example['original_dataset'] = 'truthful_qa'

        return example

    original_name, hf_params = dataset

    print(f'Processing {original_name}...')

    truthful_qa = load_dataset(**hf_params)

    truthful_qa = truthful_qa.remove_columns(['mc2_targets']).flatten().rename_columns(
        {'mc1_targets.choices': 'choices', 'mc1_targets.labels': 'answer'})

    truthful_qa = truthful_qa.map(fix_truthfulqa)

    # create train val test splits
    truthful_qa_processed = truthful_qa.train_test_split(test_size=0.4, seed=42)
    truthful_qa_processed['test'] = truthful_qa_processed['test'].train_test_split(test_size=0.5, seed=42)

    truthful_qa_processed['validation'] = truthful_qa_processed['test']['train']
    truthful_qa_processed['test'] = truthful_qa_processed['test']['test']

    return truthful_qa_processed


def process_mathqa(dataset):
    """
    Transform a dataset formatting from mathqa-style into MMLU-style
    :param dataset: tuple containing the dataset name and the parameters required from the HF's load_dataset function
    :return: the input dataset formatted in MMLU-style
    """
    def fix_mathqa(example):
        """
        Tranform a single row from mathqa formatting style into MMLU one
        :param example: row
        :param original_dataset: name of the dataset
        :return: the same row formatted in MMLU style
        """
        example['answer'] = ord(example['answer']) - ord('a')
        example['subject'] = 'n/a'
        example['original_dataset'] = 'math_qa'
        idcs = [example['choices'].find(char + ' )') for char in 'abcde']
        choices = [example['choices'][start + 3:end].strip().strip(',') for start, end in
                   zip(idcs, idcs[1:] + [len(example['choices'])])]
        example['choices'] = choices

        return example

    original_name, hf_params = dataset

    print(f'Processing {original_name}...')

    math_qa = load_dataset(**hf_params)
    math_qa = math_qa.remove_columns(['Rationale', 'annotated_formula', 'linear_formula', 'category']).rename_columns(
        {'Problem': 'question', 'options': 'choices', 'correct': 'answer'})

    return math_qa.map(fix_mathqa)


def process_medqa(dataset):
    """
    Transform a dataset formatting from medqa-style into MMLU-style
    :param dataset: tuple containing the dataset name and the parameters required from the HF's load_dataset function
    :return: the input dataset formatted in MMLU-style
    """

    def fix_med_qa(row):
        """
        Tranform a single row from medqa formatting style into MMLU one
        We consider the mc1_targets field as answers, since it contains a single correct answer
        :param example: row
        :param original_dataset: name of the dataset
        :return: the same row formatted in MMLU style
        """

        return {
            'subject': 'n/a',
            'choices': [row['ending0'], row['ending1'], row['ending2'], row['ending3']],
            'original_dataset': 'med_qa'
        }

    original_name, hf_params = dataset

    med_qa = load_dataset(**hf_params)

    med_qa = med_qa.rename_columns({
        'sent1': 'question',
        'label': 'answer'
    })
    med_qa = med_qa.map(fix_med_qa)
    return med_qa.remove_columns(['id', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3'])


def main(output_folder):
    all_clean_datasets = []

    ###########################################################################################################
    # ARC-Easy, ARC-Challenge, OpenbookQA. These have a very similar schema, so we process them together.
    ###########################################################################################################

    allenai_datasets = [
        ('ARC-Easy', {'path': 'allenai/ai2_arc', 'name': 'ARC-Easy'}),
        ('ARC-Challenge', {'path': 'allenai/ai2_arc', 'name': 'ARC-Challenge'}),
        ('OpenbookQA', {'path': 'allenai/openbookqa', 'name': 'main'})
    ]
    for allenai_dataset in allenai_datasets:
        all_clean_datasets.append(process_allenai(allenai_dataset))

    ###########################################################################################################
    # TruthfulQA
    ###########################################################################################################
    truthful_dataset = ('TruthfulQA', {'path': 'truthful_qa',
                                       'data_dir': 'multiple_choice',
                                       'split': 'validation'})

    all_clean_datasets.append(process_truthful(truthful_dataset))

    ###########################################################################################################
    # MathQA
    ###########################################################################################################

    print('Processing MathQA...')
    math_qa_datasets = ('MathQA', {'path': 'math_qa'})

    all_clean_datasets.append(process_mathqa(math_qa_datasets))

    ###########################################################################################################
    # MedQA
    ###########################################################################################################
    med_qa_datasets = ('MedQA', {'path': 'GBaker/MedQA-USMLE-4-options-hf',
                                 'data_files': {"train": "train.json", "test": "test.json", 'validation': "dev.json"}
                                 })
    all_clean_datasets.append(process_medqa(med_qa_datasets))

    ###########################################################################################################
    ### Merge the processed datasets into a single one
    ###########################################################################################################

    print('Merging datasets into final one...')

    # concatenate the train, test and validation split
    data_dict = {'train': None, 'test': None, 'validation': None}

    for split in data_dict.keys():
        split_datasets = [d[split] for d in all_clean_datasets]
        concatenated = concatenate_datasets(split_datasets)
        data_dict[split] = concatenated
        # concatenated.save_to_disk(f'allenai_{split}')

    data_dict = DatasetDict(data_dict)

    # count how many examples for each type
    # double check
    from collections import Counter
    for split in ['train', 'test', 'validation']:
        print(Counter(data_dict[split]['original_dataset']))

    # save to disk
    print('Saving HF dataset to: ', output_folder)
    data_dict.save_to_disk(output_folder)

    for split in ['train', 'test', 'validation']:
        save_to = os.path.join(output_folder, split + '.csv')
        print('Saving CSV dataset to: ', save_to)
        data_dict[split].to_csv(save_to)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("python merge_clean_dataset.py <path_to_output_folder>")
    else:
        main(sys.argv[1])
