import os

from datasets import load_dataset, concatenate_datasets
from src import constants
import argparse

from dotenv import load_dotenv
load_dotenv(constants.ENV_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Hugging Face dataset path to undersample', default=constants.DATASET_PATH)
    parser.add_argument('--dataset_name', help='Hugging Face dataset name to undersample', default='clean')
    parser.add_argument('--seed', help='Random seed', default=42)
    parser.add_argument('--push', help='If true pushes the dataset on HugginFace',
                        default=False,  action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    dataset_path = args.dataset
    dataset_name = args.dataset_name
    output_folder = constants.OUTPUT_DIRECTORY
    sets = ['train', 'test', 'validation']

    dataset = load_dataset(path=dataset_path,
                           name=dataset_name,
                           token=os.getenv('HF_READ_TOKEN')
                           )

    med_qa = dataset.filter(lambda x: x['original_dataset'] == 'med_qa')
    math_qa = dataset.filter(lambda x: x['original_dataset'] == 'math_qa')

    math_qa_subsampled = math_qa.shuffle(seeds=args.seed)

    for s in sets:
        math_qa_subsampled[s] = math_qa_subsampled[s].select(range(len(med_qa[s])))

    dataset = dataset.filter(lambda x: x['original_dataset'] != 'math_qa')
    for s in sets:
        dataset[s] = concatenate_datasets([dataset[s], math_qa_subsampled[s]])

    # save to disk
    print('Saving HF dataset to: ', output_folder)
    dataset.save_to_disk(output_folder)

    if args.push:
        dataset.push_to_hub(repo_id=dataset_path,
                            config_name='clean_subsampled',
                            token=os.getenv('HF_WRITE_TOKEN'))
