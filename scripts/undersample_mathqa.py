from datasets import load_dataset, concatenate_datasets

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random seed', default=42)
    args = parser.parse_args()

    dataset_path = 'edinburgh-dawg/labelchaos'
    output_folder = '../outputs/subsampling'
    sets = ['train', 'test', 'validation']

    dataset = load_dataset(dataset_path, 'clean')

    df = dataset['train'].to_pandas()

    med_qa = dataset.filter(lambda x: x['original_dataset'] == 'med_qa')
    math_qa = dataset.filter(lambda x: x['original_dataset'] == 'math_qa')

    math_qa_subsampled = math_qa.shuffle(seeds=args['seed'])

    for s in sets:
        math_qa_subsampled[s] = math_qa_subsampled[s].select(range(len(med_qa[s])))

    dataset = dataset.filter(lambda x: x['original_dataset'] != 'math_qa')
    for s in sets:
        dataset[s] = concatenate_datasets([dataset[s], math_qa_subsampled[s]])

    # save to disk
    print('Saving HF dataset to: ', output_folder)
    dataset.save_to_disk(output_folder)
