from datasets import load_dataset, concatenate_datasets, DatasetDict
from collections import Counter
import random
import os

from src.constants import ENV_PATH
from dotenv import load_dotenv
load_dotenv(ENV_PATH)

PERTURBATIONS = [
    'clean',
    'bad_options_clarity',
    'bad_questions_clarity',
    'multiple_correct_answers',
    'no_correct_answer',
    'wrong_groundtruth'
]

# count samples in each source dataset
pert = PERTURBATIONS[0]
d = load_dataset('edinburgh-dawg/labelchaos', pert)
datasets_count = Counter(d['test']['original_dataset'])
smaller_dataset = min(datasets_count, key=lambda x: datasets_count[x])
selected_samples = datasets_count[smaller_dataset]

result = None
for p in PERTURBATIONS:
    print(f'Downloading {p}')
    d = load_dataset('edinburgh-dawg/labelchaos', p)
    test = d['test']
    for dataset in datasets_count:
        print(f'Selecting dataset {dataset}')
        dataset_samples = test.filter(lambda x: x['original_dataset'] == dataset)

        if dataset is not smaller_dataset:
            print(f'Under sampling {dataset}')
            under_samples = random.sample(range(len(dataset_samples)), selected_samples)
            selection = dataset_samples.select(under_samples)
        else:
            selection = dataset_samples
        print(f'merging {len(selection)} samples')
        result = selection if result is None else concatenate_datasets([result, selection])
        print(f'Actual dataset has {len(result)} samples')

result = DatasetDict({'test': result})
# push
result.push_to_hub(repo_id='edinburgh-dawg/labelchaos',
                    config_name='small',
                    token=os.getenv('HF_WRITE_TOKEN'))
