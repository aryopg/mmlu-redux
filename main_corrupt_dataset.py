import argparse
import yaml

global seed
global client
global llm
global call_llm

from src.corruptions.corruption_pipeline import Corruptor

import os

DEFAULT_CONFIGURATION = './conf/perturbation_conf.yml'

DEFAULT_OUTPUT_DIRECTORY = './corrupted_dataset'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument('--dataset', help='Hugging Face dataset name to corrupt', default='alessiodevoto/purelabel')
    parser.add_argument('--output_dir', help='Directory to save corrupted dataset', default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument('--seed', help='Random seed', default=42)

    with open(DEFAULT_CONFIGURATION, 'r') as file:
        config = yaml.safe_load(file)

    args = parser.parse_args()

    corruptor = Corruptor(probabilities=config['probabilities'],
                          random_seed=args.seed,
                          llm=config['llm'],
                          hf_token=config['hugginface_token']
                          )

    corruptor.corrupt_dataset(dataset_path=args.dataset,
                              output_dir=args.output_dir,
                              test_flag=config.get('is_a_test', False))

