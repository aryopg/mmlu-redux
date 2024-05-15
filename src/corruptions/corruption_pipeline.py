from datasets import load_dataset
import numpy
from numpy.random import choice
import argparse

global seed
global client
global llm
global call_llm

import os
import src.corruptions.corruptors as corruptors


class Corruptor:
    CORRUPTIONS = ['wrong_groundtruth',
                   'no_correct_answer',
                   'multiple_correct_answers',
                   'bad_options_clarity',
                   'bad_questions_clarity']

    TEST_NUM_SAMPLES = 100

    def __init__(self, probabilities, random_seed, llm=None, hf_token=None):
        self._corruption_probabilities = None
        self.corruption_probabilities = probabilities

        self.random_seed = random_seed

        self.llm = self.set_llm(llm, random_seed)
        corruptors.llm = self.llm

        self.wrong_groundtruth = corruptors.wrong_grountruth
        self.no_correct_answer = corruptors.no_correct_answer
        self.multiple_correct_answers = corruptors.multiple_correct_answers
        self.bad_options_clarity = corruptors.bad_options_clarity
        self.bad_question_clarity = corruptors.bad_question_clarity

        self.hf_token = hf_token

    @property
    def corruption_probabilities(self):
        return self._corruption_probabilities

    @corruption_probabilities.setter
    def corruption_probabilities(self, probs):
        assert isinstance(probs, dict), 'Perturbation probabilities should be formatted as a dictionary'
        # if a corruption probability value is missing set it to 0
        self._corruption_probabilities = {corr: probs.get(corr, 0) for corr in self.CORRUPTIONS}
        total_prob = sum(self.corruption_probabilities.values())

        if total_prob == 0:
            raise ValueError(
                'No corruption probabilities provided. Please specify at least one corruption probability.')

        if total_prob > 1:
            raise ValueError('Corruption probabilities sum up to a probability greater than 1. '
                             'Please check the probabilities values in the configuration file.')

    def set_llm(self, llm, random_seed):

        llm = dict(llm)
        if llm:
            assert isinstance(llm, dict), 'LLM configuration should be formatted as a dictionary'
            assert 'model' in llm, 'If you want to use an llm you have to define a model in the llm configuration'
            assert 'type' in llm, 'If you want to use an llm you have to define an llm type in the configuration file'

            if llm['type'] == 'http':
                from huggingface_hub import InferenceClient
                client = InferenceClient(llm['model'])

                def hf_inference(messages):
                    return client.chat_completion(messages, seed=random_seed).choices[0].message.content

                llm['call_llm'] = hf_inference

            elif llm['type'] == 'gpt':
                if not 'openai_key' in llm:
                    raise ValueError('You need to provide an OpenAI key to use an OpenAI model.')
                from openai import OpenAI
                client = OpenAI(api_key=llm['openai_key'])

                def openai_inference(messages):
                    return client.chat.completions.create(model=llm['model'], messages=messages).choices[
                        0].message.content

                llm['llm_call'] = openai_inference

        return llm

    def corrupt(self, example):

        """
        Corrupts the given example using a randomly chosen corruption function.

        Parameters:
        example: The example to be corrupted.

        Returns:
            The corrupted example. Modifications depend on the specific corruption function chosen.
        """

        # pick a function to corrupt the example
        # notice that you can simply set the probabilities to 0 to avoid applying a specific corruption
        probabilities = [self.corruption_probabilities[corr] for corr in self.CORRUPTIONS]
        corruption_type = choice(self.CORRUPTIONS, replace=False, p=probabilities)
        corruption_function = self.__getattribute__(corruption_type)

        # apply corruption
        example = corruption_function(example)

        return example

    def corrupt_dataset(self, dataset_path, output_dir, test_flag=False):

        test_dir = './test'

        numpy.random.seed(self.random_seed)

        # load clean dataset
        print('Loading ***clean*** data from', dataset_path)
        dataset_to_corrupt = load_dataset(dataset_path, 'clean', hf_token=self.hf_token)

        if test_flag:
            print('Corruption set in test mode')
            dataset_to_corrupt = dataset_to_corrupt['train'].select(range(self.TEST_NUM_SAMPLES))
            output_path = os.path.join(test_dir, 'clean_test.csv')
            print('Saving uncorrupted test dataset to ', output_path)
            dataset_to_corrupt.to_csv(output_path)

        corrupted_dataset = dataset_to_corrupt.map(self.corrupt)

        if test_flag:
            # in test scenario saves the result in csv format to facilitate user inspection
            output_path = os.path.join(test_dir, 'corruption_test.csv')
            print('Saving corrupted test dataset to ', output_path)
            corrupted_dataset.to_csv(os.path.join(test_dir, 'corruption_test.csv'))

        print('Saving corrupted dataset to ', output_dir)
        corrupted_dataset.save_to_disk(output_dir)

# load the dataset

# apply corruption strategy to every single row

# store the results
