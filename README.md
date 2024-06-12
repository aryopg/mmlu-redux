# ARE WE DONE WITH MMLU?
This repository contains the evaluation code for the paper [**"Are We Done With MMLU?"**](https://arxiv.org/pdf/2406.04127)

## MMLU-Redux
MMLU-Redux is an enhanced version of the MMLU (Measuring Massive Multitask Language Understanding) dataset, carefully annotated to provide a more accurate and reliable benchmark for evaluating the performance of language models. By carefully annotating and removing erroneous data from the original MMLU dataset, MMLU-Redux offers a refined and challenging testbed for assessing the knowledge and reasoning capabilities of state-of-the-art language models.

## Dataset Overview
MMLU-Redux consists of 30 subdatasets, each containing 100 carefully selected and annotated examples. The annotation process involved identifying and removing data points that were classified as errors in the original MMLU dataset. This refinement process ensures that MMLU-Redux provides a more accurate representation of the language models' true capabilities.

Please refer to [**🤗 MMLU-Redux Dataset**] (https://huggingface.co/datasets/mmlu-redux)for more details.

## Error Detection Evaluation

This evaluation provides a comprehensive set of scripts for assessing error detections on the MMLU-Redux dataset. The methods include Zero-Shot, Zero-Shot with Chain of Thought (CoT), Few-Shot, and Few-Shot with CoT techniques.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/aryopg/mmlu-redux.git
```

2. Navigate to the project directory:
```bash
cd mmlu-redux
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Zero-Shot Evaluation
To evaluate the Zero-Shot technique on the MMLU-Redux dataset, run the following command:

```bash
python scripts/zero_shot_taxonomy.py
```

To evaluate the Zero-Shot with CoT technique, run:

```bash
python scripts/zero_shot_cot_taxonomy.py
```

#### Few-Shot Evaluation
To evaluate the Few-Shot technique on the MMLU-Redux dataset, run the following command:

```bash
python scripts/few_shot_taxonomy.py
```

To evaluate the Few-Shot with CoT technique, run:

```bash
python scripts/few_shot_cot_taxonomy.py
```

#### Evaluating Multiple Datasets
We also provide a convenient bash script to evaluate multiple MMLU-Redux subdatasets using the Chain of Thought (CoT) technique. To run the script, use the following command:

```bash
bash scripts/bash_scripts/mmlu_subdatasets_cot_taxonomy.sh
```

Make sure to modify the script if needed to specify the desired subdatasets and model type.


## LabelChaos

### Introduction

...

### Create an environment file
For interacting with the HF Hub and/or having access to OpenAI models, create an environment file containing the following keys
```bash
- HF_READ_TOKEN
- HF_WRITE_TOKEN
- OPENAI_API_KEY
```
You can create your own file starting from an [example here](.env_example)

### Corruption task

With these instructions you can apply perturbations to any dataset structured as MMLU

First, you should define the parameters required for the perturbation.
You should crate/edit the configuration file at 'project_dir/corruption/conf.yml'.
You can use [this file](conf/corruption/conf_example.yml) as a reference.

```bash
python scripts/main_corrupt_dataset.py --dataset [DATASET_PATH] --name [DATASET_NAME] --output_dir [A_PATH]
```

where
- [DATASET_PATH] is the HF path of the dataset you want to corrupt. By default, it is 'edinburgh-dawg/labelchaos'
- [DATASET_NAME] is the subset of the dataset you want to corrupt. By default, it is 'clean'
- [A_PATH] a local directory where the output files will be stored. By default, it is 'project_dir/outputs/perturbed'
