# ARE WE DONE WITH MMLU?

## MMLU-Redux

Please refer to [the Hugging Face page of MMLU-Redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux).

## Error Detection Evaluation

This evaluation includes scripts for assessing different error detection methods on the MMLU-Redux dataset. The techniques include Zero-Shot, Zero-Shot with Chain of Thought (CoT), Few-Shot, and Few-Shot with CoT.

### Run

#### Installation
1. Clone the repository:
```bash
git clone https://github.com/aryopg/mmlu-redux.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
#### Zero-Shot Evaluation
To evaluate the Zero-Shot technique on the MMLU-Redux dataset, run the following command:

```bash
Python scripts/zero_shot_taxonomy.py
```

To evaluate the Zero-Shot with CoT technique, run:

```bash
Python scripts/zero_shot_cot_taxonomy.py
```

#### Few-Shot Evaluation
To evaluate the Few-Shot technique on the MMLU dataset, run the following command:

```bash
scripts/few_shot_taxonomy.py
```

To evaluate the Few-Shot with CoT technique, run:

```bash
scripts/few_shot_cot_taxonomy.py
```

#### Evaluating Multiple Datasets
Alternatively, We provide a bash script to evaluate multiple MMLU subdatasets using the Chain of Thought (CoT) technique. To run the script, use the following command:

```bash
scripts/bash_scripts/mmlu_subdatasets_cot_taxonomy.sh
```

Make sure to modify the script if needed to specify the desired subdatasets model type.


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
