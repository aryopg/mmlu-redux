# ARE WE DONE WITH MMLU?

## Error Detection Evaluation

### Run

```bash
...
```

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
