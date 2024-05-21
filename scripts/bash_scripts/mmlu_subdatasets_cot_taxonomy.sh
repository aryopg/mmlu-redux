#!/bin/bash -l

set -e
set -u

SUB_DATASETS=("machine_learning" "professional_law" "college_chemistry" "college_mathematics" "econometrics" "formal_logic" "global_facts" "high_school_physics" "public_relations" "virology")

for sub_dataset in "${SUB_DATASETS[@]}"; do
  echo "start ${sub_dataset} gpt-4o evaluation"
  python scripts/zero_shot_cot_taxonomy.py --model_type=gpt4 --config="${sub_dataset}" --test_example_num=2
done