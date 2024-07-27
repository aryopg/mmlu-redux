#!/bin/bash -l

set -e
set -u

# SUB_DATASETS=("machine_learning" "professional_law" "college_chemistry" "college_mathematics" "econometrics" "formal_logic" "global_facts" "high_school_physics" "public_relations" "virology")
SUB_DATASETS=("global_facts" "virology" "college_mathematics" "econometrics" "formal_logic" "high_school_physics" "public_relations")

for sub_dataset in "${SUB_DATASETS[@]}"; do
  echo "start ${sub_dataset} gpt-4o evaluation"
  # python scripts/zero_shot_cot_taxonomy.py --model_type=llama --config="${sub_dataset}"
  python scripts/zero_shot_taxonomy_binary.py --model_type=llama --config="${sub_dataset}"
done