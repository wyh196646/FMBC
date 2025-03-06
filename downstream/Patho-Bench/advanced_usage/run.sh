#!/usr/bin/env bash

python run.py \
  --experiment_type linprobe \
  --model_name titan \
  --tasks_yaml configs/tasks.yaml \
  --combine_slides_per_patient False \  # This parameter is different for different models. Titan requires this to be False.
  --saveto ../artifacts/example_runs/titan_linprobe \
  --hyperparams_yaml "configs/linprobe/linprobe.yaml" \
  --pooled_dirs_root "../artifacts/pooled_features" \
  --patch_dirs_yaml "configs/patch_embeddings_paths.yaml" \
  --splits_root "../artifacts/splits" \
  --conda_venv pathobench \
  --delay_factor 2                    # Controls how much each pane is delayed by. Pane i will start after (i**delay_factor) seconds