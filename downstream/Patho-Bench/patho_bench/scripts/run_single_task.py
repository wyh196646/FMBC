import os
import yaml
import argparse
import sys; sys.path.append('../')
from patho_bench.ExperimentFactory import ExperimentFactory
from patho_bench.helpers.SpecialDtypes import SpecialDtypes

"""
##############################################################################################################
Run a hyperparameter sweep for a given experiment type and model.
A dict of list of hyperparameters is specified in a config YAML. Combinations of hyperparameters are generated and the experiment is run for each combination in series.
Only one model and task code can be specified.

NOTE:
    It is recommended to run ../tutorial/run.py instead of this script.
    When you run ../tutorial/run.py, it will use tmux to run one or more tasks with flexible parallelism.
    In contrast, this script only supports a single task.

Usage:
    python run_single_task.py \
    --experiment_type linprobe \
    --model_name threads \
    --task_code bcnb--her2 \
    --combine_slides_per_patient True \
    --saveto ../artifacts/experiments/single_task_example \
    --hyperparams_yaml "configs/linprobe/linprobe.yaml" \
    --pooled_dirs_root "../artifacts/pooled_features" \
    --patch_dirs_yaml "configs/patch_dirs.yaml" \
    --splits_root "../artifacts/splits" \
    --gpu 0
##############################################################################################################
"""
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, help='Type of experiment to run')
    parser.add_argument('--model_name', type=str, help='Name of model to use')
    parser.add_argument('--task_code', type=str, help='Task code in format datasource--task_name or train_datasource==test_datasource--task_name')
    parser.add_argument('--combine_slides_per_patient', type=SpecialDtypes.bool, help='Whether to combine patches from multiple slides when pooling at case_id level. If False, will pool each slide independently. Note that some models e.g. GigaPath require this to be False.')
    parser.add_argument('--saveto', type=str, help='Directory to save the sweep')
    parser.add_argument('--hyperparams_yaml', type=str, help='Path to config YAML specifying hyperparameters to sweep over')
    parser.add_argument('--pooled_dirs_root', type=str, default = '../artifacts/pooled_features', help='Root directory for saving pooled embeddings')
    parser.add_argument('--patch_dirs_yaml', type=str, help='Path to YAML file mapping data sources to patch directories.')
    parser.add_argument('--splits_root', type=str, default = '', help='Root directory for downloading splits from HuggingFace')
    parser.add_argument('--path_to_split', type=SpecialDtypes.none_or_str, default = None, help='Local path to split file')
    parser.add_argument('--path_to_external_split', type=SpecialDtypes.none_or_str, default = None, help='Local path to external split file')
    parser.add_argument('--path_to_task_config', type=SpecialDtypes.none_or_str, default = None, help='Local path to task config file')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use for pooling. If -1, the best available GPU is used.')
    args = parser.parse_args()
    
    # Get hyperparameters to sweep over
    with open(args.hyperparams_yaml, 'r') as f:
        sweep_over = yaml.safe_load(f)
    
    # Get patch directories from YAML file
    with open(args.patch_dirs_yaml, 'r') as f:
        patch_dirs_dict = yaml.safe_load(f)
        
    ExperimentFactory.sweep(experiment_type = args.experiment_type,
                            model_name = args.model_name,
                            task_code = args.task_code,
                            combine_slides_per_patient = args.combine_slides_per_patient,
                            saveto = args.saveto,
                            sweep_over = sweep_over,
                            splits_root = args.splits_root,
                            pooled_dirs_root = args.pooled_dirs_root,
                            patch_dirs_dict = patch_dirs_dict,
                            path_to_split = args.path_to_split,
                            path_to_external_split = args.path_to_external_split,
                            path_to_task_config = args.path_to_task_config,
                            gpu = args.gpu)