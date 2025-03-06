import os
import json
from patho_bench.SplitFactory import SplitFactory
from patho_bench.DatasetFactory import DatasetFactory


class GeneralizabilityExperimentWrapper:
    def __init__(self,
                 base_experiment,
                 model_name: str,
                 task_name: str,
                 train_source: str,
                 test_source: str,
                 test_external_only: bool,
                 patch_embeddings_dirs: list[str],
                 pooled_embeddings_root: str,
                 combine_slides_per_patient: bool,
                 saveto: str,
                 gpu = -1,
                 splits_root: str = None,
                 path_to_external_split: str = None,
                 path_to_task_config: str = None
                 ):
        '''
        Wrapper class for generalizability linprobe experiments.
        Performs training on internal dataset and testing on external dataset.
        
        Args:
            base_experiment (Experiment): Experiment object to be used for training.
            model_name (str): Name of model to use for training.
            task_name (str): Name of task to run.
            train_source (str): Name of source dataset to train on.
            test_source (str): Name of source dataset to test on.
            test_external_only (bool): If True, train on entirety of internal dataset and test on external dataset.
                                       If False, train on training split of internal dataset and test first on internal dataset, then on external dataset.
                                       Defaults to True.
            patch_embeddings_dirs (list[str]): List of directories containing patch embeddings.
            pooled_embeddings_root (str): Root directory for saving pooled embeddings.
            combine_slides_per_patient (bool): Whether to combine patches from multiple slides when pooling. If False, will pool each slide independently.
            saveto (str): Save results to this directory.
            gpu (int): GPU to use for extracting slide embeddings. Defaults to -1 (automatically select best GPU).
            splits_root (str): Root directory for downloading external split from HF. Either splits_root or path_to_external_split must be provided.
            path_to_external_split (str): Local path to external split. Either splits_root or path_to_external_split must be provided.
            path_to_task_config (str): Path to task config file. If None, task config will be loaded from HF.
        '''
        self.exp = base_experiment
        self.model_name = model_name
        self.task_name = task_name
        self.train_source = train_source
        self.test_source = test_source
        self.test_external_only = test_external_only
        self.patch_embeddings_dirs = patch_embeddings_dirs
        self.pooled_embeddings_root = pooled_embeddings_root
        self.combine_slides_per_patient = combine_slides_per_patient
        self.saveto = saveto
        self.gpu = gpu
        self.splits_root = splits_root
        self.path_to_external_split = path_to_external_split
        self.path_to_task_config = path_to_task_config
        # Directory to save external testing results
        self.external_results_dir = self.saveto.replace(f"{os.path.sep}{self.train_source}{os.path.sep}", f"{os.path.sep}{self.train_source}=={self.test_source}{os.path.sep}")
        
        if test_external_only:
            print(f"\033[96mRunning generalizability experiment, training on all of {self.train_source} and testing on all of {test_source}...\033[0m")
        else:
            import warnings; warnings.warn("Testing on external dataset is not yet supported by Agent.collect_results(). You may have to collect the results manually at the end of an Agent sweep.")
            print(f"\033[96mRunning generalizability experiment, training on train split of {self.train_source} and testing on both {self.train_source} and {self.test_source}...\033[0m")
        
    def train(self):
        if self.test_external_only:
            print(f"\033[93mAssigning all samples to a single train fold.\033[0m")
            self.exp.dataset.split.remove_all_folds()
            self.exp.dataset.split.assign_folds(num_folds=1, test_frac=0, val_frac=0, method='monte-carlo')  # Assign all samples to a single train fold
            self.exp.dataset.num_folds = 1
            self.exp.results_dir = self.external_results_dir # Save training artifacts directly to external directory
        self.exp.train()

    def test(self):
        if not self.test_external_only:
            self.exp.test() # Test on internal dataset first
            
        self._load_external_dataset()
        self.exp.test() # Test on external dataset
        
    def _load_external_dataset(self):
        '''
        Loads split and dataset from external group for testing.
        '''
        if self.path_to_external_split:
            test_split, task_info = SplitFactory.from_local(self.path_to_external_split, self.path_to_task_config, self.task_name)
        else:
            test_split, task_info = SplitFactory.from_hf(self.splits_root, self.test_source, self.task_name)            
        test_split.remove_all_folds()
        test_split.assign_folds(num_folds=self.exp.dataset.num_folds, test_frac=1, val_frac=0, method='monte-carlo')  # Assign all samples to test

        # Load external dataset
        self.exp.dataset = DatasetFactory.from_slide_embeddings(split = test_split,
                                                                source = self.test_source,
                                                                task = self.task_name,
                                                                patch_embeddings_dirs = self.patch_embeddings_dirs,
                                                                pooled_embeddings_root = self.pooled_embeddings_root,
                                                                pooling_level = task_info['sample_col'],
                                                                combine_slides_per_patient = self.combine_slides_per_patient,
                                                                model_name = self.model_name,
                                                                gpu = self.gpu)
        
                                                                                          
            
        # Save testing artifacts to external directory
        self.exp.results_dir = self.external_results_dir
        
    def report_results(self, metric: str):
        '''
        Report results of experiment. Calls BaseExperiment.report_results().        
        
        Args:
            metric (str): Metric to report. Must be implemented in self.classification_metrics()
        '''
        return self.exp.report_results(metric)