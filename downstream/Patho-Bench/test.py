from patho_bench.ExperimentFactory import ExperimentFactory # Make sure you have installed Patho-Bench and this imports correctly
import os


task_name = 'slidelevel_coarse'
model_names={
    'CONCH':'CONCH-Mean',
    'Gigapath_tile':'Gigapath-Mean',
    'UNI':'UNI-Mean',
    'UNI-2':'UNI-2-Mean',
    'Virchow':'Virchow-Mean',
    'CHIEF_tile': 'CHIEF',
    'CONCH':'TITAN',
    'Gigapath_tile':'Gigapath',
    'Virchow':'PRISM',
}
dataset = 'BRACS'
embedding_dir = '/data4/embedding'

excute_task ={
    'BRACS_COARSE':{
        'config': './configs/BRACS_COARSE.yaml',
        'dataset_tsv': './dataset_tsv/BRACS_COARSE.tsv',
        'train_source': 'BRACS',
    }
}

def run_task(embedding_dir,
             model_names,
             config,
             dataset,
             train_source,
             task_name):
  
    for base_model, model_name in model_names.items():
        

        if os.path.exists(os.path.join('./_test_linprobe',task_name, model_name,'test_metrics_summary.json')):
            print('Already done {}'.format(model_name))
            print(os.path.exists(os.path.join('./_test_linprobe',task_name, model_name,'test_metrics_summary.json')))
        else:
            experiment = ExperimentFactory.linprobe( # This is linear probing, but similar APIs are available for coxnet, protonet, retrieval, and finetune
                                model_name = model_name.lower(),
                                train_source = train_source,
                                test_source = None, # Leave as default (None) to automatically use the test split of the training source
                                task_name = 'bracs_coarse',
                                patch_embeddings_dirs = os.path.join(embedding_dir,train_source,base_model),
                                pooled_embeddings_root = '/data4/cache/_test_pooled_features',
                                splits_root = './_test_splits', # Splits are downloaded here from HuggingFace. You can also provide your own splits using the path_to_split and path_to_task_config arguments
                                path_to_split = dataset,
                                path_to_task_config = config,
                                combine_slides_per_patient = False, # Only relevant for patient-level tasks with multiple slides per patient. See NOTE below.
                                cost = 1,
                                balanced = False,
                                saveto = './_test_linprobe/{task_name}/{model_name}'.format(task_name=task_name, model_name=model_name),
                            )
            experiment.train()
            result = experiment.report_results(metric = 'macro-ovr-auc')
            
for task_name,task_config in excute_task.items():
    config = task_config['config']
    dataset = task_config['dataset_tsv']
    train_source = task_config['train_source']
    run_task(embedding_dir, model_names, config, dataset, train_source, task_name)
    

