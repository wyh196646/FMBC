from patho_bench.ExperimentFactory import ExperimentFactory # Make sure you have installed Patho-Bench and this imports correctly
import os

train_source = 'BRACS' 
task_name = 'slidelevel_coarse'

embedding_dir = '/data4/embedding/BRACS'
for model in os.listdir(embedding_dir):
    if 'CHIEF' == model or model == 'Gigapath' or model =='TITAN':
        continue
    else:
        model_name = 'mean-{}'.format(model)
        if os.path.exists(os.path.join('./_test_linprobe',task_name, model_name,'test_metrics_summary.json')):
            print('Already done {}'.format(model_name))
            print(os.path.exists(os.path.join('./_test_linprobe',task_name, model_name,'test_metrics_summary.json')))
        else:
            experiment = ExperimentFactory.linprobe( # This is linear probing, but similar APIs are available for coxnet, protonet, retrieval, and finetune
                                model_name = model_name,
                                train_source = train_source,
                                test_source = None, # Leave as default (None) to automatically use the test split of the training source
                                task_name = task_name,
                                patch_embeddings_dirs = '/data4/embedding/BRACS/{}'.format(model), # Can be list of paths if patch features are split across multiple directories. See NOTE below.
                                pooled_embeddings_root = '/data4/cache/_test_pooled_features',
                                splits_root = './_test_splits', # Splits are downloaded here from HuggingFace. You can also provide your own splits using the path_to_split and path_to_task_config arguments
                                path_to_split ='/home/yuhaowang/project/FMBC/downstream/Patho-Bench/BRACS_COARSE.tsv',
                                path_to_task_config = '/home/yuhaowang/project/FMBC/downstream/Patho-Bench/BRACS_COARSE.yaml',
                                combine_slides_per_patient = False, # Only relevant for patient-level tasks with multiple slides per patient. See NOTE below.
                                cost = 1,
                                balanced = False,
                                saveto = './_test_linprobe/{task_name}/{model_name}'.format(task_name=task_name, model_name=model_name),
                            )
            experiment.train()
            experiment.test()
            result = experiment.report_results(metric = 'macro-ovr-auc')
    # except:
    #     pass