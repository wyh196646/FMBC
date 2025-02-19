from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            balanced_accuracy_score, accuracy_score, \
                            cohen_kappa_score, mean_absolute_error, \
                            mean_squared_error, f1_score


class MakeMetrics:
    '''
    A class to calculate metrics for multilabel classification and regression tasks.
    
    Arguments:
    ----------
    metric (str): the metric to calculate. Default is 'auroc'. Options are 'auroc', 'auprc', 'bacc', 'acc', 'qwk', 'pearson', and 'spearman'.
    average (str): the averaging strategy. Default is 'micro'.
    label_dict (dict): the label dictionary, mapping from label to index. Default is None. 
    '''
    def __init__(self, metric='auroc', average='micro', label_dict: dict=None):
        self.metric = metric
        self.average = average
        self.label_dict = label_dict
        self.reversed_dict = {v: k for k, v in label_dict.items()}

    def get_metric(self, labels: np.array, probs: np.array):
        '''Return the metric score based on the metric name.'''
        if self.metric == 'auroc':
            return roc_auc_score(labels, probs, average=self.average)
        elif self.metric == 'auprc':
            return average_precision_score(labels, probs, average=self.average)
        elif self.metric == 'bacc':
            return balanced_accuracy_score(labels, probs)
        elif self.metric == 'acc':
            return accuracy_score(labels, probs)
        elif self.metric == 'qwk':
            return cohen_kappa_score(labels, probs, weights='quadratic')
        elif self.metric == 'weighted_f1':
            probs = probs > 0.5
            return f1_score(labels, probs, average='weighted')
        elif self.metric == 'mae':
            return mean_absolute_error(labels, probs)
        elif self.metric == 'mse':
            return mean_squared_error(labels, probs)
        elif self.metric == 'rmse':
            return mean_squared_error(labels, probs, squared=False)
        elif self.metric == 'pearson':
            # Compute Pearson Correlation Coefficient
            evaluation_results={f"pearson_{self.reversed_dict.get(i)}": 
                pearsonr(labels[:, i], probs[:, i])[0] for i in range(labels.shape[1])}
            evaluation_results['average_pearson'] = np.mean(list(evaluation_results.values()))
            return evaluation_results
        elif self.metric == 'spearman':
            evaluation_results={f"spearman_{self.reversed_dict.get(i)}": 
                spearmanr(labels[:, i], probs[:, i])[0] for i in range(labels.shape[1])}
            evaluation_results['average_spearman'] = np.mean(list(evaluation_results.values()))
            return evaluation_results
            #return np.median(spearman_values)
        else:
            raise ValueError('Invalid metric: {}'.format(self.metric))
        
    def process_preds(self, labels: np.array, probs: np.array):
        '''Process the predictions and labels.'''
        if self.metric in ['bacc', 'acc', 'qwk']:
            return np.argmax(labels, axis=1), np.argmax(probs, axis=1)
        else:
            return labels, probs
    
    @property
    def get_metric_name(self):
        '''Return the metric name.'''
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return '{}_{}'.format(self.average, self.metric)
            else:
                label_keys = sorted(self.label_dict.keys(), key=lambda x: self.label_dict[x])
                return ['{}_{}'.format(key, self.metric) for key in label_keys]
        else:
            return self.metric
        
    def __call__(self, labels: np.array, probs: np.array) -> dict:
        '''Calculate the metric based on the given labels and probabilities.
        Args:
            labels (np.array): the ground truth labels.
            probs (np.array): the predicted probabilities.'''
        # process the predictions
        labels, probs = self.process_preds(labels, probs)
        if self.metric in ['auroc', 'auprc']:
            if self.average is not None:
                return {self.get_metric_name: self.get_metric(labels, probs)}
            else:
                score = self.get_metric(labels, probs)
                return {k: v for k, v in zip(self.get_metric_name, score)}
        else:
            return {self.get_metric_name: self.get_metric(labels, probs)}


def calculate_multilabel_metrics(probs: np.array, labels: np.array, label_dict, add_metrics: list=None) -> dict: 
    metrics = ['auroc', 'auprc'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in ['micro', 'macro', None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results


def calculate_multiclass_or_binary_metrics(probs: np.array, labels: np.array, label_dict, add_metrics: list=None) -> dict:
    metrics = ['bacc', 'acc', 'auroc', 'auprc', 'weighted_f1'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in ['macro', None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results

def calculate_regression_metrics(probs: np.array, labels: np.array, label_dict, add_metrics: list=None) -> dict:
    metrics = ['mae', 'mse', 'rmse', 'pearson', 'spearman'] + (add_metrics if add_metrics is not None else [])
    results = {}
    for average in [None]: 
        for metric in metrics: 
            metric_func = MakeMetrics(metric=metric, average=average, label_dict=label_dict)
            results.update(metric_func(labels, probs))
    return results

def calculate_metrics_with_task_cfg(probs: np.array, labels: np.array, task_cfg: dict) -> dict:
    task_setting = task_cfg.get('setting', 'multi_class')
    add_metrics = task_cfg.get('add_metrics', None)

    if task_setting == 'multi_label':
        return calculate_multilabel_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
    elif task_setting == 'regression':
        return calculate_regression_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
    else:
        return calculate_multiclass_or_binary_metrics(probs, labels, task_cfg['label_dict'], add_metrics)
