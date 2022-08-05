import pandas as pd
import numpy as np
from random import sample
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, f1_score, jaccard_score

THRESHOLD = 0.5
SAMPLE_TIME = 100
SAMPLE_RATIO = 0.5

def sampling_dataset(gt, predict):
    metrics = {}
    metrics['roc_auc_score'], metrics['precision_score'], metrics['f1_score'], metrics['jaccard_score'] = [], [], [], []
    metrics['sensitivity_score'], metrics['specificity_score'] = [], []

    for _ in tqdm(range(SAMPLE_TIME)):
        rand_index = sample([i for i in range(len(gt))], int(SAMPLE_RATIO*len(gt)))
        metrics['roc_auc_score'].append(roc_auc_score(gt[rand_index], predict[rand_index]>THRESHOLD))
        metrics['precision_score'].append(precision_score(gt[rand_index], predict[rand_index]>THRESHOLD))
        metrics['f1_score'].append(f1_score(gt[rand_index], predict[rand_index]>THRESHOLD))
        metrics['jaccard_score'].append(jaccard_score(gt[rand_index], predict[rand_index]>THRESHOLD))

        tn, fp, fn, tp = confusion_matrix(gt[rand_index], predict[rand_index]>THRESHOLD).ravel()
        metrics['sensitivity_score'].append(tp/(tp+fn))
        metrics['specificity_score'].append(tn/(tn+fp))
    
    return metrics