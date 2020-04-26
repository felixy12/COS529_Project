import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from os import listdir, path, mkdir

        
def bootstrap(targets_all, scores_all, weights_all, repeat=1000):        
    max_val = targets_all.squeeze().shape[0]
    avg_prec_weights = np.zeros(repeat)
    avg_prec = np.zeros(repeat)
    print(targets_all[:10], scores_all.shape)
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets = targets_all[rand_index]
        scores = scores_all[rand_index]
        weights = weights_all[rand_index]
        
        avg_prec_weights[i] = average_precision_score(targets, scores, sample_weight=weights) 
        avg_prec[i] = average_precision_score(targets, scores)
    
    
    dictionary_results = {'Weighted_AP': np.median(avg_prec_weights), 'Weighted_AP_std': np.std(avg_prec_weights),
        'AP': np.median(avg_prec), 'AP_std': np.std(avg_prec)}
    return dictionary_results

def make_dir(pathname):
    if not path.isdir(pathname):
        mkdir(pathname)

def get_accuracy(predict_prob, targets, k=5):
    
    corr = 0.0
    for i in range(predict_prob.shape[0]):
        topk = np.argpartition(predict_prob[i], -k)[-k:]
        if targets[i] in topk:
            corr+=1.0

    return corr/predict_prob.shape[0]


def per_class_acc(predict_prob, targets, k=5):
    
    per_class=np.zeros(101)
    total_per_class = np.zeros(101)

    for i in range(predict_prob.shape[0]):
        topk = np.argpartition(predict_prob[i], -k)[-k:]
        if targets[i] in topk:
            per_class[int(targets[i])]+=1.0
        total_per_class[int(targets[i])]+=1.0

    return (per_class/total_per_class)

