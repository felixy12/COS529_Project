import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

def per_class_accuracy(targets, preds):
    
    cm = confusion_matrix(targets,preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm.diagonal()

def per_class_average_precision(targets, scores):
    
    zero_one_enc_targets = np.zeros((targets.shape[0], 101))
    zero_one_enc_targets[np.arange(targets.size),targets] = 1
    return average_precision_score(zero_one_enc_targets, scores, average=None)

