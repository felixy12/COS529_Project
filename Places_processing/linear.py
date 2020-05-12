import numpy as np
from sklearn import svm
from sklearn import linear_model
import pickle
from load_data import ucf_places_features, create_dataset_places_features
import argparse

from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_path')
    parser.add_argument('--split_path')
    
    opt = vars(parser.parse_args())
    
    params = {'batch_size': 64,
             'shuffle': True,
             'num_workers': 1}

    loader_train = create_dataset_places_features(opt['features_path'], opt['split_path'], params, ucf_places_features)
    loader_test = create_dataset_places_features(opt['features_path'], opt['split_path'], params, ucf_places_features, split='test')

    
    X_train = []
    target_train = []
    for x,y in loader_train:
        X_train.append(x.cpu().numpy())
        target_train.append(y.cpu().numpy())
    print(len(X_train)) 
    X_train = np.concatenate(X_train)
    target_train = np.concatenate(target_train)
    
    X_test = []
    target_test = []
    for x,y in loader_test:
        X_test.append(x.cpu().numpy())
        target_test.append(y.cpu().numpy())

    X_test = np.concatenate(X_test)
    target_test = np.concatenate(target_test)
    
    print(X_train.shape, X_test.shape)

    clf = svm.LinearSVC(max_iter=500000) 
    clf.fit(X_train, target_train)

    print(clf.score(X_train, target_train))
    print(clf.score(X_test, target_test))
    
    preds = clf.predict(X_test)

    cm = confusion_matrix(target_test,preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class = cm.diagonal()

    with  open('per_class_acc.pkl', 'wb+') as handle:
        pickle.dump(per_class, handle)
    """
    raw_scores = X_test.dot(clf.coef_.T)+clf.intercept_

    #probs = clf.predict_proba(X_test)
    best_3 = np.argsort(raw_scores, axis=1)[:,-3:]

    count=0
    for i in range(best_3.shape[0]):
        if target_test[i] in best_3[i]:
            count+=1
            
    print(float(count)/best_3.shape[0])
    """
