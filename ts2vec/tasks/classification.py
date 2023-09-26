import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd

def eval_classification(model, train_data, train_labels, test_data, test_labels, includePreops, outcome, eval_protocol='xgbt'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # obtaining the encoding of the preops over here after training it at a different place.


    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'xgbt':
        fit_clf = eval_protocols.fit_xgbt
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    breakpoint()

    # data_dir = '/input/'
    data_dir = 'datasets/Epic/'

    if includePreops:
        preops_train = np.load(data_dir + "preops_proc_train.npy")
        preops_test = np.load(data_dir + "preops_proc_test.npy")
        preops_valid = np.load(data_dir + "preops_proc_test.npy")
        #
        # train_X = np.load(data_dir + "preops_proc_train.npy")
        # test_X = np.load(data_dir + "preops_proc_test.npy")
        # preops_valid = np.load(data_dir + "preops_proc_test.npy")

        cbow_train = np.load(data_dir + "cbow_proc_train.npy")
        cbow_test = np.load(data_dir + "cbow_proc_test.npy")

        # train_X_cbow = np.load(data_dir + "cbow_proc_train.npy")
        # test_X_cbow = np.load(data_dir + "cbow_proc_test.npy")

        if outcome=='icu': # evaluation only on the non preplanned ICU cases
            preops_raw = pd.read_csv(data_dir + "Raw_preops_used_in_ICU.csv")
            train_idx = pd.read_csv(data_dir + "train_test_id_orlogid_map.csv")
            test_index_orig = train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values
            test_index = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3][
                'plannedDispo'].index

            # breakpoint()

            preops_test = preops_test[test_index]
            cbow_test = cbow_test[test_index]

            # test_X = test_X[test_index]
            # test_X_cbow = test_X_cbow[test_index]

            del preops_raw, test_index, train_idx


        # is the scaling needed again as the preops have been processes already?
        scaler = StandardScaler()
        scaler.fit(preops_train)
        train_X = scaler.transform(preops_train)
        test_X = scaler.transform(preops_test)

        scaler_cbow = StandardScaler()
        scaler_cbow.fit(cbow_train)
        train_X_cbow = scaler_cbow.transform(cbow_train)
        test_X_cbow = scaler_cbow.transform(cbow_test)

        train_repr = np.concatenate((train_repr, train_X, train_X_cbow), axis=1)
        test_repr = np.concatenate((test_repr, test_X, test_X_cbow), axis=1)



    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)


    if (eval_protocol == 'linear') or (eval_protocol == 'xgbt'):
        y_score = clf.predict_proba(test_repr)
        auprc = average_precision_score(test_labels, y_score[:,1])
        auroc = roc_auc_score(test_labels, y_score[:,1])
    else:
        y_score = clf.decision_function(test_repr)
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
        auprc = average_precision_score(test_labels_onehot, y_score)
        auroc = roc_auc_score(test_labels_onehot, y_score)
    
    return y_score, { 'acc': acc, 'auprc': auprc, 'auroc': auroc }
