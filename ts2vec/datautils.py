import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
import json
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.profiler import profile, record_function, ProfilerActivity

class Med_embedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.v_units = kwargs["v_units"]
        self.v_med_ids = kwargs["v_med_ids"]
        self.e_dim_med_ids = kwargs["e_dim_med_ids"]
        self.e_dim_units = kwargs["e_dim_units"]
        self.p_idx_med_ids = kwargs["p_idx_med_ids"]
        self.p_idx_units = kwargs["p_idx_units"]

        self.embed_layer_med_ids = nn.Embedding(
            num_embeddings=self.v_med_ids + 1,
            embedding_dim=self.e_dim_med_ids,
            padding_idx=self.p_idx_med_ids
        )  # there is NA already in the dictionary that can be used a padding token

        ## NOTE: for the multiply structure, need to be conformable
        if self.e_dim_units is True:
            self.e_dim_units = self.e_dim_med_ids
        else:
            self.e_dim_units = 1
        self.embed_layer_units = nn.Embedding(
            num_embeddings=self.v_units + 1,
            embedding_dim=self.e_dim_units,
            padding_idx=self.p_idx_units
        )

        self.linear = nn.Linear(in_features=self.e_dim_med_ids, out_features=1)

    def forward(self, medication_ids, dose, units, test=0):

        # if test==1:
        #     print("debug")
        #     breakpoint()
        # breakpoint()
        # Get a list of all non-garbage collected objects and their sizes.
        # non_gc_objects_with_size_After = get_non_gc_objects_with_size()
        #
        # # Print the number of non-garbage collected objects and their total size.
        # print("before forward ",len(non_gc_objects_with_size_After), sum([size for obj, size in non_gc_objects_with_size_After]))
        #
        # print("input min max")
        # print(medication_ids.min(), medication_ids.max())
        # print(units.min(), units.max())
        #
        # if units.min()<0 or units.max() > 219:
        #     print("debug")
        #     breakpoint()

        units_embedding = self.embed_layer_units(units.long())
        med_ids_temp_embedding = self.embed_layer_med_ids(medication_ids.long())
        med_combined_embed = torch.mul(torch.mul(units_embedding, dose.unsqueeze(-1)), med_ids_temp_embedding)
        med_combined_embed = torch.sum(med_combined_embed, 2)

        outcome_pred = self.linear(torch.sum(med_combined_embed,1))
        outcome_pred = torch.sigmoid(outcome_pred)


        # # Get a list of all non-garbage collected objects and their sizes.
        # non_gc_objects_with_size_After = get_non_gc_objects_with_size()
        #
        # # Print the number of non-garbage collected objects and their total size.
        # print("after forward ", len(non_gc_objects_with_size_After), sum([size for obj, size in non_gc_objects_with_size_After]))
        # print("-----\n")

        return med_combined_embed, outcome_pred


class customdataset(Dataset):
    def __init__(self, data, outcome, transform = None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------
        data: multidimensional torch tensor
        """
        self.n = data[0].shape[0]
        self.y = outcome
        self.X_1 = data[0]
        self.X_2 = data[1]
        self.X_3 = data[2]

        self.transform = transform

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx],[self.X_1[idx], self.X_2[idx], self.X_3[idx]]]


import gc

def get_object_size(obj):
    """Returns the size of the given object in bytes."""
    import sys
    return sys.getsizeof(obj)

def get_non_gc_objects_with_size():
    """Returns a list of all non-garbage collected objects and their sizes."""
    objects = []
    for obj in gc.get_objects():
        if not gc.is_tracked(obj):
            objects.append((obj, get_object_size(obj)))
    return objects

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    
def load_epic(dataset, outcome):  #dataset is whether it is flowsheets or meds, outcome is the postoperative outcome

    #move this part to somewhere else more universal
    # data_dir = '/input/'
    data_dir = 'datasets/Epic/'

    if dataset ==  'Flowsheets' or dataset ==  'Flowsheets_meds' :
        dense_flowsheet_train = np.load(data_dir + "flowsheet_very_dense_proc_tr.npy")
        dense_flowsheet_test = np.load(data_dir + "flowsheet_very_dense_proc_te.npy")
        dense_flowsheet_valid = np.load(data_dir + "flowsheet_very_dense_proc_val.npy")

        other_flowsheet_train = np.load(data_dir + "other_flow_dense_proc_train.npy")
        other_flowsheet_test = np.load(data_dir + "other_flow_dense_proc_test.npy")
        other_flowsheet_valid = np.load(data_dir + "other_flow_dense_proc_valid.npy")

        train_Xflow = np.concatenate((dense_flowsheet_train, other_flowsheet_train), axis=2)
        test_Xflow = np.concatenate((dense_flowsheet_test, other_flowsheet_test), axis=2)

    if dataset == 'Med_doses' or dataset ==  'Flowsheets_meds' :
        dense_med_doses_train = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_train.npy"))
        dense_med_doses_test = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_test.npy"))
        dense_med_doses_valid = torch.from_numpy(np.load(data_dir + "dense_med_dose_proc_valid.npy"))

        dense_med_id_train = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_train.npy"))
        dense_med_id_test = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_test.npy"))
        dense_med_id_valid = torch.from_numpy(np.load(data_dir + "dense_med_id_proc_valid.npy"))

        dense_med_units_train = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_train.npy"))
        dense_med_units_test = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_test.npy"))
        dense_med_units_valid = torch.from_numpy(np.load(data_dir + "dense_med_units_proc_valid.npy"))

        train_y_temp = torch.from_numpy(np.load(data_dir + "outcome_icu_train.npy"))
        # test_y_temp = np.load(data_dir + "outcome_icu_test.npy")
        valid_y_temp = torch.from_numpy(np.load(data_dir + "outcome_icu_valid.npy"))

        data_tr = [train_y_temp[:10000], dense_med_id_train[:10000], dense_med_doses_train[:10000], dense_med_units_train[:10000]]
        data_val =[valid_y_temp,dense_med_id_valid,dense_med_doses_valid,dense_med_units_valid]

        # train_dataset = customdataset(data=[dense_med_id_train,dense_med_units_train,dense_med_doses_train], outcome=train_y_temp)
        # valid_dataset = customdataset(data=[dense_med_id_valid,dense_med_units_valid,dense_med_doses_valid], outcome=valid_y_temp)
        # test_dataset = customdataset(data=[dense_med_id_test,dense_med_units_test,dense_med_doses_test])

        # Get a list of all non-garbage collected objects and their sizes.
        non_gc_objects_with_size = get_non_gc_objects_with_size()

        # Print the number of non-garbage collected objects and their total size.
        print(len(non_gc_objects_with_size), sum([size for obj, size in non_gc_objects_with_size]))


        # breakpoint()
        batchSize_temp = 8
        # train_loader = DataLoader(train_dataset, batchSize_temp, shuffle=True)
        # valid_loader = DataLoader(valid_dataset, dense_med_doses_valid.shape[0])
        device1 = torch.device('cpu')
        med_net = Med_embedding(
            v_units=218,
            v_med_ids=92,
            e_dim_med_ids=5,
            e_dim_units=False,
            p_idx_med_ids=0,  # putting these 0 because the to dense sets everything not available as 0
            p_idx_units=0,
        ).to(device1)

        # create an optimizer object
        # Adam optimizer
        optimizer = optim.Adam(med_net.parameters(), lr=0.0001, weight_decay=1e-5)

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

        # initializing the loss function
        criterion = nn.BCELoss()

        total_train_loss = []
        total_valid_loss = []
        epochs_temp=20
        for epoch in range(epochs_temp):
            loss_tr = 0
            med_net.train()
            shuffle_index = torch.randperm(n=data_tr[0].shape[0])
            nbatch = data_tr[0].shape[0] // batchSize_temp
            for i in range(nbatch):
                these_index = shuffle_index[range(i * batchSize_temp, (i + 1) * batchSize_temp)]
                local_data = [torch.index_select(x, 0, these_index) for x in data_tr]

                true_y = local_data[0].float().to(device1)
                x_1 = local_data[1].to(device1)  # this is being done because the input has 3 components
                x_2 = local_data[2].to(device1)
                x_3 = local_data[3].to(device1)

                # reset the gradients back to zero as PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                # breakpoint()
                # calling the model
                _, y_pred = med_net(x_1, x_2, x_3)

                # computing the loss
                train_loss = criterion(y_pred.squeeze(-1), torch.log(true_y + 1)).float()
                # print(train_loss)

                if train_loss == 'nan':
                    for j in range(batchSize_temp):
                        print(true_y[j], y_pred[j])
                    print("..")

                train_loss.backward()
                optimizer.step()

                loss_tr += train_loss.item()

            # for i, data in enumerate(train_loader):
            #     true_y = data[0].float().to(device1)
            #     x_1 = data[1][0].to(device1)  # this is being done because the input has 3 components
            #     x_2 = data[1][1].to(device1)
            #     x_3 = data[1][2].to(device1)
            #
            #     # reset the gradients back to zero as PyTorch accumulates gradients on subsequent backward passes
            #     optimizer.zero_grad()
            #     breakpoint()
            #     # calling the model
            #     y_pred, _ = med_net(x_1, x_2, x_3)
            #
            #     # computing the loss
            #     train_loss = criterion(y_pred, torch.log(true_y + 1)).float()
            #     # print(train_loss)
            #
            #     if train_loss == 'nan':
            #         for j in range(64):
            #             print(true_y[j], y_pred[j])
            #         print("..")
            #
            #     train_loss.backward()
            #     optimizer.step()
            #
            #     loss_tr += train_loss.item()
            #
            loss_tr = loss_tr / data_tr[0].shape[0]

            # computing test loss
            loss_val = 0
            med_net.eval()
            nbatch_Val = 1 # just for the ease since the validation set is very small
            for i in range(nbatch_Val):
                true_y = data_val[0].float().to(device1)
                x_1 = data_val[1].to(device1)  # this is being done because the input has 3 components
                x_2 = data_val[2].to(device1)
                x_3 = data_val[3].to(device1)
                # breakpoint()

                _, y_pred = med_net(x_1, x_2, x_3, 1)

                # computing the loss
                valid_loss = criterion(y_pred.squeeze(-1), torch.log(true_y + 1)).float()

                loss_val += valid_loss.item()

            loss_val = loss_val / data_val[0].shape[0]

            scheduler.step(loss_val)

            # display the epoch training and test loss
            print("epoch : {}/{}, training loss = {:.8f}, valid loss = {:.8f}".format(epoch + 1, epochs_temp, loss_tr,
                                                                                      loss_val))

            total_train_loss.append(loss_tr)
            total_valid_loss.append(loss_val)

        # Get a list of all non-garbage collected objects and their sizes.
        non_gc_objects_with_size_After = get_non_gc_objects_with_size()

        # Print the number of non-garbage collected objects and their total size.
        print(len(non_gc_objects_with_size_After), sum([size for obj, size in non_gc_objects_with_size_After]))

        # breakpoint()
        # memory_stats = torch.cuda.memory_stats()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
        # print(f"Available GPU memory: {available_memory / 1024 ** 3:.2f} GB")
        #
        torch.cuda.empty_cache()
        #
        # memory_stats = torch.cuda.memory_stats()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
        # print(f"Available GPU memory: {available_memory / 1024 ** 3:.2f} GB")
        #
        del local_data
        #
        # memory_stats = torch.cuda.memory_stats()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        # available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
        # print(f"Available GPU memory: {available_memory / 1024 ** 3:.2f} GB")

        # breakpoint()
        med_net.eval()
        # nbatch_inf_tr = data_tr[0].shape[0] // batchSize_temp  # inference time computation needed to be done in batches because it was getting difficult to move all the data to gpu and cuda was running out of memory
        # for i in range(nbatch_inf_tr):
        #     true_y = data_tr[0].float().to(device1)
        #     x_1 = data_tr[1].to(device1)  # this is being done because the input has 3 components
        #     x_2 = data_tr[2].to(device1)
        #     x_3 = data_tr[3].to(device1)
        #     temp_emb, _ = med_net(x_1, x_2, x_3)
        #     if i ==0:
        #         med_combined_embed_train = temp_emb
        #     else:
        #         med_combined_embed_train = torch.cat((med_combined_embed_train, temp_emb), 0)  # the ordering in conat matter because later on the index are being selected accordingly
        #
        # breakpoint()

        # with profile(activities=[ProfilerActivity.CUDA],
        #              profile_memory=True, record_shapes=True) as prof:
        #     med_net(dense_med_id_train[:10000].to(device1),dense_med_doses_train[:10000].to(device1), dense_med_units_train[:10000].to(device1))
        #
        # # print(prof.key_averages().table(sort_by="cpu_memory_usage"))
        # # breakpoint()


        med_combined_embed_train, _ = med_net(dense_med_id_train.to(device1),dense_med_doses_train.to(device1), dense_med_units_train.to(device1))
        med_combined_embed_test, _ = med_net(dense_med_id_test.to(device1),dense_med_doses_test.to(device1),dense_med_units_test.to(device1),1)

        # breakpoint()
        # # this need training; currently the parameters are hardcoded here
        # embed_layer_units = nn.Embedding(num_embeddings=218 + 1,embedding_dim=1,padding_idx=0)
        # embed_layer_med_ids = nn.Embedding(num_embeddings=92 + 1,embedding_dim=5,padding_idx=0)  # there is NA already in the dictionary that can be used a padding token
        #
        # units_embedding_train = embed_layer_units(dense_med_units_train.long())
        # units_embedding_test = embed_layer_units(dense_med_units_test.long())
        # med_ids_temp_embedding_train = embed_layer_med_ids(dense_med_id_train.long())
        # med_ids_temp_embedding_test = embed_layer_med_ids(dense_med_id_test.long())
        #
        # med_combined_embed_train = torch.mul(torch.mul(units_embedding_train, dense_med_doses_train.unsqueeze(-1)), med_ids_temp_embedding_train)
        # med_combined_embed_test = torch.mul(torch.mul(units_embedding_test, dense_med_doses_test.unsqueeze(-1)), med_ids_temp_embedding_test)
        #
        # med_combined_embed_train = torch.sum(med_combined_embed_train, 2) # these are still torch tensors
        # med_combined_embed_test = torch.sum(med_combined_embed_test, 2)


        train_X_med = med_combined_embed_train.detach().cpu().numpy()
        test_X_med = med_combined_embed_test.detach().cpu().numpy()

    if dataset ==  'Flowsheets_meds':
        train_X = np.concatenate((train_Xflow, train_X_med), axis=2)
        test_X = np.concatenate((test_Xflow, test_X_med), axis=2)
    elif dataset == 'Flowsheets':
        train_X = train_Xflow
        test_X = test_Xflow
    else:
        train_X = train_X_med
        test_X = test_X_med

    train_idx = pd.read_csv(data_dir + "train_test_id_orlogid_map.csv")
    all_outcomes = pd.read_csv(data_dir + "all_outcomes_with_orlogid.csv")
    train_id_withoutcomes = train_idx.merge(all_outcomes, on=['new_person'], how='left')

    if outcome=='mortality':
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['death_in_30'].values
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['death_in_30'].values
    elif outcome=='aki1':
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] >= 1, 'post_aki_status'] = 1
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] < 1, 'post_aki_status'] = 0
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status'].values
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status'].values
    elif outcome=='aki2':
        train_id_withoutcomes.loc[train_id_withoutcomes[
                            'post_aki_status'] < 2, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] >= 2, 'post_aki_status'] = 1
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status'].values
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status'].values
    elif outcome=='aki3':
        train_id_withoutcomes.loc[train_id_withoutcomes[
                            'post_aki_status'] < 3, 'post_aki_status'] = 0  # the order matters here otherwise everything will bbecome zero :(; there is aone liner too that can be used
        train_id_withoutcomes.loc[train_id_withoutcomes['post_aki_status'] == 3, 'post_aki_status'] = 1
        train_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==1]['post_aki_status'].values
        test_y = train_id_withoutcomes[train_id_withoutcomes['train_id_or_not']==0]['post_aki_status'].values
    elif outcome == 'icu':
        train_y = np.load(data_dir + "outcome_icu_train.npy")
        test_y = np.load(data_dir + "outcome_icu_test.npy")
        valid_y = np.load(data_dir + "outcome_icu_valid.npy")

        preops_raw = pd.read_csv(data_dir + "Raw_preops_used_in_ICU.csv")
        test_index_orig = train_idx[train_idx['train_id_or_not'] == 0]['new_person'].values
        test_index = preops_raw.iloc[test_index_orig][preops_raw.iloc[test_index_orig]['plannedDispo'] != 3]['plannedDispo'].index

        test_y = test_y[test_index]
        test_X = test_X[test_index]

        del preops_raw, test_index

        # preops_test = np.load(data_dir + "preops_proc_test.npy")
        #
        # md_f = open(data_dir + 'preops_metadataicu.json')
        # metadata = json.load(md_f)
        # metadata['column_all_names'].remove('person_integer')
        # plannedDispoIndexNumber = metadata['column_all_names'].index('plannedDispo')
        #
        # plannedDispo_mean = metadata['norm_value_ord']['mean'][metadata['norm_value_ord']['ord_names'].index('plannedDispo')]
        # plannedDispo_std = metadata['norm_value_ord']['std'][metadata['norm_value_ord']['ord_names'].index('plannedDispo')]

        # preops.iloc[test_index]['plannedDispo'].unique().min()

        # breakpoint()

    # train_file = os.path.join(data_dir + 'dense_flowsheets_train_2d.csv')
    # test_file = os.path.join(data_dir + 'dense_flowsheets_test_2d.csv')
    # train_file_labels = os.path.join(data_dir + 'epic_outcome_train.csv')
    # test_file_labels = os.path.join(data_dir + 'epic_outcome_test.csv')
    # train_df_labels = pd.read_csv(train_file_labels, header=None)
    # test_df_labels = pd.read_csv(test_file_labels, header=None)
    #
    # train_df = pd.read_csv(train_file, header=None)
    # test_df = pd.read_csv(test_file, header=None)
    # train_array_2d = np.array(train_df)
    # test_array_2d = np.array(test_df)
    # train_X = train_array_2d.reshape(train_array_2d.shape[0], 511, 15)
    # test_X = test_array_2d.reshape(test_array_2d.shape[0], 511, 15)
    #
    # train_y = np.array(train_df_labels)
    # test_y = np.array(test_df_labels)

    # breakpoint()
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]
        
    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
        
    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)
    
    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
           res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
           res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data
