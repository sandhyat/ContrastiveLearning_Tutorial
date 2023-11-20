
# importing packages

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, linear_model, model_selection, metrics, ensemble
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, pairwise,mutual_info_score,mean_squared_error
from scipy import linalg, stats
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import random
from matching.games import StableMarriage
import pingouin as pg
import datetime
# from datetime import datetime
import json, sys, argparse

class JointLoss(torch.nn.Module):
  """
  Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
  When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
  quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                      P . . . P . . .
                                      . P . . . P . .
                                      . . P . . . P .
                                      . . . P . . . P
                                      P . . . P . . .
                                      . P . . . P . .
                                      . . P . . . P .
                                      . . . P . . . P
  """

  def __init__(self, options):
    super(JointLoss, self).__init__()
    # Assign options to self
    self.options = options
    # Batch size
    self.batch_size = options["batch_size"]
    # Temperature to use scale logits
    self.temperature = options["tau"]
    # Device to use: GPU or CPU
    self.device = options["device"]
    # initialize softmax
    self.softmax = torch.nn.Softmax(dim=-1)
    # Mask to use to get negative samples from similarity matrix
    self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
    # Function to generate similarity matrix: Cosine, or Dot product
    self.similarity_fn = self._cosine_simililarity if options["cosine_similarity"] else self._dot_simililarity
    # Loss function
    self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

  def _get_mask_for_neg_samples(self):
    # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
    diagonal = np.eye(2 * self.batch_size)
    # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
    q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
    # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
    q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
    # Generate mask with diagonals of all four quadrants being 1.
    mask = torch.from_numpy((diagonal + q1 + q3))
    # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
    mask = (1 - mask).type(torch.bool)
    # Transfer the mask to the device and return
    return mask.to(self.device)

  @staticmethod
  def _dot_simililarity(x, y):
    # Reshape x: (2N, C) -> (2N, 1, C)
    x = x.unsqueeze(1)
    # Reshape y: (2N, C) -> (1, C, 2N)
    y = y.T.unsqueeze(0)
    # Similarity shape: (2N, 2N)
    similarity = torch.tensordot(x, y, dims=2)
    return similarity

  def _cosine_simililarity(self, x, y):
    similarity = torch.nn.CosineSimilarity(dim=-1)
    # Reshape x: (2N, C) -> (2N, 1, C)
    x = x.unsqueeze(1)
    # Reshape y: (2N, C) -> (1, C, 2N)
    y = y.unsqueeze(0)
    # Similarity shape: (2N, 2N)
    return similarity(x, y)

  def XNegloss(self, representation):
    # breakpoint()
    # Compute similarity matrix
    similarity = self.similarity_fn(representation, representation)
    # Get similarity scores for the positive samples from the diagonal of the first quadrant in 2Nx2N matrix
    try:
      l_pos = torch.diag(similarity, self.batch_size)
    except RuntimeError:
      print("Error encountered. Debug.")
      breakpoint()
    # Get similarity scores for the positive samples from the diagonal of the third quadrant in 2Nx2N matrix
    r_pos = torch.diag(similarity, -self.batch_size)
    # Concatenate all positive samples as a 2nx1 column vector
    positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
    # Get similarity scores for the negative samples (samples outside diagonals in 4 quadrants in 2Nx2N matrix)
    negatives = similarity[self.mask_for_neg_samples].view(2 * self.batch_size, -1)
    # Concatenate positive samples as the first column to negative samples array
    logits = torch.cat((positives, negatives), dim=1)
    # Normalize logits via temperature
    logits /= self.temperature
    # Labels are all zeros since all positive samples are the 0th column in logits array.
    # So we will select positive samples as numerator in NTXentLoss
    labels = torch.zeros(2 * self.batch_size).to(self.device).long()
    # Compute total loss
    closs = self.criterion(logits, labels)
    # # Loss per sample; this is being computed together in the main training loop
    # closs = loss / (2 * self.batch_size)
    # Return contrastive loss
    return closs

  def forward(self, representation):
    """
    Args:
        representation (torch.FloatTensor): representation is the projected latent value and latent is the output of the encoder
    """

    closs = self.XNegloss(representation)

    return closs

def svd(X, n_components=2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]

def my_norm(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def pair_wise_similarity(rep, view, status):  # status is whether it is before or after
    normed_feature = my_norm(rep)
    similarity = normed_feature @ normed_feature.T
    similarity = similarity[~np.eye(similarity.shape[0],dtype=bool)].tolist()

    x_name = '{} cosine pair similarity'.format(view)
    tmp_df = pd.DataFrame({x_name: similarity})
    print(tmp_df[x_name].describe())

    plt.figure(figsize=(6,6))
    sns.histplot(data=tmp_df, x=x_name)
    plt.xlabel('')
    plt.ylabel('')
    plt.figtext(0.6, 0.85, "Avg cosine = " + str(np.round(tmp_df[x_name].describe()['mean'], decimals=3)))
    plt.title("Cosine similarity for " + str(view) + " representation" + str(status) + " training")
    plt.savefig(fig_saving_dir + "/PairwiseCos_sim_Dataset_" + str(dataset_number)+ "X_view" +str(view) + "_tau_" + str(tau)+ "_mapped_features_" +str(mpfeatures)+ str(status)+".pdf")
    plt.savefig(fig_saving_dir + "/PairwiseCos_sim_Dataset_" + str(dataset_number)+ "X_view" +str(view) + "_tau_" + str(tau)+ "_mapped_features_" +str(mpfeatures)+ str(status)+".png")
    plt.close()
    # plt.close('all')
    return

def generate_noisy_xbar(x, noise_type="Zero-out", noise_level=0.1):
  """Generates noisy version of the samples x; Noise types: Zero-out, Gaussian, or Swap noise

  Args:
      x (np.ndarray): Input data to add noise to

  Returns:
      (np.ndarray): Corrupted version of input x

  """
  # Dimensions
  no, dim = x.shape
  # Initialize corruption array
  x_bar = torch.zeros_like(x)

  # Randomly (and column-wise) shuffle data
  if noise_type == "swap_noise":
    for i in range(dim):
      idx = torch.randperm(no)
      x_bar[:, i] = x[idx, i]
  # Elif, overwrite x_bar by adding Gaussian noise to x
  elif noise_type == "gaussian_noise":
    # breakpoint()
    x_bar = x + torch.normal(0, noise_level, size=x.shape, device='cuda')
  else:
    x_bar = x_bar

  return x_bar

def Stable_matching_algorithm(C_X1_train, C_X2_train, index_O_to_R, index_R_to_O,num_mapped_axis):
    # creating the preference dictionaries
    ####### ----------  X1 train ------------- ##########


    true_features_pref_X1_train = {}
    cross_recon_features_pref_X1_train = {}

    for i in range(C_X1_train.shape[0]):
        sorted_index = np.argsort(-C_X1_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X1_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X1_train.shape[1]):
        sorted_index = np.argsort(-C_X1_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X1_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X1_train)
    # print(cross_recon_features_pref_X1_train)

    game_X1_train = StableMarriage.create_from_dictionaries(true_features_pref_X1_train,
                                                            cross_recon_features_pref_X1_train)

    ####### ----------  X2 train ------------- ##########

    true_features_pref_X2_train = {}
    cross_recon_features_pref_X2_train = {}

    for i in range(C_X2_train.shape[0]):
        sorted_index = np.argsort(-C_X2_train[i, :])
        sorted_col_index = ["C" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        true_features_pref_X2_train["R" + str(i + 1)] = sorted_col_index

    for j in range(C_X2_train.shape[1]):
        sorted_index = np.argsort(-C_X2_train[:, j])
        sorted_col_index = ["R" + str(sorted_index[v] + 1) for v in range(len(sorted_index))]
        cross_recon_features_pref_X2_train["C" + str(j + 1)] = sorted_col_index

    # print(true_features_pref_X2_train)
    # print(cross_recon_features_pref_X2_train)

    game_X2_train = StableMarriage.create_from_dictionaries(true_features_pref_X2_train,
                                                            cross_recon_features_pref_X2_train)


    ######   ------------  Final matching -----------   ##########

    print("\n ------- Matching from X1_train  --------- \n")
    matching_x1_train = game_X1_train.solve()
    print(matching_x1_train)

    print("\n ------- Matching from X2_train  --------- \n")
    matching_x2_train = game_X2_train.solve()
    print(matching_x2_train)


    # for comparison to the the initial index that were passed
    x1_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x1_train.values()])
    x2_train_y = np.array([int(str(v).split("C")[1]) + num_mapped_axis for v in matching_x2_train.values()])

    # getting the number of mismatches
    mismatched_x1_train = [i for i, j in zip(index_O_to_R, x1_train_y) if i != j]
    mismatched_x2_train = [i for i, j in zip(index_R_to_O, x2_train_y) if i != j]

    # matching matrices
    matching_x1_train_matrix = np.zeros(C_X1_train.shape)
    matching_x2_train_matrix = np.zeros(C_X2_train.shape)

    for i in range(matching_x1_train_matrix.shape[0]):
        # print(i, x1_train_y[i]-1)
        matching_x1_train_matrix[i,x1_train_y[i]-1-num_mapped_axis]=1


    for i in range(matching_x2_train_matrix.shape[0]):
        # print(i, x2_train_y[i]-1)
        matching_x2_train_matrix[i,x2_train_y[i]-1-num_mapped_axis]=1

    print("Mistakes x1")
    print(mismatched_x1_train)
    print(" Mistakes x2 train")
    print(mismatched_x2_train)

    return mismatched_x1_train, mismatched_x2_train, matching_x1_train_matrix, matching_x2_train_matrix


def NTXentLoss(embeddings_knw_o, embeddings_knw_r, embeddings_unknw_o, embeddings_unknw_r,
                 temperature=0.1):  # embeddings from known features of both databases followed by the unknown features
    # compute the cosine similarity bu first normalizing and then matrix multiplying the known and unknown tensors
    cos_sim_o = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
                          temperature)
    cos_sim_or = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_o),
                                        torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
                           temperature)
    cos_sim_r = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
                                       torch.transpose(torch.nn.functional.normalize(embeddings_unknw_r), 0, 1)),
                          temperature)
    cos_sim_ro = torch.div(torch.matmul(torch.nn.functional.normalize(embeddings_knw_r),
                                        torch.transpose(torch.nn.functional.normalize(embeddings_unknw_o), 0, 1)),
                           temperature)
    # for numerical stability  ## TODO update this logit name
    logits_max_o, _ = torch.max(cos_sim_o, dim=1, keepdim=True)
    logits_o = cos_sim_o - logits_max_o.detach()
    logits_max_or, _ = torch.max(cos_sim_or, dim=1, keepdim=True)
    logits_or = cos_sim_or - logits_max_or.detach()
    logits_max_r, _ = torch.max(cos_sim_r, dim=1, keepdim=True)
    logits_r = cos_sim_r - logits_max_r.detach()
    logits_max_ro, _ = torch.max(cos_sim_ro, dim=1, keepdim=True)
    logits_ro = cos_sim_ro - logits_max_ro.detach()

    breakpoint()
    if True:
      # computing the exp logits
      exp_o = torch.exp(logits_o)
      exp_r = torch.exp(logits_r)
      batch_loss_o = - torch.log(exp_o.diag() / exp_o.sum(dim=0)).sum() - torch.log(
        exp_o.diag() / exp_o.sum(dim=1)).sum()
      batch_loss_r = - torch.log(exp_r.diag() / exp_r.sum(dim=0)).sum() - torch.log(
        exp_r.diag() / exp_r.sum(dim=1)).sum()
      # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
      # since we are computing the rank on the similarity so higher the better
      breakpoint()
      avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_o.cpu().detach().numpy(), axis=1)) / len(cos_sim_o)
      avg_rank_cos_sim_r = np.trace(stats.rankdata(cos_sim_r.cpu().detach().numpy(), axis=1)) / len(cos_sim_r)

    # alternative way of computing the loss where the unknown feature part of the examples from the other database are treated as negative examples
    if False:
      cos_sim_combined = torch.concat(
        [torch.concat([logits_o, logits_or], dim=1), torch.concat([logits_ro, logits_r], dim=1)], dim=0)
      exp_comb = torch.exp(cos_sim_combined)
      batch_loss = - torch.log(exp_comb.diag() / exp_comb.sum(dim=0)).sum() - torch.log(
        exp_comb.diag() / exp_comb.sum(dim=1)).sum()
      # computing the avg rank of the positive examples for checking if the algo is learning the representation closer
      # since we are computing the rank on the similarity so higher the better
      # breakpoint()
      avg_rank_cos_sim_o = np.trace(stats.rankdata(cos_sim_combined.cpu().detach().numpy(), axis=1)) / len(
        cos_sim_combined)
      avg_rank_cos_sim_r = avg_rank_cos_sim_o
      batch_loss_o = batch_loss
      batch_loss_r = batch_loss

    # print("This batch's loss and avg rank ", batch_loss_o.item(), batch_loss_r.item(), avg_rank_cos_sim_o, avg_rank_cos_sim_r)
    return batch_loss_o, batch_loss_r, avg_rank_cos_sim_o, avg_rank_cos_sim_r
def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    # print(c_xy)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()

    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std

    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v

    return data

# function to give initial random weights
def weights_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size, device="cuda") * xavier_stddev, requires_grad=True)

class TabularDataset(Dataset):
    def __init__(self, data, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
          The data frame object for the input data. It must
          contain all the continuous, categorical and the
          output columns to be used.

        cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding
          layers in the model. These columns must be
          label encoded beforehand.

        output_col: string
          The name of the output variable column in the data
          provided.
        """

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        # self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in [output_col]]
        # print(self.cont_cols)

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        # if self.cat_cols:
        #     self.cat_X = data[cat_cols].astype(np.int64).values
        # else:
        #     self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx]]


class AE_CL(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

    self.hidden = kwargs["hidden_units"]
    self.hidden_final = kwargs["hidden_units_final"]
    self.hidden_depth = kwargs["hidden_depth"]
    self.input_size = kwargs["input_shape"]
    self.drop_out_rate = kwargs["drop_out_p"] # initial dropout rate

    self.drop_layer1 = nn.Dropout(p=self.drop_out_rate)  # dropout layer just before the input layer to be applied to both views
    self.hidden_layers = torch.nn.ModuleList()
    ## always have at least 1 layer
    self.hidden_layers.append(nn.Linear(in_features=self.input_size, out_features=self.hidden))
    ## sizes for subsequent layers
    hiddensizes = np.ceil(np.linspace(start=self.hidden, stop=self.hidden_final, num=self.hidden_depth)).astype('int64')
    for thisindex in range(len(hiddensizes) - 1):
      self.hidden_layers.append(nn.Linear(in_features=hiddensizes[thisindex], out_features=hiddensizes[thisindex + 1]))

  def forward(self, data):
    data = self.drop_layer1(data)
    code0 = self.hidden_layers[0](data)
    if (len(self.hidden_layers) > 1):
      for thisindex in range(len(self.hidden_layers) - 1):
        code0 = torch.tanh(self.hidden_layers[thisindex + 1](code0))

    return code0
