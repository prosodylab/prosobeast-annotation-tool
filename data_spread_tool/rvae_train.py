#!/usr/bin/env python3
"""LSTM model training based on 3noi and 4noi data.

Created on Nov  6 2020

@author: Branislav Gerazov
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os
from tqdm import tqdm
import sys

# run from Spyder
# from data_spread_tool import data_spread_utils as utils
# from data_spread_tool import data_vae_utils as vae_utils
# from data_spread_tool import data_vae_params as vae_params
# run from CLI
import data_spread_utils as utils
import data_vae_utils as vae_utils
import data_vae_params as vae_params

# %% load semitone data frame
# pkl_path = '../pkls'
pkl_path = 'pkls'
data_pkl_name = f'{pkl_path}/semitone_data_3noi_4noi_5.pkl'
# data_pkl_name = f'{pkl_path}/semitone_data_3noi_5.pkl'
with open(data_pkl_name, 'rb') as f:
    df_data = pickle.load(f)

if len(sys.argv) > 1:
    n_latent = int(sys.argv[1])
    print(f'n_latent {n_latent}')
else:
    n_latent = 10

if len(sys.argv) > 2:
    n_hidden = int(sys.argv[2])
    print(f'n_hidden {n_hidden}')
else:
    n_hidden = 128

if len(sys.argv) > 3:
    rnn_model = sys.argv[3]
    print(f'rnn_model {rnn_model}')
else:
    rnn_model = 'gru'

# %% Experiment 1
# f0 data shape [n_samples x noi x n_feats]
n_feats = 5

# find max sequence length
max_len = 0
for file in tqdm(df_data.file.unique()):
    mask_file = df_data.file == file
    n_nois = df_data.loc[
        mask_file, 'pitch0': f'pitch{n_feats - 1}'
        ].values.shape[0]
    if n_nois > max_len:
        max_len = n_nois


# %% get data
def get_noi_ramps(seq_len):
    nois = []
    for i in range(seq_len):
        nois.append([i, seq_len - i - 1])
    return np.array(nois, dtype='float32')


# gather data and pas with NaNs accordingly
data_f0s = None
data_nois = None
data_lengths = np.array([], dtype='int32')
for file in tqdm(df_data.file.unique()):
    mask_file = df_data.file == file
    f0s = df_data.loc[
        mask_file, 'pitch0': f'pitch{n_feats - 1}'
        ].values.astype('float32')
    seq_len = np.int32(f0s.shape[0])
    data_lengths = np.append(data_lengths, seq_len)
    nois = get_noi_ramps(seq_len)

    # pad
    if seq_len < max_len:
        f0s = np.pad(
            f0s, ((0, max_len - seq_len), (0, 0)), constant_values=np.nan
            )
        nois = np.pad(
            nois, ((0, max_len - seq_len), (0, 0)), constant_values=np.nan
            )

    # aggregate
    f0s = np.expand_dims(f0s, 1)
    nois = np.expand_dims(nois, 1)
    if data_f0s is None:
        data_f0s = f0s
        data_nois = nois
    else:
        data_f0s = np.concatenate([data_f0s, f0s], axis=1)
        data_nois = np.concatenate([data_nois, nois], axis=1)

# merge f0s and nois for encoder
# data_nois_f0s = np.concatenate([data_nois, data_f0s], axis=2)

# %% sort sequences in descending order - good for packing the tensor
i_sort = np.argsort(data_lengths)[::-1]
data_f0s = data_f0s[:, i_sort, :]
data_nois = data_nois[:, i_sort, :]
data_lengths = data_lengths[i_sort]

np.savez_compressed(
    f'{pkl_path}/semitone_34nois_rvae_data.npz',
    data_f0s=data_f0s,
    data_nois=data_nois,
    data_lengths=data_lengths,
    )

# %% train model
params = vae_params.Params()
# params.n_latent = n_latent
# params.hidden_units = [n_hidden]
# params.model_type = 'rnn'
# params.rnn_model = rnn_model
# params.n_feats = n_feats
# params.shuffle = False
params.model_type = 'rnn'
params.shuffle = False
params.rnn_model = 'gru'
params.hidden_units = [256]
params.n_latent = 8
params.n_feats = n_feats
params.learn_rate = 0.0001
params.patience = 1000
params.batch_size = 512
print(params.learn_rate)
print(params.batch_size)
wrapper = vae_utils.init_model(
    params
    )
wrapper, best_error, *__ = vae_utils.train_model(
    data_f0s, wrapper, data_nois, data_lengths, params=params
    )

# %% save wrapper
wrapper_name = (
        f'wrapper_{params.rnn_model}_{params.n_latent}lat_{best_error:.2f}'
        f'_{params.hidden_units[0]}hid'
        # '_inp'  # latent as input
        )
with open(f'{pkl_path}/{wrapper_name}.pkl', 'wb') as f:
    pickle.dump(wrapper, f, -1)

# %% load wrapper
with open(f'{pkl_path}/{wrapper_name}.pkl', 'rb') as f:
    wrapper = pickle.load(f)

# %%once trained extract the locations in the latent space
# locs_rvae, __ = vae_utils.get_mus_rvae(
#     data_f0s,
#     data_nois,
#     data_lengths,
#     wrapper.model,
#     )
# __ = utils.plot_data_spread(
#     locs_vae_2d, n_feats, method='VAE 2D latent space',
#     x_scale=0.15,
#     y_scale=0.015,
#     )
# locs['location_VAE-2D'] = locs_vae_2d.tolist()
