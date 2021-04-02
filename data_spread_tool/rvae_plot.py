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
import glob

from sklearn.manifold import TSNE

# run from Spyder
# from data_spread_tool import data_spread_utils as utils
# from data_spread_tool import data_vae_utils as vae_utils
# from data_spread_tool import data_vae_params as vae_params
# from data_spread_tool import data_vae_models

# run from CLI
import data_spread_utils as utils
import data_vae_utils as vae_utils
import data_vae_params as vae_params

# %% load data and model
# pkl_path = '../pkls'
pkl_path = '../../pkls'  # from flask annotation tool
data_pkl_name = f'{pkl_path}/semitone_data_3noi_4noi_5.pkl'
with open(data_pkl_name, 'rb') as f:
    df_data = pickle.load(f)

with np.load(f'{pkl_path}/semitone_34nois_rvae_data.npz') as data:
    data_f0s = data['data_f0s']
    data_nois = data['data_nois']
    data_lengths = data['data_lengths']

n_feats = 5
# model_path = f'{pkl_path}/rvae_models'
# model_path = f'{pkl_path}'
model_path = f'{pkl_path}/rvae_models_gru_new_run'
rnn_model = 'gru'
# n_hidden = 128
n_hidden = 256
n_latent = 8

wrapper_name = glob.glob(
        f'{model_path}/wrapper_{rnn_model}_{n_latent}lat'
        '_*'
        f'_{n_hidden}hid.pkl'
        )
if not isinstance(wrapper_name, list):
    wrapper_name = list(wrapper_name)

# params = vae_params.Params()
# params.rnn_model = rnn_model
# params.hidden_units = [n_hidden]
# params.n_latent = n_latent
# params.n_feats = n_feats
# wrapper = vae_utils.init_model(params)

with open(wrapper_name[0], 'rb') as f:
    wrapper = pickle.load(f)

# %% extract the locations in the (10D) latent space
locs_rvae, __ = vae_utils.get_mus_rvae(
    data_f0s,
    data_nois,
    data_lengths,
    wrapper.model,
    )
locs_rvae = locs_rvae.squeeze(1)

# %% t-SNE down to 2D
# for perp in [5, 10, 30, 50]:
for n_iter in [1000, 3000, 5000]:
    for perp in [10, 30, 50]:
        print(f't-SNE perplexity {perp}')
        # perp = 50
        # for n_iter in [500, 1000, 3000, 5000]:
        print(f't-SNE iter {n_iter}')
        # n_iter = 5000
        # perp = 50
        random_seed = 42
        tsne = TSNE(
            n_components=2, random_state=random_seed,
            perplexity=perp,
            init='pca',
            n_iter=n_iter
            )
        locs_rvae_10d_tsne = tsne.fit_transform(locs_rvae)
        f0s = data_f0s.transpose((1, 0, 2))
        f0s = f0s.reshape(f0s.shape[0], -1)
        # plot
        fig, ax = utils.plot_data_spread(
            locs_rvae_10d_tsne,
            f0s,
            data_lengths,
            n_feats=n_feats,
            method=(
                f'RVAE {rnn_model} {n_hidden}hid {n_latent}lat'
                f' {perp}perp {n_iter}iter'
                ),
            x_scale=0.35,
            y_scale=0.0025,
            save_path='figures',
            )
        # locs['location_VAE-4D'] = locs_vae_4d_tsne.tolist()
