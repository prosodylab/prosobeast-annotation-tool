#!/usr/bin/env python3
"""
Data spread calculation.

The code loads a CSV file of pitch contours and outputs a CSV file of their
locations in a 2D projection space based on their similarity.

The tool offers the following ways to calculate the spread:
    1. For fixed size contours:
        - PCA
        - t-SNE
        - 2D DNN based VAE latent space
        - 4D DNN based VAE latent space reduced to 2D with t-SNE
    2. For contours of fixed and varying size:
        - 10D GRU based VAE latent space reduced to 2D with t-SNE

The input should contain a single column of pitch data formatted as strings
with no header :
    [-2.50, -2.81, -3.12, ... -3.21]
    [-1.23, -0.81, -1.12, ... -2.35]
    ...

@author:
    Branislav Gerazov Mar 2020
"""
import numpy as np
import pandas as pd

# methods
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# utility functions - run from its folder
import data_spread_utils as utils
try:
    import data_vae_utils as vae_utils
    import data_vae_params as vae_params
except ImportError:
    print('No PyTorch installed!')

# %% init parameters
data_path = '../dataset_sample/'
# new input reads original data and saves locs in additional columns
data_name = data_path + 'prosobeast_3nois.csv'  # input
# data_name = data_path + 'prosobeast_all_nois.csv'  # input

labels_name = data_path + 'prosobeast_labels.csv'
colors = utils.load_label_colors(labels_name)
save_name = data_path + 'prosobeast_locs.csv'

n_feats = 5

plot = True
# choose method(s)
# # for fixed size data:
do_pca = False
do_tsne = True
do_vae2d = False
do_vae4d = False

# for fixed and varying size data:
do_rvae10d = False

random_seed = 42  # for reproducible results from t-SNE and VAEs

# load data
data, f0s, lens, labels = utils.load_data(data_name)
n_samples = f0s.shape[0]
if isinstance(lens, int):  # same length data
    same_length = True
else:
    same_length = False

# init locs df
locs = pd.DataFrame()

# %% PCA
if do_pca:
    if not same_length:
        raise ValueError('F0 data has to have the same length!')
    pca = PCA(n_components=2, whiten=False, copy=True)
    locs_pca = pca.fit_transform(f0s)
    if plot:
        __ = utils.plot_data_spread(
            locs_pca, f0s,
            method='PCA',
            x_scale=15,
            y_scale=0.06,
            labels=labels,
            colors=colors,
            )
    locs['location_PCA'] = locs_pca.tolist()

# %% tSNE analysis
if do_tsne:
    if not same_length:
        raise ValueError('F0 data has to have the same length!')
    tsne = TSNE(
        n_components=2,
        random_state=random_seed,
        perplexity=30,
        init='pca',
        n_iter=5000
        )
    if plot:
        locs_tsne = tsne.fit_transform(f0s)
        __ = utils.plot_data_spread(
            locs_tsne, f0s,
            method='t-SNE',
            x_scale=9,
            y_scale=0.04,
            labels=labels,
            colors=colors,
            )
    locs['location_t-SNE'] = locs_tsne.tolist()

# %% DNN based 2D VAE latent space
if do_vae2d:
    if not same_length:
        raise ValueError('F0 data has to have the same length!')
    params = vae_params.Params()
    params.n_latent = 2
    params.hidden_units = [128] * 2  # works well for our data
    params.n_feats = f0s.shape[1]
    wrapper = vae_utils.init_model(params)
    wrapper = vae_utils.train_model(f0s, wrapper, params=params)[0]
    # once trained extract the locations in the latent space
    locs_vae_2d, __ = vae_utils.get_mus(f0s, wrapper.model)
    if plot:
        __ = utils.plot_data_spread(
            locs_vae_2d, f0s,
            method='VAE 2D latent space',
            x_scale=0.65,
            y_scale=0.003,
            labels=labels,
            colors=colors,
            )
    locs['location_VAE-2D'] = locs_vae_2d.tolist()

# %% DNN based 4D VAE latent space
if do_vae4d:
    if not same_length:
        raise ValueError('F0 data has to have the same length!')
    params = vae_params.Params()
    params.n_latent = 4
    params.hidden_units = [128] * 2  # works well for our data
    params.n_feats = f0s.shape[1]
    wrapper = vae_utils.init_model(params)
    wrapper = vae_utils.train_model(f0s, wrapper, params=params)[0]
    # once trained extract the locations in the latent space
    locs_vae_4d, __ = vae_utils.get_mus(f0s, wrapper.model)
    # % t-SNE down to 2D
    tsne = TSNE(
        n_components=2, random_state=random_seed,
        perplexity=30,
        init='pca',
        n_iter=5000
        )
    locs_vae_4d_tsne = tsne.fit_transform(locs_vae_4d)
    if plot:
        __ = utils.plot_data_spread(
            locs_vae_4d, f0s,
            method='VAE 4D latent space',
            x_scale=0.8,
            y_scale=0.006,
            labels=labels,
            colors=colors,
            )
    locs['location_VAE-4D'] = locs_vae_4d_tsne.tolist()

# %% GRU based 10D RVAE latent space
if do_rvae10d:
    # %%% reshape data
    # f0 is n_samples x n_f0s (= n_nois * n_samples)
    # should be n_nois x n_samples x n_samples
    if same_length:
        max_len = lens
        n_nois = int(lens / n_feats)
        rvae_lens = np.repeat(n_nois, n_samples)
    else:
        n_nois = int(lens.max() / n_feats)
        rvae_lens = np.array(lens / n_feats, dtype='int')

    rvae_f0s = f0s.reshape(n_samples, n_nois, n_feats)
    rvae_f0s = rvae_f0s.transpose((1, 0, 2))

    # generate ramps
    rvae_nois = [vae_utils.get_noi_ramps(x) for x in rvae_lens]
    rvae_nois = [
        np.pad(x, ((0, n_nois - x.shape[0]), (0, 0)), constant_values=np.nan)
        for x in rvae_nois
        ]
    rvae_nois = np.array(rvae_nois)
    rvae_nois = rvae_nois.transpose((1, 0, 2))

    # sort contours in descending length
    # for packing the tensor
    i_sort = np.argsort(rvae_lens)[::-1]
    rvae_f0s = rvae_f0s[:, i_sort, :]
    rvae_nois = rvae_nois[:, i_sort, :]
    rvae_lens = rvae_lens[i_sort]

    # init model
    params = vae_params.Params()
    # RVAE params that work well for our data
    params.model_type = 'rnn'
    params.shuffle = False
    params.rnn_model = 'gru'
    params.hidden_units = [512]
    params.n_latent = 10
    params.n_feats = n_feats
    params.learn_rate = 0.001
    params.patience = 500
    params.batch_size = 512
    wrapper = vae_utils.init_model(params)

    # train model
    wrapper, best_error, *__ = vae_utils.train_model(
        rvae_f0s,
        wrapper,
        rvae_nois,
        rvae_lens,
        params=params
        )
    if best_error > 10:
        raise ValueError('Training VRAE did not converge!')

    # once trained extract the locations in the latent space
    locs_rvae, __ = vae_utils.get_mus_rvae(
        rvae_f0s,
        rvae_nois,
        rvae_lens,
        wrapper.model,
        )
    locs_rvae = locs_rvae.squeeze(1)
    # % t-SNE down to 2D
    tsne = TSNE(
        n_components=2, random_state=random_seed,
        perplexity=30,
        init='pca',
        n_iter=5000
        )
    locs_rvae_10d_tsne = tsne.fit_transform(locs_rvae)

    # unsort
    locs_rvae_10d_tsne[i_sort, :] = locs_rvae_10d_tsne.copy()
    rvae_lens[i_sort] = rvae_lens.copy()

    # %% plot
    if plot:
        __ = utils.plot_data_spread(
            locs_rvae_10d_tsne,
            f0s,
            rvae_lens,
            n_feats=n_feats,
            method='RVAE 10D latent space',
            x_scale=0.25,
            y_scale=0.0015,
            labels=labels,
            colors=colors,
            )
    # %%% save to dataframe
    locs['location_RVAE-10D'] = locs_rvae_10d_tsne.tolist()

# %% save the new locations in the original CSV
data = pd.concat([data, locs], axis=1, ignore_index=False)
utils.save_data(data, save_name)
