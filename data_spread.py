#!/usr/bin/env python3
"""
Data spread calculation.

This code is adapted from the script meant to be used by the annotation tool.

The code loads a CSV file of pitch contours and outputs a CSV file of their
locations in a 2D projection space based on their similarity.

The tool offers the following ways to calculate the spread:
    1. For fixed size contours:
        - PCA
        - t-SNE
        - 2D DNN based VAE latent space
        - 4D DNN based VAE latent space reduced to 2D with t-SNE
    2. For contours of fixed and varying size
        - 10D GRU based VAE latent space reduced to 2D with t-SNE

The input should contain a column of pitch data formatted as strings
with a header 'f0':
    [-2.50, -2.81, -3.12, ... -3.21]
    [-1.23, -0.81, -1.12, ... -2.35]
    ...

@author:
    Branislav Gerazov Mar 2020
"""
import json

import numpy as np
import pandas as pd

# methods
try:  # won't work if there is no sklearn
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    print('No scikit-learn installed!')

try:  # for these you need PyTorch
    from data_spread_tool import data_vae_utils as vae_utils
    from data_spread_tool import data_vae_params as vae_params
except ImportError:
    print('No PyTorch installed!')


def load_data(data):
    """Load dataa multi column CSV with all intonation data.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame of data stored in CSV.

    Returns
    -------
    data : DataFrame
        Pandas DataFrame of data stored in CSV.
    f0s : ndarray
        F0 data from the CSV.
    lens : ndarray
        Lengths of the f0 contours.
    """
    data.f0 = data.f0.map(lambda x: json.loads(x))
    f0s = data.f0.tolist()
    # check if all data is the same length
    lens = np.array([len(f0) for f0 in f0s])
    if np.all(lens == lens[0]):
        f0s = np.array(f0s)
        lens = int(lens[0])
    else:
        # pad to max len
        max_len = lens.max()
        f0s = np.array([
            np.pad(f0, (0, max_len - len(f0)), constant_values=np.nan)
            for f0 in f0s
            ])
    labels = data.label.tolist()
    return f0s, lens, labels


def load_label_colors(csv_name):
    """Return a dictionary of color values for each label.
    """
    colors = pd.read_csv(csv_name)
    colors.index = colors.label
    colors.drop('label', axis=1, inplace=True)
    colors = colors.to_dict()['color']
    return colors


def calculate_data_spread(
        data=None,
        choice=None,
        seed=None,
        ):
    """
    Calculate data spread for f0 contours based on choices.

    Parameters
    ----------
    data : DataFrame
        Pandas dataframe with a 'f0' column containing the f0 data.
    choice : str
        One of the supported methods to use for calculating the data spread.
        Possible choices: PCA, t-SNE, VAE-2D, VAE-4D
    seed : int
        Random seed to ensure reproducible results.

    Returns
    -------
    data : DataFrame
        Pandas dataframe with a 'location' column containing the f0 data.

    """

    print(f'Calculate data spread {choice}.')
    # %% load data
    f0s, lens, labels = load_data(data)
    if isinstance(lens, int):  # same length data
        same_length = True
    else:
        same_length = False
    n_samples = f0s.shape[0]
    len_f0 = f0s.shape[1]
    n_feats = 5  # f0 samples per NOI - important for the RVAE

    # init locs df
    locs = pd.DataFrame()

    # %% PCA
    if choice == 'pca':
        if not same_length:
            raise ValueError('F0 data has to have the same length!')
        print('calculating PCA spread ... ', end='', flush=True)
        pca = PCA(n_components=2, whiten=False, copy=True)
        locs_pca = pca.fit_transform(f0s)
        locs['location'] = locs_pca.tolist()
        print('Done!')

    # %% tSNE analysis
    elif choice == 'tsne':
        if not same_length:
            raise ValueError('F0 data has to have the same length!')
        print('calculating t-SNE spread ... ', end='', flush=True)
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=30,
            init='pca',  # PCA initialization is usually more globally stable
            n_iter=5000
            )
        locs_tsne = tsne.fit_transform(f0s)
        print(f'fit transform complete')
        locs['location'] = locs_tsne.tolist()
        print('Done!')

    # %% DNN based 2D VAE latent space
    elif choice == 'vae2d':
        if not same_length:
            raise ValueError('F0 data has to have the same length!')
        print('calculating VAE-2D spread ... ', end='', flush=True)
        params = vae_params.Params()
        params.n_latent = 2
        params.hidden_units = [128] * 2  # works well for our data
        params.n_feats = len_f0
        wrapper = vae_utils.init_model(params)
        wrapper = vae_utils.train_model(f0s, wrapper, params=params)[0]
        # once trained extract the locations in the latent space
        locs_vae_2d, __ = vae_utils.get_mus(f0s, wrapper.model)
        locs['location'] = locs_vae_2d.tolist()
        print('Done!')

    # %% DNN based 4D VAE latent space
    elif choice == 'vae4d':
        if not same_length:
            raise ValueError('F0 data has to have the same length!')
        print('calculating VAE-4D spread ... ', end='', flush=True)
        params = vae_params.Params()
        params.model_type = 'dnn'
        params.n_latent = 4
        params.hidden_units = [128] * 2  # works well for our data
        params.n_feats = len_f0
        wrapper = vae_utils.init_model(params)
        wrapper = vae_utils.train_model(f0s, wrapper, params=params)[0]
        # once trained extract the locations in the latent space
        locs_vae_4d, __ = vae_utils.get_mus(f0s, wrapper.model)
        # % t-SNE down to 2D
        print('Done! calculating t-SNE spread ... ', end='', flush=True)
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=30,
            init='pca',
            n_iter=5000
            )
        locs_vae_4d_tsne = tsne.fit_transform(locs_vae_4d)
        locs['location'] = locs_vae_4d_tsne.tolist()
        print('Done!')

    # %% RNN based 10D VAE latent space
    elif choice == 'rvae10d':
        print('calculating Recurrent VAE 10D spread ... ', end='', flush=True)

        # reshape data
        # f0 is n_samples x n_f0s (= n_nois * n_samples)
        # should be n_nois x n_samples x n_samples
        if same_length:
            max_len = len_f0
            n_nois = int(max_len / n_feats)
            rvae_lens = np.repeat(n_nois, n_samples)
        else:
            n_nois = int(lens.max() / n_feats)
            rvae_lens = np.array(lens / n_feats, dtype='int')

        rvae_f0s = f0s.reshape(n_samples, n_nois, n_feats)
        rvae_f0s = rvae_f0s.transpose((1, 0, 2))
        # generate ramps
        rvae_nois = [vae_utils.get_noi_ramps(x) for x in rvae_lens]
        rvae_nois = [
            np.pad(
                x,
                ((0, n_nois - x.shape[0]), (0, 0)),
                constant_values=np.nan
                )
            for x in rvae_nois
            ]
        rvae_nois = np.array(rvae_nois)
        rvae_nois = rvae_nois.transpose((1, 0, 2))

        # sort contours in descending length for packing the tensor
        i_sort = np.argsort(rvae_lens)[::-1]
        rvae_f0s = rvae_f0s[:, i_sort, :]
        rvae_nois = rvae_nois[:, i_sort, :]
        rvae_lens = rvae_lens[i_sort]

        # RVAE params that work well
        params = vae_params.Params()
        params.model_type = 'rnn'
        params.shuffle = False
        params.rnn_model = 'gru'
        params.hidden_units = [512]
        params.n_latent = 10
        params.n_feats = n_feats
        params.learn_rate = 0.001
        params.patience = 500
        params.batch_size = 512
        params.use_validation = False
        params.validation_size = 0.2
        wrapper = vae_utils.init_model(params)

        # train model
        wrapper, best_error, *__ = vae_utils.train_model(
            rvae_f0s,
            wrapper,
            data_nois=rvae_nois,
            data_lens=rvae_lens,
            params=params
            )
        # if best_error > 5:
        #     raise ValueError('Training VRAE did not converge!')

        print('Done! calculating t-SNE spread ... ', end='', flush=True)
        # extract the projections in the latent space
        locs_rvae_10d, sigmas = vae_utils.get_mus_rvae(
            rvae_f0s,
            rvae_nois,
            rvae_lens,
            wrapper.model,
            )
        locs_rvae_10d = locs_rvae_10d.squeeze(1)
        sigmas = sigmas.squeeze(1)
        print(f'Sigma mean {sigmas.mean()}, std {sigmas.std()}')

        if params.n_latent > 2:
            # t-SNE down to 2D
            tsne = TSNE(
                n_components=2,
                random_state=seed,
                perplexity=30,
                init='pca',
                n_iter=5000
                )
            locs_rvae_10d = tsne.fit_transform(locs_rvae_10d)
            print('locs rvae mus shape ', locs_rvae_10d.shape)

        # unsort
        locs_rvae_10d_unsorted = locs_rvae_10d.copy()
        locs_rvae_10d_unsorted[i_sort, :] = locs_rvae_10d

        locs['location'] = locs_rvae_10d_unsorted.tolist()
        print('Done!')

    return locs
