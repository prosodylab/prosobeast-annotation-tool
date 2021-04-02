#!/usr/bin/env python3
"""Analyse model performance for the RNNs.

Created on Dec 29 2020

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
import re
import seaborn as sb
import matplotlib

font = {'family' : 'Dejavu Sans',
        # 'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

# %% glob file names
# pkl_path = 'pkls/rvae_models'
pkl_path = '../pkls/rvae_models_gru_new_run'
pkl_names = glob.glob(f'{pkl_path}/wrapper*.pkl')
pkl_names = [os.path.basename(x).replace('.pkl','') for x in pkl_names]
# models = ['rnn', 'gru', 'lstm']
models = ['gru']
hiddens = [16, 32, 64, 128, 256, 512]
latents = list(range(2, 18, 2))

results_path = 'results/rvae_nois'

columns = 'model hidden latent inp loss'.split()
df = pd.DataFrame(columns=columns)

# %% extract best loss from all results and popultate
for file in pkl_names:
    model = re.search(r'^wrapper_([a-z]+)_', file).group(1)
    assert model in models
    hidden = re.search(r'([0-9]+)hid', file)
    if hidden is not None:
        hidden = int(hidden.group(1))
    else:
        hidden = 32
    assert hidden in hiddens
    latent = int(re.search(r'([0-9]+)lat', file).group(1))
    assert latent in latents
    loss = float(re.search(r'_([0-9]+.[0-9]+)($|_)', file).group(1))
    inp = 'inp' in file
    row = pd.Series([model, hidden, latent, inp, loss], index=columns)
    df = df.append(row, ignore_index=True)

# %% sort results
df = df.sort_values(by=['model', 'hidden', 'latent'])

# %% plot heatmaps
for model in models:
    mask_noinp = df.inp == False
    mask = mask_noinp & (df.model == model)
    df_pivot = df[mask].copy()
    # df_pivot.loss = df_pivot.loss.map(
    #     lambda x: int(x)).astype('int')
    df_pivot = df_pivot.pivot(
        index='hidden', columns='latent', values='loss'
        )
    for column in df_pivot.columns:
        df_pivot[column] = df_pivot[column].astype('int')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    heat = sb.heatmap(
        df_pivot,
        ax=ax,
        linewidths=0.5,
        linecolor='w',
        cmap='rocket_r',
        vmax=100,
        annot=True, fmt="d"
        )
    # xticklabels = heat.get_xticklabels()
    # for t in xticklabels:
    #     t.set_text(t.get_text().lower())
    # yticklabels = heat.get_yticklabels()
    # for t in yticklabels:
    #     t.set_text(t.get_text().lower())
    # heat.set_xticklabels(xticklabels, rotation=45)
    # pos = heat.get_yticks()
    heat.set_yticklabels(
        heat.get_yticklabels(), rotation=0, va="center"
        )
    # heat.set_ylim([0, len(pos)])
    # heat.invert_yaxis()
    ax.set_title(model.upper())
    fig.tight_layout()
    fig.savefig(
        f'{results_path}/'
        f'{model}'
        f'_{hidden}hid'
        f'_{latent}lat'
        '.png',
        dpi='figure', bbox_inches='tight', pad_inches=0,
        )
