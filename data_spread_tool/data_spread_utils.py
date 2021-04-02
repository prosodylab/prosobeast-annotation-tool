#!/usr/bin/env python3
"""
Utility functions for Data spread calculation.

@author:
    Branislav Gerazov Mar 2020
"""
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def load_csv(csv_name):
    """Load a single column CSV with no headers into an ndarray.

    Parameters
    ----------
    csv_name : str
        CSV filename.

    Returns
    -------
    data : ndarray
        Ndarray of data stored in CSV.
    """
    df = pd.read_csv(csv_name, header=None, names=['f0'])
    df.f0 = df.f0.map(
        lambda x: json.loads(x)
        )
    data = df.f0.tolist()
    data = np.array(data)
    return data


def load_data(data_name):
    """Load a multi column CSV with all intonation data.

    Parameters
    ----------
    csv_name : str
        CSV filename.

    Returns
    -------
    data : DataFrame
        Pandas DataFrame of data stored in CSV.
    f0s : ndarray
        F0 data from the CSV.
    lens : ndarray
        Lengths of the f0 contours.
    """
    data = pd.read_csv(data_name)
    # load from str all location columns
    for column in data.columns:
        if 'location' in column:
            data[column] = data[column].map(
                lambda x: json.loads(x)
                )
    # load f0 data
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
    return data, f0s, lens, labels


def load_label_colors(csv_name):
    """Return a dictionary of color values for each label.
    """
    colors = pd.read_csv(csv_name)
    colors.index = colors.label
    colors.drop('label', axis=1, inplace=True)
    colors = colors.to_dict()['color']
    return colors


def save_csv(data, csv_name):
    """Save an ndarray into a single column CSV with no headers.

    Parameters
    ----------
    data : ndarray
        Data to be saved.
    csv_name : str
        CSV filename.
    """
    data = data.tolist()
    data = [json.dumps(x) for x in data]
    df = pd.DataFrame(data)
    df.to_csv(
        csv_name,
        index=False,
        header=None,
        )


def save_data(data, csv_name):
    """Save a DataFrame of data into a multi column CSV with headers.

    Parameters
    ----------
    data : DataFrame
        Data to be saved.
    csv_name : str
        CSV filename.
    """
    for column in data.columns:
        if 'location' in column or ('f0' in column):
            data[column] = data[column].map(
                lambda x: json.dumps(x)
                )
    data.to_csv(
        csv_name,
        index=False,
        )


def plot_data_spread(
        locs,
        f0s,
        lengths=None,
        n_feats=5,
        method=None,
        x_scale=0.07,
        y_scale=0.008,
        save_path=None,
        labels=None,
        colors=None,
        ):
    """Plot calculated data spread of f0s.

    Parameters
    ----------
    locs : ndarray
        2D locations of pitch contours.
    f0s : ndarray
        Pitch contours to plot.
    lengths: array like
        Lengths of contours in number of NOIs for padded `f0s` holding
        variable length data.
    n_feats : int
        Number of f0 samples per NOI. Used with lengths.
    method : str
        Method name to use for plot title.
    x_scale : float
        scaling coefficient for the x-axis, use this to change the x-axis
        zoom of the plot
    y_scale : float
        scaling coefficient for the y-axis, use this to change the y-axis
        zoom of the plot
    save_path : str
        Path to save figure to. If not set, figure is not saved.
    labels : array like
        List of labels for each contour.
    colors : pd.DataFrame
        Dataframe mapping of labels to color codes.

    Returns
    -------
    fig, ax : obj
        Handles to the fig and ax objects of the plot for saving/closing.
    """
    figsize = 12, 10
    if lengths is None:
        lengths = [None] * locs.shape[0]  # so that the for loop works with zip
    # else:
    #     figsize = 16, 10
    if labels is None:
        labels = [None] * locs.shape[0]  # so that the for loop works with zip
    fig, ax = plt.subplots(figsize=figsize)

    plt.rc('font', size=16)

    for loc, f0, length, label in zip(locs, f0s, lengths, labels):
        # find each contours length
        if length is not None:
            x_length = length * n_feats
            f0 = f0[:x_length]
            if length > 3:
                c = 'C1'  # color longer contours with a different color
                lw = 4
                x_min, x_max = -1.33, 1.33
            else:
                lw = 3
                x_length = len(f0)
                c = 'C0'
                x_min, x_max = -1, 1
        else:
            lw = 3
            x_length = len(f0)
            c = 'C0'
            x_min, x_max = -1, 1
        # override color if labels are input
        if label is not None and colors is not None:
            c = colors[label]
        # generate x data
        x = np.linspace(x_min, x_max, x_length) * x_scale + loc[0]
        # generate y data
        y = f0 * y_scale + loc[1]
        if length is not None and length > 3:
            ax.plot(
                x, y,
                c=(.9, .9, .6),
                linewidth=lw+4, alpha=.8,
                )
            ax.plot(
                x, y,
                c=c,
                linewidth=lw, alpha=.8,
                )
        else:
            ax.plot(
                x, y,
                c=c,
                linewidth=lw, alpha=.8,
                )
    ax.grid(True)
    ax.set_xlabel('Dimension 0')
    ax.set_ylabel('Dimension 1')
    # title = '2D pitch contour projection '
    if method is not None:
        title = method
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(f'{save_path}/{method}.png')
    plt.show()
    return fig, ax
