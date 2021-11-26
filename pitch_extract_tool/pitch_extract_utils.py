#!/usr/bin/env python3
import os
import subprocess

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.stats import gaussian_kde
import librosa

try:
    from matplotlib import pyplot as plt
except ImportError:
    print('No matplotlib installed!')
try:
    import seaborn as sns
except ImportError:
    print('No seaborn installed!')


def kaldi_extract_pitch(kaldi_path, wav_path, wav_name, f0min=80, f0max=300, fs=None):
    """
    Kaldi based pitch extractor - adapted from WCAD implementation:
    https://github.com/dipteam/wcad
    """
    # Make temp files
    kaldi_scp_file = f"{kaldi_path}/temp.scp"
    kaldi_ark_file = f"{kaldi_path}/temp.ark"
    kaldi = f"{kaldi_path}/compute-kaldi-pitch-feats"

    # Write scp file for Kaldi
    with open(kaldi_scp_file, "w") as text_file:
        text_file.write(f"{wav_name} {wav_path}/{wav_name}\n")

    # Set Kaldi"s parameters:
    if fs is None:
        fs, _ = wavfile.read(f"{wav_path}/{wav_name}")
    frame_shift = 5  # in milliseconds, so frameRate is 200Hz
    frame_rate = 1000 / frame_shift
    frame_size = 50

    kaldi_params = f" --sample-frequency={fs}"
    kaldi_params += f" --frame-length={frame_size}"
    kaldi_params += f" --frame-shift={frame_shift}"
    kaldi_params += f" --min-f0={f0min}"
    kaldi_params += f" --max-f0={f0max}"
    kaldi_params += f" scp:{kaldi_scp_file}"
    kaldi_params += f" ark:{kaldi_ark_file}"

    subprocess.call(
        kaldi + kaldi_params, shell=True,
        stdout=open(os.devnull, "wb"), stderr=open(os.devnull, 'wb')
        )
    # Kaldi's result is stored as tuple (nccf, f0), with file's name in the header
    kaldi_pitch_t = np.dtype([("nccf", np.float32), ("f0", np.float32)])
    # new type: (nccf, f0)
    with open(kaldi_ark_file, "rb") as file_obj:
        file_obj.seek(len(wav_name) + 16)  # skipping the header
        kaldi_data = np.fromfile(file_obj, dtype=kaldi_pitch_t)
    f0 = kaldi_data["f0"]
    nccf = kaldi_data["nccf"]

    # Convert NCCF to POV. According to Kaldi paper:
    # [ l = log(p(voiced)/p(unvoiced)) ]
    a = np.abs(nccf)
    l = -5.2 + 5.4*np.exp(7.5*(a - 1)) + 4.8*a - 2*np.exp(-10*a) + 4.2*np.exp(20*(a - 1))
    pov = 1./(1 + np.exp(-l))

    # offset = frame_size/1000
    step = frame_shift / 1000
    # t = np.arange(offset, f0.size*step + offset, step)
    t = np.arange(0, f0.size * step, step)
    t = t[: f0.size]  # avoid +-1 errors in length
    # ## TODO check time alignment of results!
    # Then we need to extend both sides because Kaldi starts from half window length
    # len_energy = len(energy)
    # len_f0 = len(f0)
    # dif = len_energy - len_f0
    # dif_h = dif // 2
    # f0 = np.pad(f0, (dif_h, dif_h), "edge")
    # pov = np.pad(pov, (dif_h, dif_h), "edge")
    #
    # if dif % 2:  # odd, then 1 is missing
    #     f0 = np.pad(f0, (1, 0), "edge")
    #     pov = np.pad(pov, (1, 0), "edge")
    #
    f0_log = np.log(f0)

    # Temp
    lp_fg = 5  # 5 Hz works best for C2
    lp_wg = lp_fg / (0.5 * frame_rate)
    b, a = signal.butter(4, lp_wg, btype="lowpass")
    f0_log_filt = signal.filtfilt(b, a, f0_log, padtype="odd", padlen=len(f0_log)-1)
    f0_filt = signal.filtfilt(b, a, f0, padtype="odd", padlen=len(f0)-1)

    return t, f0, f0_log, f0_filt, f0_log_filt, pov


def get_hirst_bounds(f0s_vec):
    """ Hirst"s code:
    To Pitch... 0.01 60 750
    q25 = Get quantile... 0 0 0.25 Hertz
    q75 = Get quantile... 0 0 0.75 Hertz
    minPitch = q25 * 0.75
    maxPitch = q75 * 1.5
    """
    q25 = np.percentile(f0s_vec, 25)
    q75 = np.percentile(f0s_vec, 75)
    f0_min_hirst = q25 * 0.75
    f0_max_hirst = q75 * 1.5
    return f0_min_hirst, f0_max_hirst


def get_stft(wav_path, wav_name):
    y, fs = librosa.load(f"{wav_path}/{wav_name}", sr=None)
    # default n_fft = 2048, hop = 512,
    # center=True -->  `D[:, t]` is centered at `y[t * hop_length]`.
    n_fft = 1024
    hop = 128
    stft = librosa.core.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=None,
        window='hann',
        center=False,
        pad_mode='reflect'
        )
    hop_t = hop/fs
    t_frames = np.arange(0,stft.shape[1]*hop_t, hop_t)
    return fs, y, stft, t_frames


def plot_spectrogram(frames, t_frames, fs):
    plt.imshow(
        frames,
        extent=[0, t_frames[-1], 0, fs/2],
        aspect='auto',
        origin='lower',
        vmin=-40,
        vmax=60,
        cmap='gray'
        )


def plot_pitch_spectrogram(audio_path, wav_name, f0_max_init, f0_min_init):
    fs, y, stft, t_frames = get_stft(audio_path, wav_name)
    spectrum = -20 * np.log(np.abs(stft))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(
        np.arange(0, y.size)/fs,
        y/np.max(y) * f0_max_init/2 + f0_max_init/2,
        alpha=.3
        )
    ax.imshow(
        spectrum,
        extent=[0, t_frames[-1], f0_min_init, 2 * f0_max_init],
        aspect="auto",
        origin="lower",
        vmin=-50,
        vmax=80,
        cmap="gray",
        alpha=.5
        )
    return fig, ax

def highlight_segment(ax, t_start, t_end, f0_max_init):
    ax.axvspan(
        xmin=t_start, xmax=t_end,
        ymin=0, ymax=f0_max_init,
        alpha=0.25, color="C3"
        )
    return ax


def mark_phone_segment(ax, t_start, t_end, f0_max_init, central_time, text):
    ax.axvline(x=t_start, ymax=f0_max_init, c="C3", lw=2, alpha=.3)
    ax.axvline(x=t_end, ymax=f0_max_init, c="C3", lw=2, alpha=.3)
    ax.text(
        central_time, f0_max_init-50,
        text, horizontalalignment="center"
        )
    return ax


def add_f0_contour(ax, ts, f0s, ts_pov, f0s_pov):
    ax.plot(ts, f0s, lw=5, c="C0", alpha=.3)
    ax.plot(ts_pov, f0s_pov, "o", ms=6, c="C0", alpha=.8)
    return ax


def format_and_save_plot(
        fig, ax, save_name,
        start_time, end_time,
        f0_min_init, f0_max_init,
        show_plots=False,
        ):
    ax.axis([start_time, end_time, f0_min_init, f0_max_init])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_name, dpi="figure")
    if not show_plots:
        plt.close(fig)


def get_kde(f0s_vec):
    """ Get kde using seaborn - deprecated"""
    # fig, ax = plt.subplots()
    # ax = sns.distplot(f0s_vec, ax=ax)
    # kde_x, kde_y = ax.get_lines()[0].get_data()
    # f0_kde = kde_x[np.argmax(kde_y)]
    # plt.close(fig)
    """ Get kde using scipy.stats.
    https://stackoverflow.com/questions/31198020/how-to-find-local-maxima-in-kernel-density-estimation#31212174
    """
    kde = gaussian_kde(f0s_vec)
    n_samples = 100
    f0_samples = np.linspace(f0s_vec.min(), f0s_vec.max(), n_samples)
    kde_probs = kde.evaluate(f0_samples)
    kde_max_ind = kde_probs.argmax()
    kde_max= f0_samples[kde_max_ind]
    return kde_max


def plot_histogram(
            f0s_vec, f0_mean, f0_median, f0_kde, f0_min_hirst, f0_max_hirst,
            speaker, plot_name, show_plots=False,
        ):
    sns.set(color_codes=True, style='ticks')
    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.histplot(f0s_vec, ax=ax, kde=True)
    ax.axvline(f0_mean, c="C1", lw=2, alpha=.7, label="mean")
    ax.axvline(f0_median, c="C2", lw=2, alpha=.7, label="median")
    ax.axvline(f0_kde, c="C4", lw=2, alpha=.7, label="kde")
    ax.axvline(f0_min_hirst, c="C5", lw=2, alpha=.7, label="f0 min hirst")
    ax.axvline(f0_max_hirst, c="C5", lw=2, alpha=.7, label="f0 max hirst")
    ax.legend()
    plt.title(f"Kaldi speaker {speaker} f0 histogram\n"
              f"mean = {f0_mean:.0f}, median = {f0_median:.0f}, "
              f"kde = {f0_kde:.0f}, "
              f"Hirst bounds = {f0_min_hirst:.0f} - {f0_max_hirst:.0f}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Density")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(plot_name, dpi="figure")
    if not show_plots:
        plt.close(fig)


def check_consistency(wav_names, file_names, raise_error=False):
    for wav_name in wav_names:
        if wav_name.replace(".wav", "") not in file_names:
            message = f"{wav_name} not found in CSV file names!"
            if raise_error:
                raise ValueError(f"> Error! {message}")
            else:
                print(f"> Warning: {message}")
    for file_name in file_names:
        if file_name + ".wav" not in wav_names:
            message = f"{file_name} not found in audio path!"
            if raise_error:
                raise ValueError(f"> Error! {message}")
            else:
                print(f"> Warning: {message}")

