#!/usr/bin/env python3
"""
Pitch extraction.

This code is adapted from the script pitch_extract_tool/pitch_extract_master_script.py
to be used by the annotation tool.

The tool takes as input a csv file (optional), the audio files and TextGrid
annotations.
    - The csv file has at least a column `file` with the filenames.
        Optionally it also includes columns `info` and `label` for use with
        the annotation tool.
        If the csv file is not found, one is created and populated with the names
        of the audio files in the path.
    It is important that the filename contains a speaker id to be used in the
    pitch extraction process.

    - The TextGrid annotations need to have at least two tiers:
        1. phones
        2. woi - Words of Interest

The tool extracts the pitch values from the Nuclei of Interest (NOI), defined as
the vowel regions in the WOIs.  Vowels are matched with a RegEx, currently
searching for a numbered stress mark at the end of the phone label as used in
the CMUdict and ARPABET.

The structure of the script is:
0. Init
1. Kaldi first pass
2. Calculate bounds
3. Kaldi second pass
4. Select good contours and sample
    - good contours are selected based on a percentage of the Probability of
    Voicing (POV) values in the NOIs being above the threshold

Notes:
- we use the min and max pitch bounds for each speaker to improve pitch
  extraction. Hirst has some thoughts on how to do it:

    We have found that a good approach is to do pitch detection in two
    steps. In the first step you use standard parameters and then from
    the distribution of pitch values, you get the 1st and 3rd quartile
    which we have found are quite well correlated with the minimum and
    maximum pitch, and finally use the estimated min and max for a second
    pitch detection. This avoids a considerable number of octave errors
    which are frequently found when using the standard arguments.

    https://uk.groups.yahoo.com/neo/groups/praat-users/conversations/topics/3472?guce_referrer=aHR0cDovL3d3dy5wcmFhdHZvY2FsdG9vbGtpdC5jb20vZXh0cmFjdC1waXRjaC5odG1s&guce_referrer_sig=AQAAAIDU5m6QVh0fVdsdE0b2etWRi49u3PKIN2BLKLWeuqlPrqXlo1Nn_TouJlGByEa361pcFeAnN6DWEbBvpd4ElCouJ0fD7eRiNz1-c_du6Psv3Gn4NXaCe62oQ8DCUa-HMspxd0d432ABbpukit0deIPiTc9Ba61WnenR24Kb66V2

- the tool plots with spectrogram, annotations and noi on plots.

Some of the code is taken from previous work on ProsoDeep.

@author:
    Branislav Gerazov Nov 2021
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import pickle
from collections import namedtuple
import os
import sys
from tqdm import tqdm
import re
import tgt  # textgrid tools
from pitch_extract_tool import pitch_extract_utils as pitch_utils
import json
from icecream import ic


def create_csv(wav_names, csv_name='prosobeast.csv'):
    data = pd.DataFrame(columns="file info label".split())
    file_names = [wav_name.replace(".wav", "") for wav_name in wav_names]
    data.file = file_names
    data.label = "No label"
    data.info = "No info"
    data.to_csv(csv_name)
    return data, file_names


def calculate_f0s(
        df_source,
        audio_path,
        textgrid_path,
        opts=None,
        save_pkls=False,
        save_plots=False,
        do_2nd_pass=True,  # allow skipping 2nd pass
        ):

    if save_pkls:
        pkl_path = "pkls"
        os.makedirs(pkl_path, exist_ok=True)
    if save_plots:
        plot_path = "figures"
        f0_plot_path= f"{plot_path}/f0s"
        hist_plot_path= f"{plot_path}/hists"
        good_plot_path= f"{plot_path}/good_f0s"
        os.makedirs(f0_plot_path, exist_ok=True)
        os.makedirs(hist_plot_path, exist_ok=True)
        os.makedirs(good_plot_path, exist_ok=True)

    kaldi_path = "pitch_extract_tool/bin"
    re_wav = re.compile(r".*\.wav")
    print(f"> Reading audio files from {audio_path} ...")
    wav_names = sorted([f for f in os.listdir(audio_path) if re_wav.match(f)])
    if df_source is None:
        print(f"> CSV file not found! Generating one ...")
        csv_name = 'prosobeast.csv'
        save_csv = True  # save CSV at the end
        df_source, file_names = create_csv(wav_names, csv_name=csv_name)
    else:
        save_csv = False  # save CSV at the end
        file_names = df_source.file.tolist()
        pitch_utils.check_consistency(
            wav_names,
            file_names,
            raise_error=False,
            verbose=1
            )

    # opts = {
    #     'f0_max_init': '750',
    #     'f0_min_init ': '60',
    #     'n_samples': '5',
    #     're_speaker': 'contour_([0-9]+?)_',
    #     're_vowels': '[0-2]$'}
    if opts is None:
        # speakers are designated as the first _ _ field in the prosobeast sample data
        # e.g. contour_618_5_1 -> speaker id 618
        re_speaker = re.compile(r"contour_([0-9]+?)_")  # non-greedy capture group
        re_vowels = re.compile(r"[0-2]$")

        # f0 bounds for first pass, tweak max if necessary
        f0_min_init = 60  # Hirst's suggested min
        # f0_min_init = 75
        f0_max_init = 750  # Hirst's suggested max
        # f0_max_init = 450  # for male speakers
        # f0_max_init = 600
        # how many samples
        n_samples = 5
    else:
        re_speaker = re.compile(opts["re_speaker"])  # non-greedy capture group
        re_vowels = re.compile(opts["re_vowels"])
        f0_min_init = float(opts["f0_min_init"])
        f0_max_init = float(opts["f0_max_init"])
        n_samples = int(opts["n_samples"])

    # define if phone tier starts and ends with a silence interval
    tier_phones_sil_start = False
    tier_phones_sil_end = True

    # probability of voicing thresholding for f0 extracted by Kaldi
    # 0.5 seems too restrictive, 0.2 is about right
    pov_thresh = 0.2
    # set good NOI selection POV percent threshold
    pov_perc_thresh = .5

    # sampling settings
    t_samples_perc = np.linspace(1/n_samples/2, 1-1/n_samples/2, n_samples)
    # nr of NOIs to take into consideration in data selection
    n_nois = None  # None takes all NOI numbers

    # plots settings
    if save_plots == True:
        do_plots = True
    else:
        do_plots = False
    # no sense in showin all the plots
    show_plots = False

    # %% 1. Kaldi first pass to find min and max
    # %%% extract start and end intervals from TextGrid
    print('> Loading textgrids ...')
    columns = "file speaker wav textgrid start end start_phones end_phones".split()
    df_f0_params = pd.DataFrame(columns=columns)
    for file_name in tqdm(file_names, ncols=90):
        # file_name = "contour_15_2_2"  # debug
        wav_name = file_name + ".wav"
        textgrid_name = file_name + ".TextGrid"
        # find speaker id
        res = re_speaker.search(file_name)
        if res is None:
            raise ValueError(f"Speaker id not found in {file_name}!")
        speaker = res.groups()[0]
        try:
            textgrid = tgt.read_textgrid(f"{textgrid_path}/{textgrid_name}")
        except:
            print(f"> Can't load TextGrid {textgrid_name}")
            print(sys.exc_info()[0])
            continue
        # find start and end times from first and last phone
        tier = textgrid.get_tier_by_name("phones")
        start_time = tier[0].start_time
        end_time = tier[-1].end_time  # last interval defaults to end of file
        # phone start and end times
        start_phones = tier[1].start_time if tier_phones_sil_start else start_time
        end_phones = tier[-2].end_time if tier_phones_sil_end else end_time
        # aggregate in data fram
        data = [
            file_name, speaker, wav_name, textgrid_name,
            start_time, end_time, start_phones, end_phones
            ]
        row = pd.Series(data, index=columns)
        df_f0_params = df_f0_params.append(row, ignore_index=True)
        # break  # debug

    if save_pkls:
        pkl_name = f"{pkl_path}/df_f0_params.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(df_f0_params, f, -1)

    # %%% get the f0s and plot them
    print("> Kaldi 1st pass ...")
    f0s_speakers_kaldi_1st = {}  # dict of f0s per speaker
    noi_f0s_kaldi_1st = {}  # dict of noi f0s
    noi_povs_kaldi_1st = {}  # dict of noi povs
    for (
            __, file_name, speaker, wav_name, textgrid_name,
            start_time, end_time, start_phones, end_phones
            ) in tqdm(
                df_f0_params.itertuples(), total=len(df_f0_params), ncols=90
                ):
        t, f0, __, __, __, pov = pitch_utils.kaldi_extract_pitch(
            kaldi_path, audio_path, wav_name, f0min=f0_min_init, f0max=f0_max_init
            )
        # trim
        i_start = np.where(t > start_phones)[0][0]
        i_end = np.where(t < end_phones)[0][-1]
        f0s = f0[i_start : i_end]
        povs = pov[i_start : i_end]
        ts = t[i_start : i_end]
        # apply pov based thresholding
        f0s_pov = f0s[povs > pov_thresh]
        ts_pov = ts[povs > pov_thresh]
        # update dictionary
        if speaker in f0s_speakers_kaldi_1st.keys():  # concatenate
            f0s_speakers_kaldi_1st[speaker] = np.r_[
                f0s_speakers_kaldi_1st[speaker],
                f0s_pov
                ]
        else:
            f0s_speakers_kaldi_1st[speaker] = f0s_pov

        # plot spectrogram
        if do_plots:
            fig, ax = pitch_utils.plot_pitch_spectrogram(
                audio_path, wav_name, f0_max_init, f0_min_init
                )
        # extract vowel f0s for ROIs = NOIs
        textgrid = tgt.read_textgrid(f"{textgrid_path}/{textgrid_name}")
        tier_phones = textgrid.get_tier_by_name("phones")
        tier_woi = textgrid.get_tier_by_name("woi")
        noi_f0s = []  # list of f0s per noi
        noi_povs = []  # list of povs per noi
        for interval in tier_phones.intervals:
            t_start = interval.start_time
            t_end = interval.end_time
            text = interval.text
            central_time = (t_start + t_end)/2
            # find NOIs
            if re_vowels.search(text):
                # check if vowel in woi accumulate as noi
                # TODO verify if it works for non consecutive WOIs
                if tier_woi.get_annotations_between_timepoints(
                        central_time, central_time,
                        left_overlap=True, right_overlap=True
                        ):
                    i_start_noi = np.where(t > t_start)[0][0]
                    i_end_noi = np.where(t < t_end)[0][-1]
                    noi_f0s.append(f0[i_start_noi : i_end_noi])
                    noi_povs.append(pov[i_start_noi : i_end_noi])
                    # highlight NOI segments
                    if do_plots:
                        ax = pitch_utils.highlight_segment(ax, t_start, t_end, f0_max_init)
            # add phone level annotations
            if do_plots:
                ax = pitch_utils.mark_phone_segment(
                    ax, t_start, t_end, f0_max_init, central_time, text
                    )
        noi_f0s_kaldi_1st[file_name] = noi_f0s
        noi_povs_kaldi_1st[file_name] = noi_povs
        # add f0 contour
        if do_plots:
            ax = pitch_utils.add_f0_contour(ax, ts, f0s, ts_pov, f0s_pov)
            save_name = (
                f"{f0_plot_path}/f0_{file_name}_"
                f"1st_pass_{f0_min_init}_{f0_max_init}.png"
                )
            pitch_utils.format_and_save_plot(
                fig, ax, save_name, start_time, end_time, f0_min_init, f0_max_init,
                show_plots=show_plots,
                )

    if save_pkls:
        pkl_name = f"{pkl_path}/kaldi_f0s_1st_pass_{f0_min_init}_{f0_max_init}.pkl"
        data = (
            f0s_speakers_kaldi_1st, noi_f0s_kaldi_1st, noi_povs_kaldi_1st,
            f0_min_init, f0_max_init
            )
        with open(pkl_name, "wb") as f:
            pickle.dump(data, f, -1)

    # %% 2. Calculate the bounds
    print("> Calculating the bounds ...")
    if do_2nd_pass:
        # dataframe for f0 bounds
        columns = "csv_min csv_max praat_min praat_max kaldi_min kaldi_max".split()
        data_f0_bounds_hirst = pd.DataFrame(columns=columns)
        # Stats = namedtuple("Stats", "mean median kde")  # namedtuples don't
        # pickle well
        f0_stats_kaldi = {}
        for speaker in tqdm(list(f0s_speakers_kaldi_1st.keys()), ncols=90):
            f0s_vec = f0s_speakers_kaldi_1st[speaker]
            f0_mean = np.mean(f0s_vec)
            f0_median = np.median(f0s_vec)
            f0_kde = pitch_utils.get_kde(f0s_vec)
            f0_stats_kaldi[speaker] = f0_mean, f0_median, f0_kde
            # upper and lower bounds
            f0_min_hirst, f0_max_hirst = pitch_utils.get_hirst_bounds(f0s_vec)
            data_f0_bounds_hirst.loc[speaker, "kaldi_min"] = f0_min_hirst
            data_f0_bounds_hirst.loc[speaker, "kaldi_max"] = f0_max_hirst
            #% plot histogram
            if do_plots:
                plot_name = f"{hist_plot_path}/f0_histogram_speaker_{speaker}.png"
                pitch_utils.plot_histogram(
                    f0s_vec, f0_mean, f0_median, f0_kde, f0_min_hirst, f0_max_hirst,
                    speaker, plot_name, show_plots=show_plots,
                    )
        if save_pkls:
            pkl_name = f"{pkl_path}/kaldi_f0_hirst_bounds_{f0_min_init}_{f0_max_init}.pkl"
            data = data_f0_bounds_hirst, f0_stats_kaldi, f0_min_init, f0_max_init
            with open(pkl_name, "wb") as f:
                pickle.dump(data, f, -1)

    # %% 3. Kaldi second pass
    print("> Kaldi 2nd pass ...")
    if do_2nd_pass:
        f0s_speakers_kaldi_2nd = {}  # dict of f0s per speaker
        f0s_file_kaldi_2nd = {}  # dict of f0s per file
        ts_file_kaldi_2nd = {}  # dict of f0s per file
        noi_f0s_kaldi_2nd = {}  # dict of noi f0s
        noi_ts_kaldi_2nd = {}  # dict of noi ts
        noi_povs_kaldi_2nd = {}  # dict of noi povs
        noi_bounds_files = {}  # dict of noi bounds
        for (
                __, file_name, speaker, wav_name, textgrid_name,
                start_time, end_time, start_phones, end_phones
                ) in tqdm(
                    df_f0_params.itertuples(), total=len(df_f0_params), ncols=90
                    ):
            f0_min_speaker = data_f0_bounds_hirst.loc[speaker, "kaldi_min"]
            f0_max_speaker = data_f0_bounds_hirst.loc[speaker, "kaldi_max"]
            t, f0, __, __, __, pov = pitch_utils.kaldi_extract_pitch(
                kaldi_path, audio_path, wav_name, f0min=f0_min_speaker, f0max=f0_max_speaker
                )
            # trim
            i_start = np.where(t > start_phones)[0][0]
            i_end = np.where(t < end_phones)[0][-1]
            f0s = f0[i_start : i_end]
            povs = pov[i_start : i_end]
            ts = t[i_start : i_end]
            # pov
            f0s_pov = f0s[povs > pov_thresh]
            ts_pov = ts[povs > pov_thresh]
            # update dictionaries
            if speaker in f0s_speakers_kaldi_2nd.keys():  # concatenate
                f0s_speakers_kaldi_2nd[speaker] = np.r_[
                        f0s_speakers_kaldi_2nd[speaker], f0s_pov]
            else:
                f0s_speakers_kaldi_2nd[speaker] = f0s_pov
            f0s_file_kaldi_2nd[file_name] = f0s
            ts_file_kaldi_2nd[file_name] = ts

            # plot f0s
            if do_plots:
                fig, ax = pitch_utils.plot_pitch_spectrogram(
                    audio_path, wav_name, f0_max_init, f0_min_init
                    )
            textgrid = tgt.read_textgrid(f"{textgrid_path}/{textgrid_name}")
            tier_phones = textgrid.get_tier_by_name("phones")
            tier_woi = textgrid.get_tier_by_name("woi")
            noi_f0s = []  # list of f0s per noi
            noi_povs = []  # list of povs per noi
            noi_ts = []  # list of ts where f0 is sampled
            noi_bounds = []  # list of noi bounds
            for interval in tier_phones.intervals:
                t_start = interval.start_time
                t_end = interval.end_time
                text = interval.text
                central_time = (t_start + t_end)/2
                # find NOIs
                if re_vowels.search(text):
                    # check if in woi accumulate as noi
                    # TODO verify if it works for non consecutive WOIs
                    if tier_woi.get_annotations_between_timepoints(
                            central_time, central_time,
                            left_overlap=True, right_overlap=True
                            ):
                        i_start_noi = np.where(t > t_start)[0][0]
                        i_end_noi = np.where(t < t_end)[0][-1]
                        noi_f0s.append(f0[i_start_noi : i_end_noi])
                        noi_povs.append(pov[i_start_noi : i_end_noi])
                        noi_ts.append(t[i_start_noi : i_end_noi])
                        noi_bounds.append((t_start, t_end))
                        # highlight NOI segments
                        if do_plots:
                            ax = pitch_utils.highlight_segment(ax, t_start, t_end, f0_max_init)
                # add phone level annotations
                if do_plots:
                    ax = pitch_utils.mark_phone_segment(
                        ax, t_start, t_end, f0_max_init, central_time, text
                        )
            noi_f0s_kaldi_2nd[file_name] = noi_f0s
            noi_povs_kaldi_2nd[file_name] = noi_povs
            noi_ts_kaldi_2nd[file_name] = noi_ts
            noi_bounds_files[file_name] = noi_bounds
            # add f0 contour
            if do_plots:
                ax = pitch_utils.add_f0_contour(ax, ts, f0s, ts_pov, f0s_pov)
                save_name = (
                    f"{f0_plot_path}/f0_{file_name}_"
                    f"2nd_pass_{f0_min_init}_{f0_max_init}.png"
                    )
                pitch_utils.format_and_save_plot(
                    fig, ax, save_name, start_time, end_time, f0_min_init, f0_max_init,
                    show_plots=show_plots,
                    )

        if save_pkls:
            pkl_name = f"{pkl_path}/kaldi_f0s_2nd_pass_{f0_min_init}_{f0_max_init}.pkl"
            data = (
                f0s_speakers_kaldi_2nd, f0s_file_kaldi_2nd, ts_file_kaldi_2nd,
                noi_f0s_kaldi_2nd, noi_povs_kaldi_2nd, noi_ts_kaldi_2nd,
                noi_bounds_files, data_f0_bounds_hirst,
                f0_min_init, f0_max_init
                )
            with open(pkl_name, "wb") as f:
                pickle.dump(data, f, -1)

    # %% 4. select good contours and save csv
    print("> Selecting the good contours and sampling ...")
    # find good files and check if pov_perc of pov is above threshold
    # df_csv  # should have columns="file info label"
    columns = "file info label durs f0s".split()
    df_f0s_good = pd.DataFrame(columns=columns)
    # columns = 'file speaker wav textgrid start end start_phones end_phones'
    for __, file_name, speaker, wav_name, textgrid_name, __, __, __, __ in tqdm(
            df_f0_params.itertuples(), total=len(df_f0_params), ncols=90
            ):
        f0s = f0s_file_kaldi_2nd[file_name]
        ts = ts_file_kaldi_2nd[file_name]
        noi_povs = noi_povs_kaldi_2nd[file_name]  # list of povs
        noi_bounds = noi_bounds_files[file_name]
        if n_nois is None or len(noi_povs) == n_nois:
            # POV checks
            for i, pov in enumerate(noi_povs):
                pov_perc = np.sum(pov > pov_thresh) / len(pov)
                if pov_perc < pov_perc_thresh:
                    print(f"\n> {file_name} did not pass pov check NOI {i} perc {pov_perc}\n")
                    continue

            # check passed, sample f0s and store
            t_dur_vec = []
            f0_samples_vec = []
            t_samples_vec = []  # for plotting
            for (t_start, t_end), ts, f0s in zip(
                    noi_bounds,
                    noi_ts_kaldi_2nd[file_name], noi_f0s_kaldi_2nd[file_name]
                    ):
                # sample
                interfunc = interp1d(
                    ts, f0s,
                    kind="linear",  # "linear", "nearest", "zero", "slinear",
                    # "quadratic", "cubic", "previous", "next", where "zero",
                    # "slinear", "quadratic" and "cubic" refer to spline interpolation
                    fill_value="extrapolate",  # might happen
                    bounds_error=False,
                    )
                t_dur = t_end - t_start
                t_samples = t_start + t_dur * t_samples_perc
                t_samples = t_samples.tolist()
                f0_samples = interfunc(t_samples).tolist()
                # store
                t_samples_vec += t_samples
                f0_samples_vec += f0_samples
                t_dur_vec.append(t_dur)

            # aggregate data
            # retrieve info and labels from csv
            mask = df_source.file == file_name
            info = df_source.loc[mask, "info"]
            label = df_source.loc[mask, "label"]
            row = pd.Series(
                [file_name, info, label,
                 json.dumps(t_dur_vec), json.dumps(f0_samples_vec)],
                index=columns
                )
            df_f0s_good = df_f0s_good.append(row, ignore_index=True)

            if do_plots:
                fig, ax = pitch_utils.plot_pitch_spectrogram(
                    audio_path, wav_name, f0_max_init, f0_min_init
                    )
                textgrid = tgt.read_textgrid(f"{textgrid_path}/{textgrid_name}")
                tier_phones = textgrid.get_tier_by_name("phones")
                tier_woi = textgrid.get_tier_by_name("woi")
                for interval in tier_phones.intervals:
                    t_start = interval.start_time
                    t_end = interval.end_time
                    text = interval.text
                    central_time = (t_start + t_end)/2
                    if re_vowels.search(text):
                        if tier_woi.get_annotations_between_timepoints(
                                central_time, central_time,
                                left_overlap=True, right_overlap=True
                                ):
                                ax = pitch_utils.highlight_segment(ax, t_start, t_end, f0_max_init)
                    # add phone level annotations
                    ax = pitch_utils.mark_phone_segment(
                        ax, t_start, t_end, f0_max_init, central_time, text
                        )
                ax = pitch_utils.add_f0_contour(ax, ts, f0s, t_samples_vec, f0_samples_vec)
                save_name = (
                    f"{good_plot_path}/good_f0_{file_name}_"
                    f"2nd_pass_{f0_min_init}_{f0_max_init}.png"
                    )
                pitch_utils.format_and_save_plot(
                    fig, ax, save_name, start_time, end_time, f0_min_init, f0_max_init,
                    show_plots=show_plots,
                    )

    if save_pkls:
        pkl_name = f"{pkl_path}/good_f0s.pkl"
        with open(pkl_name, "wb") as f:
            pickle.dump(df_f0s_good, f, -1)
    # save data
    if save_csv:
        df_f0s_good.to_csv(csv_name)

    # df_f0s_good columns = "file info label durs f0s".split()
    # add durs and f0s to original data sent
    df_source['dur'] = df_f0s_good.durs
    df_source['f0'] = df_f0s_good.f0s
    # clean up kaldi files
    os.remove(f"{kaldi_path}/temp.scp")
    os.remove(f"{kaldi_path}/temp.ark")
    return df_source
