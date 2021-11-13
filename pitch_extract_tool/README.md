# Pitch extraction tool

Extract pitch using Kaldi pitch extractor and sample Regions of Interest using
`TextGrid` annotations.

The tool takes as input a `csv` file (optional), the audio files and `TextGrid`
annotations.
    - The `csv` file has at least a column `file` with the filenames.
        Optionally it also includes columns `info` and `label` for use with
        the annotation tool.
        If the csv file is not found, one is created and populated with the names
        of the audio files in the path.
    It is important that the filename contains a speaker id to be used in the
    pitch extraction process.

    - The `TextGrid` annotations need to have at least two tiers:
        1. phones
        2. woi - Words of Interest

The tool extracts the pitch values from the Nuclei of Interest (NOI), defined as
the vowel regions in the WOIs. Vowels are matched with a RegEx, currently
searching for a numbered stress mark at the end of the phone label as used in
the CMUdict and ARPABET.

The structure of the script is:
0. Init
1. Kaldi first pass
2. Calculate bounds
3. Kaldi second pass
4. Select good contours and sample
    - good contours are selected based on a percentage of the Probability of Voicing (POV) values in the NOIs being above the threshold

## Kaldi Pitch Extractor

To run the tool you need a compiled binary for the pitch tracker `compute-kaldi-pitch-feats` from [Kaldi](http://kaldi-asr.org/). It provides a continuous pitch estimate and a probability of voicing. Kaldi's code can be found [here](https://github.com/kaldi-asr/kaldi). For convenience we include a precompiled binary for a `x86_64 GNU/Linux` architecture.


## Sample dataset

When run on the sample dataset provided with the ProsoBeast Annotation Tool, assuming a POV threshold of 0.2 and a POV percentage threshold of 0.5, it will generate the CSV discarding two of the contours:
    - `contour_1645_5_2`
    - `contour_1646_4_1`

```
 44%|██████████████████████▌                            | 133/300 [00:43<00:49,  3.37it/s]contour_1645_5_2 did not pass pov check NOI 2 perc 0.4642857142857143
 47%|████████████████████████▏                          | 142/300 [00:45<00:50,  3.12it/s]contour_1646_4_1 did not pass pov check NOI 1 perc 0.45454545454545453
100%|███████████████████████████████████████████████████| 300/300 [01:35<00:00,  3.13it/s]
```

## Notes

We use the min and max pitch bounds for each speaker to improve pitch extraction. Hirst has some thoughts on [how to do it](https://uk.groups.yahoo.com/neo/groups/praat-users/conversations/topics/3472?guce_referrer=aHR0cDovL3d3dy5wcmFhdHZvY2FsdG9vbGtpdC5jb20vZXh0cmFjdC1waXRjaC5odG1s&guce_referrer_sig=AQAAAIDU5m6QVh0fVdsdE0b2etWRi49u3PKIN2BLKLWeuqlPrqXlo1Nn_TouJlGByEa361pcFeAnN6DWEbBvpd4ElCouJ0fD7eRiNz1-c_du6Psv3Gn4NXaCe62oQ8DCUa-HMspxd0d432ABbpukit0deIPiTc9Ba61WnenR24Kb66V2):

> We have found that a good approach is to do pitch detection in two steps. In the first step you use standard parameters and then from the distribution of pitch values, you get the 1st and 3rd quartile which we have found are quite well correlated with the minimum and maximum pitch, and finally use the estimated min and max for a second pitch detection. This avoids a considerable number of octave errors which are frequently found when using the standard arguments.


Some of the code is taken from previous work on [ProsoDeep](https://github.com/gerazov/prosodeep).

