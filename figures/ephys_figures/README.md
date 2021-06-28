# Ephys figures

## Dependencies

All of these figures depend on external data, either NWB files downloaded via
the AllenSDK, or manuscript-specific data stored in `data/local_large_data`.

Figure generation uses the standard numpy/scipy/matplotlib/pandas stack, as well as 
some methods from the [AllenSDK](https://github.com/alleninstitute/allensdk) and
[ecephys_spike_sorting](https://github.com/alleninstitute/ecephys_spike_sorting)
packages.

## Overview of files

### Data pre-processing

* `ks_results.py` - a class that stores spike sorting outputs from Kilosort
* `manual_noise_unit_ids.py` - contains a dictionary of noise units identified in a manual curation step
* `generate_confusion_matrix.py` - calculates the precision, accuracy, and recall of spike times between pairs of nearby units before and after denoising
* `calculate_reliabilities.py` - calculates 3 natural movie reliability metrics for units in visual cortex
* `h5_to_dat_converter.py` - (for reference only) converts the output of ephys DeepInterpolation to a .dat file that can be used for spike sorting

### Figure generation

* `Figure_3bc.py` - plots side-by-side comparison of raw data before and after denoising
* `Figure_3d.py` - plots histograms of channel RMS noise
* `Figure_3ef.py` - plots histograms of unit amplitude and SNR
* `Supp_Figure_6a.py` - plots example residual data and average power spectra
* `Supp_Figure_6b.py` - plots number of included units as a function of quality metric threshold
* `Supp_Figure_7.py` - plots depth histograms and waveform metric distributions before and after denoising