import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

path_to_raw_noisy = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "local_large_data",
    "ground_truth",
    "20191215_raw_noisy.npy",
)


path_to_raw_gaussian = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data",
    "local_large_data",
    "ground_truth",
    "20191215_raw_gaussian.npy",
)

raw_movie_noisy = np.load(path_to_raw_noisy)

raw_movie_gaussian = gaussian_filter1d(raw_movie_noisy, sigma=3)

np.save(path_to_raw_gaussian, raw_movie_gaussian)