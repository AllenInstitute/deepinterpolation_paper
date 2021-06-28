import numpy as np 
import matplotlib.pyplot as plt 

import os
import glob

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from ecephys_spike_sorting.common.utils import rms

data_directory = os.path.join('..','..','data','local_large_data','rms_calculation')

files = {}

files['original'] = glob.glob(os.path.join(data_directory, '*original*')) 
files['original'].sort()

files['processed'] = glob.glob(os.path.join(data_directory, '*processed*')) 
files['processed'].sort()

# %%

file_types = ['original', 'processed']
threshold = 50

results = {}
results['original'] = np.zeros((10, 374))
results['processed'] = np.zeros((10, 374))

def rms_for_channel(data, threshold):

	above_thresh = np.where(np.abs(data) > threshold)[0]
	masked_data = np.delete(data, above_thresh)
	return rms(masked_data)

def calculate_rms(data, threshold):

	rms_values = []

	for ch in np.arange(data.shape[1]):

		rms_values.append(rms_for_channel(data[:, ch], threshold))

	return rms_values

for file_type in file_types:

    print(file_type)
    
    for file_path_idx, file_path in enumerate(files[file_type]):
        
        print(file_path)

        data = np.load(file_path)

        rms_values = calculate_rms(data, threshold)

        results[file_type][file_path_idx, :] = np.array(rms_values)

# %%
fold_decrease = np.median(results['original']) / np.median(results['processed'])
median_after_denoising = np.median(results['processed'])
        
print(str(np.around(fold_decrease,1)) + '-fold decrease after denoising')

print('Median RMS after denoising: '+ str(np.around(median_after_denoising, 2)) + ' uV')

# %%

plt.figure(1711, figsize=(8,5))
plt.clf()

from scipy.ndimage.filters import gaussian_filter1d

filter_window = 2
bins = np.linspace(0,30,50)

def plot_histogram(data, bins, color, filter_window):
    
    h, b, = np.histogram(data, 
                             bins=bins, 
                             density=True)
        
    plt.bar(b[:-1], gaussian_filter1d(h,filter_window), 
            width =np.mean(np.diff(b)),
            color=color, 
            alpha=0.2)
    
    plt.plot(b[:-1], gaussian_filter1d(h,filter_window), 
             color=color,
             linewidth=3.0)

plot_histogram(results['original'], bins, 'grey', 2)
plot_histogram(results['processed'], bins, 'darkred', 2)

plt.xlabel('RMS Noise (microvolts)')
plt.ylabel('Density')
plt.tight_layout()

plt.show()

# %%

np.save('rms_original.npy', results['original'])
np.save('rms_processed.npy', results['processed'])