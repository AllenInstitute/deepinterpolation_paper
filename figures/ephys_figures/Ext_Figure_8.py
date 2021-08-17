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

file_idx = 7

data = np.load(files['original'][file_idx])
data_post = np.load(files['processed'][file_idx])

# %%

from scipy.ndimage import gaussian_filter, convolve

plt.figure(199)
plt.clf()

num_samples = 3000

sigma = 1.0
boxcar_size = 4

channels = np.arange(350)
channels = np.delete(channels, 141)
channels= np.delete(channels, 168)

original = data[60:num_samples+60,channels]
gaussian = gaussian_filter(original, sigma)
boxcar = convolve(original, np.ones((boxcar_size,boxcar_size)) / pow(boxcar_size,2))
deepinterp = data_post[:num_samples,channels]

limit = 25
plt.subplot(1,4,1)
plt.imshow(original.T, 
    				origin='lower', 
    				vmin=-limit, 
    				vmax=limit, 
    				cmap='RdGy',
    				aspect='auto')

plt.title(' - '.join(os.path.basename(files['original'][file_idx]).split('_')[1:4]))

plt.subplot(1,4,2)
plt.imshow(gaussian.T, 
    				origin='lower', 
    				vmin=-limit, 
    				vmax=limit, 
    				cmap='RdGy',
    				aspect='auto')

plt.subplot(1,4,3)
plt.imshow(boxcar.T, 
    				origin='lower', 
    				vmin=-limit, 
    				vmax=limit, 
    				cmap='RdGy',
    				aspect='auto')


plt.subplot(1,4,4)
plt.imshow(deepinterp.T, 
    				origin='lower', 
    				vmin=-limit, 
    				vmax=limit, 
    				cmap='RdGy',
    				aspect='auto')

# %%

def rms_for_channel(data, threshold):

	above_thresh = np.where(np.abs(data) > threshold)[0]
	masked_data = np.delete(data, above_thresh)
	return rms(masked_data)

def calculate_rms(data, threshold):

	rms_values = []

	for ch in np.arange(data.shape[1]):

		rms_values.append(rms_for_channel(data[:, ch], threshold))

	return rms_values

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

# %%

from scipy.ndimage.filters import gaussian_filter1d

plt.figure(200)
plt.clf()

plt.subplot(1,4,1)
rms_values = calculate_rms(original, 50)
plot_histogram(rms_values, bins, 'orange', filter_window)
M = np.mean(rms_values)
plt.plot([M, M],[0,0.13],'--',color='grey')

plt.subplot(1,4,2)
rms_values = calculate_rms(gaussian, 50)
plot_histogram(rms_values, bins, 'orange', filter_window)
M = np.mean(rms_values)
plt.plot([M, M],[0,0.13],'--',color='grey')

plt.subplot(1,4,3)
rms_values = calculate_rms(boxcar, 50)
plot_histogram(rms_values, bins, 'orange', filter_window)
M = np.mean(rms_values)
plt.plot([M, M],[0,0.13],'--',color='grey')

plt.subplot(1,4,4)
rms_values = calculate_rms(deepinterp, 50)
plot_histogram(rms_values, bins, 'orange', filter_window)
M = np.mean(rms_values)
plt.plot([M, M],[0,0.13],'--',color='grey')


# %%
from scipy.signal import find_peaks

plt.figure(201)
plt.clf()
offset = 200
channel = 243
threshold = 100
lim = -300

mask = np.arange(290000)

sigma = 1.0
boxcar_size = 4

original = data[60+mask,:350]
gaussian = gaussian_filter(original, sigma)
boxcar = convolve(original, np.ones((boxcar_size,boxcar_size)) / pow(boxcar_size,2))
deepinterp = data_post[mask,:350]

peaks, properties = find_peaks(-original[:, channel], height=threshold)

plt.subplot(1,5,1)
plt.plot(original[:,channel] ,color='darkslategrey', linewidth=0.5)
plt.plot(peaks, original[peaks, channel],'.', color='black')
plt.ylim([lim,150])
plt.xlim([40000,55000])

plt.subplot(1,5,2)
plt.plot(gaussian[:,channel],color='darkslategrey', linewidth=0.5)
plt.plot(peaks, gaussian[peaks,channel],'.', color='teal')
plt.ylim([lim,150])
plt.xlim([40000,55000])

plt.subplot(1,5,3)
plt.plot(boxcar[:,channel],color='darkslategrey', linewidth=0.5)
plt.plot(peaks, boxcar[peaks,channel],'.', color='purple')
plt.ylim([lim,150])
plt.xlim([40000,55000])

di_peaks = []

for i in peaks:
    snip = deepinterp[i-3:i+3,channel]
    di_peaks.append(np.min(snip))
    
di_peaks = np.array(di_peaks)

plt.subplot(1,5,4)
plt.plot(deepinterp[:,channel],color='darkslategrey', linewidth=0.5)
plt.plot(peaks, di_peaks,'.', color='darkred')
plt.ylim([lim,150])
plt.xlim([40000,55000])

plt.subplot(1,5,5)
plt.plot(-original[peaks, channel], -gaussian[peaks, channel], '.', color='teal', alpha=0.75, markersize=3)
plt.plot(-original[peaks, channel], -boxcar[peaks, channel], '.', color='purple', alpha=0.75, markersize=3)
plt.plot(-original[peaks, channel], -di_peaks, '.', color='darkred', alpha=0.75, markersize=3)
plt.plot([0,250],[0,250],'--', color='darkslategrey')


# %%

