import numpy as np 
import matplotlib.pyplot as plt 

import os

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

data_directory = os.path.join('..','..','data','local_large_data','example_ephys_data')

pre_data = np.load(os.path.join(data_directory, 'original_example.npy'))
post_data = np.load(os.path.join(data_directory, 'processed_example.npy'))

# %%

plt.figure(2, figsize=(9,5))

from scipy.signal import welch

num_samples = 100000
start_pt = 30000 * 600 + 5500

residual = post_data - pre_data

def plot_snip(data, num_samples, axis_index):
    
    ax = plt.subplot(1,2,axis_index)
    plt.imshow(data[:,:num_samples], 
    				origin='lower', 
    				vmin=-50, 
    				vmax=50, 
    				cmap='RdGy',
    				aspect='auto')
    
    plt.colorbar()
    
    pc = PatchCollection([mpatches.Rectangle((490,222), 110, 28)], 
                         facecolor='none', 
                         edgecolor='k')
    
    ax.add_collection(pc)
    
    axins = inset_axes(ax, 1.0, 1.0, loc='lower right')
    
    axins.imshow(data[222:250,490:600], 
                      origin="lower",
                      vmin=-50,
                      vmax=50,
                      cmap='RdGy',
                      aspect='auto')
    
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    
plot_snip(residual, 1000, 1)


def plot_fft(data, color='gray', nfft=2048):

	psd = np.zeros((data.shape[0], nfft//2+1))

	for i in range(data.shape[0]):

		f, psd[i,:] = welch(data[i,:], fs=30000., nfft=nfft)

	plt.semilogx(f, np.mean(psd, 0), color=color)


plt.subplot(1,2,2)
plot_fft(pre_data, color='darkgrey')
plot_fft(post_data, color='darkred')
plot_fft(residual, color='orange')

plt.xlim([50,15000])
plt.xticks([100,1000,10000],
        labels=['100 Hz','1 kHz','10 kHz'])

plt.show()