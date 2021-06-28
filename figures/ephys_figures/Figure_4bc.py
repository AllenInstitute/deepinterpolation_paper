import numpy as np 
import matplotlib.pyplot as plt 
import os

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

data_directory = os.path.join('..','..','data','local_large_data','rms_calculation')


pre_data = np.load(os.path.join(data_directory, 'original_example.npy'))
post_data = np.load(os.path.join(data_directory, 'processed_example.npy'))

# %%

plt.figure(1, figsize=(15,5))
plt.clf()

num_samples = 1000

def plot_snip(data, num_samples, axis_index):
    
    ax = plt.subplot(1,3,axis_index)
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
    
for (data, axis_index) in zip((pre_data, post_data), (1, 2)):
    
    plot_snip(data, num_samples, axis_index)

plt.subplot(1,3,3)
plt.plot(pre_data[32,:num_samples] , color='darkgrey')
plt.plot(post_data[32,:num_samples] , color='darkred')

plt.plot(pre_data[160,:num_samples] + 200, color='darkgrey')
plt.plot(post_data[160,:num_samples] + 200, color='darkred')

plt.plot(pre_data[237,:num_samples] + 400, color='darkgrey')
plt.plot(post_data[237,:num_samples] +400, color='darkred')

plt.show()

# %%