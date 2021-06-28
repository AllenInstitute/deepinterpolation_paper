import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from manual_noise_unit_ids import get_manual_noise_unit_ids
from ks_results import ks_results

data_directory = os.path.join('..','..','data','local_large_data','kilosort_results')

rms_original = np.load('rms_original.npy')
rms_processed = np.load('rms_processed.npy')

directories = glob.glob(os.path.join(data_directory,'*probe*'))
directories.sort()

threshold = 0.7

amplitudes_pre = []
amplitudes_post = []

snr_pre = []
snr_post = []

reliable_pre = 0
reliable_post = 0

num_units_pre = []
num_units_post = []

num_units_pre_filter = []
num_units_post_filter = []

increase = []

for directory_idx, directory in enumerate(directories):
    
    print(directory)

    original = ks_results(os.path.join(directory, 'original'), get_manual_noise_unit_ids(), offset=30)
    processed = ks_results(os.path.join(directory, 'processed'), get_manual_noise_unit_ids())
    
    R = original.labels
    original_ids = R[R.label == 'good'].sort_values(by='depth').index.values
    
    R = processed.labels
    processed_ids = R[R.label == 'good'].sort_values(by='depth').index.values
    
    num_units_pre.append(len(original_ids))
    num_units_post.append(len(processed_ids))
    
    confusion_matrix = np.load(directory + '/confusion_matrix.npy')

    max_accuracy = np.max(confusion_matrix[:,:,2],1)
    matching_ids = np.argmax(confusion_matrix[:,:,2],1)

    O = original_ids[max_accuracy > threshold]
    P = processed_ids[matching_ids[max_accuracy > threshold]]
    
    Ao = original.metrics.loc[O].amplitude.values
    Ro = rms_original[directory_idx, original.metrics.loc[O].peak_channel]
    
    Ap = processed.metrics.loc[P].amplitude.values
    Rp = rms_processed[directory_idx, processed.metrics.loc[P].peak_channel]
    
    snr_pre.extend(Ao/Ro)
    snr_post.extend(Ap/Rp)
    
    original_reliability = pd.read_csv(directory + '/original_reliability.csv', index_col=0)
    processed_reliability = pd.read_csv(directory + '/processed_reliability.csv', index_col=0)
    
    processed_qc = processed.metrics[(processed.metrics.isi_viol < 0.5) &
                                     (processed.metrics.amplitude_cutoff < 0.1) &
                                     (processed.metrics.presence_ratio > 0.95)].index.values
    original_qc = original.metrics[(original.metrics.isi_viol < 0.5) &
                                     (original.metrics.amplitude_cutoff < 0.1) &
                                     (original.metrics.presence_ratio > 0.95)].index.values
    

    original_ids = np.intersect1d(original_ids, original_qc)
    processed_ids = np.intersect1d(processed_ids, processed_qc)
    
    num_units_pre_filter.append(len(original_ids))
    num_units_post_filter.append(len(processed_ids))
    
    increase.append(len(processed_ids) / len(original_ids))
    
    amplitudes_pre.extend(original.metrics.loc[original_ids].amplitude.values)
    amplitudes_post.extend(processed.metrics.loc[processed_ids].amplitude.values)
    
    b = np.sum(original_reliability.p_value < 0.05) #/ len(original_reliability)
    a = np.sum(processed_reliability.p_value < 0.05) #/ len(processed_reliability)
    
    reliable_pre += b
    reliable_post += a
    

 # %%
    
snr_pre = np.array(snr_pre)
snr_post =np.array(snr_post)

amplitudes_pre = np.array(amplitudes_pre)
amplitudes_post =np.array(amplitudes_post)
    
print(str(np.mean(increase)) + ' +/- ' + str(np.std(increase)))

print(str(np.mean(snr_pre)) + ' --> ' + str(np.mean(snr_post)))

cut_point = 75

print(str(np.sum(amplitudes_pre >= cut_point)) + ' --> ' + str(np.sum(amplitudes_post >= cut_point)))
print(str(np.sum((amplitudes_pre < cut_point) & (amplitudes_pre > 0))) + \
      ' --> ' + str(np.sum((amplitudes_post < cut_point) & (amplitudes_post > 0))))
    
A = np.array(num_units_post_filter) / np.array(num_units_pre_filter)

print(np.mean(A))
print(np.std(A))


#%%    
amplitudes_pre = np.array(amplitudes_pre)
amplitudes_post = np.array(amplitudes_post)

snr_pre = np.array(snr_pre)
snr_post = np.array(snr_post)
    
plt.figure(17771, figsize=(10,4))
plt.clf()

import matplotlib

matplotlib.rcParams.update({'font.size': 14})

from scipy.ndimage.filters import gaussian_filter1d

def smoothed_hist(data, bins, color, window_size=2):
    
    h,b = np.histogram(data, bins)
    
    h = gaussian_filter1d(h.astype('float'), window_size)
    plt.bar(b[:-1],h,width=np.mean(np.diff(b)), color=color, alpha=0.2)
    plt.plot(b[:-1], h, color=color, linewidth=3)
    axis = plt.gca()    
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    plt.xlim([np.min(bins), np.max(bins)])

plt.subplot(1,3,1)

smoothed_hist(amplitudes_pre[amplitudes_pre > 0], bins=np.arange(0,500,5), color='gray')
smoothed_hist(amplitudes_post[amplitudes_post > 0], bins=np.arange(0,500,5), color='darkred')
plt.title('N_pre = ' + str(np.sum(amplitudes_pre > 0)) + 
          '; N_post = ' + str(np.sum(amplitudes_post > 0)) , fontsize=9)
plt.xlabel('Amplitude')
    
plt.subplot(1,3,2)

smoothed_hist(snr_pre[snr_pre > 0.5], bins=np.arange(0,55,1), color='gray') #, alpha=0.5)
smoothed_hist(snr_post[snr_post > 0.5], bins=np.arange(0,55,1), color='darkred') #, alpha=0.5)
plt.xlabel('SNR')

plt.title('N_pre = ' + str(np.sum(snr_pre > 0.5)) + 
          '; N_post = ' + str(np.sum(snr_post > 0.5)) , fontsize=9)

plt.subplot(1,3,3)
plt.bar(0, reliable_pre, color='gray')
plt.bar(1, reliable_post, color='darkred')
plt.xlim([-1,2])
plt.tight_layout()

plt.show()