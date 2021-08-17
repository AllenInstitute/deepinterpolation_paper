#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:19:32 2021

@author: joshs
"""

data_directory = '/mnt/hdd0/deep_interpolation_paper/data/local_large_data/simulated_ground_truth'

original = np.memmap(os.path.join(data_directory, 'continuous_sim.dat'), dtype='int16')
original = np.reshape(original, (original.size//384, 384))

processed = np.memmap(os.path.join(data_directory, 'continuous_processed.dat'), dtype='int16')
processed = np.reshape(processed, (processed.size//384, 384))

scaling = 10

offset = 30

samples = np.load(os.path.join(data_directory, 'samples.npy'))
channel_offsets = np.load(os.path.join(data_directory, 'channel_offsets.npy'))
amps = np.load(os.path.join(data_directory, 'amps.npy'))
units = np.load(os.path.join(data_directory, 'units.npy'))

waveforms = np.load(os.path.join(data_directory, 'mean_waveforms.npy'))


# %%

xlocs = [0, 2, 1, 3] * 96
ylocs = (np.arange(384)/2).astype('int')

scaling = 10
        
def plot_waveform(waveform, color='k'):

    wv = waveforms[unit_id, :, :] / 2
        
    peak_channel = np.argmax(np.max(wv, 1) - 
                             np.min(wv,1))
    
    low_channel = np.max([peak_channel-16,0])
    high_channel = np.min([peak_channel+16,383])
    
    for i in range(low_channel, high_channel):
        
        channel_data = wv[i,:] * 0.5
        x = np.linspace(0,1,len(channel_data)) + xlocs[i]
        y = channel_data + ylocs[i] * scaling
        plt.plot(x,y,color=color)
        
    plt.ylim([ylocs[low_channel] * scaling-20, ylocs[low_channel] * scaling+180])
    
    # %%
    
units_to_use = [29,  43,  191, 220, 230, 400, 428, 594, 548, 589,
               107, 153, 154, 164, 257, 361, 450, 570, 612, 90]

def get_scaled_waveform(waveforms, unit_id):

    wv = waveforms[unit_id, :, :]
        
    peak_channel = np.argmax(np.max(wv, 1) - 
                             np.min(wv,1))
    
    peak_amplitude = np.max(np.max(wv, 1) - np.min(wv,1))
    
    low_channel = np.max([peak_channel-16,0])
    high_channel = np.min([peak_channel+16,383])
    
    channels = np.arange(low_channel, high_channel)
    
    return wv[channels,:] / peak_amplitude, channels
    

# %%
    
def plot_wv(wv, color='k'):

    for i in range(wv.shape[0]):
        
        channel_data = wv[i,:] * 0.5
        x = np.linspace(0,1,len(channel_data)) + xlocs[i]
        y = channel_data + ylocs[i] * 20
        plt.plot(x,y,color=color)

# %%
    
amp = 200
sample_buffer = 300
channel_buffer = 32

inds = np.where((units == 164) * (amps == amp))[0]

spike_idx = inds[16]

s = samples[spike_idx] #+ 512
c = channel_offsets[spike_idx]
a = amps[spike_idx]
u = units[spike_idx]

original_waveform, channels = get_scaled_waveform(waveforms, u)
    
c = c + channels[0] % 4

sample_range = np.arange(s-sample_buffer, s+sample_buffer)
channel_range = np.arange(c-channel_buffer,c+channel_buffer*2)

original_snip = original[s-sample_buffer:s+sample_buffer, c-channel_buffer:c+channel_buffer*2] * 0.195
processed_snip = processed[s-sample_buffer-offset:s+sample_buffer-offset, c-channel_buffer:c+channel_buffer*2] * 0.195
osnip = np.copy(original_snip)

plt.figure(19911)
plt.clf()
plt.subplot(1,4,2)
plt.imshow(original_snip.T, origin='lower', aspect='auto', vmin=-35, vmax=35, cmap='RdGy')
plt.subplot(1,4,3)
plt.imshow(processed_snip.T, origin='lower', aspect='auto', vmin=-35, vmax=35, cmap='RdGy')
plt.subplot(1,4,1)
osnip[sample_buffer:sample_buffer+82,channel_buffer:channel_buffer+len(channels)] = \
    osnip[sample_buffer:sample_buffer+82,channel_buffer:channel_buffer+len(channels)]  - original_waveform.T * amp
plt.imshow(osnip.T, origin='lower', aspect='auto', vmin=-35, vmax=35, cmap='RdGy')
plt.subplot(1,4,4)
plot_wv(original_waveform * a)
snip = original[s:s+80,c:c+len(channels)].T * 0.195
plot_wv(snip, color='gray')
s -= offset
snip = processed[s:s+80,c:c+len(channels)].T * 0.195
plot_wv(snip, color='darkred')


# %%


all_unit_ids = np.unique(units)

all_amplitudes = np.unique(amps)

num_units = len(all_unit_ids)
num_amps = len(all_amplitudes)

wv_clean = np.zeros((32, 82, num_units, num_amps))
wv_original = np.zeros((32, 82,  num_units, num_amps, 100))
wv_processed = np.zeros((32, 82, num_units, num_amps, 100))

# %%

for unit_idx, unit_id in enumerate(all_unit_ids):
    
    print(unit_id)
    
    for amp_idx, amp in enumerate(all_amplitudes):
        
        print('  amp: ' + str(amp))
        
        wv, channels = get_scaled_waveform(waveforms, unit_id)
        
        wv_clean[:len(channels),:, unit_idx, amp_idx] = wv * amp
        
        inds = np.where((amps == amp) * (units == unit_id))[0]
        
        for idx, spike_idx in enumerate(inds):
            
            s = samples[spike_idx] #+ 512
            c = channel_offsets[spike_idx]
            a = amps[spike_idx]
            u = units[spike_idx]

            c = c + channels[0] % 4
            
            snip = original[s:s+82,c:c+len(channels)].T * 0.195
            
            try:
                wv_original[:len(channels),:, unit_idx, amp_idx, idx] = snip
            except ValueError:
               # print('Spike ' + str(idx) + ' not found in original')
                wv_original[:, :, unit_idx, amp_idx, idx] = np.nan

            s -= offset
            
            snip = processed[s:s+82,c:c+len(channels)].T * 0.195
            
            try:
                wv_processed[:len(channels), :, unit_idx, amp_idx, idx] = snip
            except ValueError:
                ##print('Spike ' + str(idx) + ' not found in processed')
                wv_processed[:, :, unit_idx, amp_idx, idx] = np.nan

# %%

# example waveforms
                
plt.figure(122)  
plt.clf()

IDX = -1

for unit_idx, unit_id in zip([6,0,4],[164,29,153]):
    
    IDX += 1

    mean_wv = wv_clean[:,:, unit_idx, -1]
    
    peak_chan = np.argmax(np.max(mean_wv,1) - np.min(mean_wv,1))        
    
    t = np.linspace(0,82/30000,82)
    for amp_idx, amp in enumerate(all_amplitudes[:10]): 
    
        plt.subplot(3,10,amp_idx+1+IDX*10)
        plt.plot(t,wv_clean[peak_chan, :, unit_idx, amp_idx], color='black')
        plt.plot(t,np.nanmean(wv_original[peak_chan, :, unit_idx, amp_idx, :],1), color='gray')
        plt.plot(t,np.nanmean(wv_processed[peak_chan, :, unit_idx, amp_idx, :],1), color='darkred')
        
        if (unit_idx == 0):
            plt.title(str(int(amp)) + ' uV')
                
                # %%
    
from sklearn.metrics.pairwise import cosine_similarity
    
# amplitude smoothing 
            
import matplotlib

cmap = matplotlib.cm.get_cmap('terrain')

            
plt.figure(11771)
plt.clf()

plt.subplot(1,2,1)
            
for amp_idx, amp in enumerate(all_amplitudes):

    for unit_idx, unit_id in enumerate(all_unit_ids):
        
        mean_wv = wv_clean[:,:, unit_idx, -1]
    
        peak_chan = np.argmax(np.max(mean_wv,1) - np.min(mean_wv,1))  
        peak_wv = mean_wv[peak_chan,:]

        selected_waveforms = wv_processed[peak_chan,:,unit_idx, amp_idx,:]

        peak_denoised = np.nanmean(selected_waveforms,1)
        
        kernel_matrix = cosine_similarity(peak_wv[:80].reshape(-1,1).T, 
                                          selected_waveforms[:80,:-10].T)
        
        plt.scatter(amp + unit_idx * 0.5, np.nanmean(kernel_matrix), s=3, c = cmap(unit_idx / 25))
        
plt.xlabel('Original amplitude (uV)')

plt.ylabel('Mean cosine similarity')                

plt.subplot(1,2,2)
        
# amplitude by fwhm
        
amp_idx = 4 # 100 uV
        
for unit_idx, unit_id in enumerate(all_unit_ids):
        
    mean_wv = wv_clean[:,:, unit_idx, -1]

    peak_chan = np.argmax(np.max(mean_wv,1) - np.min(mean_wv,1))  
    
    wv = wv_clean[peak_chan, :, unit_idx, amp_idx]
    
    trough_amp = np.min(wv)
    
    fwhm = np.sum(wv < trough_amp/2)

    selected_waveforms = wv_processed[peak_chan,:,unit_idx, amp_idx, :]

    denoised_amps = np.max(selected_waveforms, 0) - np.min(selected_waveforms, 0)   
    
    xvals = np.ones(denoised_amps.shape) * fwhm + np.random.rand(len(denoised_amps)) * 0.5 - 0.25
    plt.scatter(xvals / 30000 * 1000, denoised_amps, s=1, c=cmap(unit_idx/25))
    
    
plt.plot([0.06,0.27],[100,100],'--k')
plt.ylim([0,200])
plt.xlabel('FWHM (ms)')
plt.ylabel('Denoised amplitude')
    
    
    # %%