import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class ks_results:
    
    def __init__(self, directory, to_remove, sample_rate=30, offset=0):
        
        self._directory = directory
        
        self.indices = self.load('spike_times.npy')
        
        self.times = (self.indices - offset) / (sample_rate * 1000.0)
        self.clusters = self.load('spike_clusters.npy')
        
        self.channel_map = self.load('channel_map.npy')
        
        amplitudes = self.load('amplitudes.npy')
        
        templates = self.load('templates.npy')[:,20:,:]
        unwhitened_temps = np.zeros(templates.shape)
        unwhitening_mat = self.load('whitening_mat_inv.npy')
        
        for temp_idx in range(templates.shape[0]):
        
            unwhitened_temps[temp_idx,:,:] = \
                np.dot(np.ascontiguousarray(templates[temp_idx,:,:]),
                       np.ascontiguousarray(unwhitening_mat))
    
        self.templates = unwhitened_temps
        self.labels = self.load_table('cluster_group.tsv.v2', sep='\t').rename(columns={'group':'label'})
        
        self.labels.loc[to_remove[os.path.basename(os.path.dirname(directory))]
                    [os.path.basename(directory)]] = 'noise'
        
        self.amps = np.max(np.max(self.templates,1) - np.min(self.templates,1),1) #* 0.195

        self.amplitudes = np.squeeze(self.amps[self.clusters] * amplitudes) * 0.195
        
        self.depths = np.squeeze(self.channel_map[np.argmax(np.max(self.templates,1) - 
                                                 np.min(self.templates,1),1)])
        
        self.labels['depth'] = self.depths[self.labels.index.values]
        
        self.metrics = self.load_table('metrics.csv').set_index('cluster_id')
        self.waveforms = self.load('mean_waveforms.npy') * 0.195
        
        self.xlocs = [0, 2, 1, 3] * 96
        self.ylocs = (np.arange(384)/2).astype('int')

    def load(self, filename):
        temp = np.load(os.path.join(self._directory, filename))
        return np.squeeze(temp)
    
    def load_table(self, filename, sep=','):
        return pd.read_csv(os.path.join(self._directory, filename),
                           sep=sep,
                           index_col=0)
    
    def plot_amplitudes(self, unit_id, color='k'):
        
        for_unit = self.clusters == unit_id
        for y in np.arange(50, 400, 50):
            plt.plot([0,60],[y,y],'-k',alpha=0.25)
        plt.scatter(self.times[for_unit],
                 self.amplitudes[for_unit], s=1,c=color,
                 alpha=0.5)

        ax = plt.gca()
        [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]   
        
    def plot_isi_hist(self, unit_id, color='k'):
        
        for_unit = self.clusters == unit_id
        spike_times = self.times[for_unit]
        
        h,b = np.histogram(np.diff(spike_times),
                           bins = np.arange(0,0.03,0.001))
        
        plt.bar(b[:-1]*1000,h,width=1,color=color)
        plt.axis('off')
        
    def plot_amplitude_histogram(self, unit_id, color='k'):
        
        for_unit = self.clusters == unit_id
        h, b = np.histogram(self.amplitudes[for_unit], bins=50)
        plt.plot(h,b[:-1],color=color)

        ax = plt.gca()
        [ax.spines[loc].set_visible(False) for loc in ['right', 'top']]   
        
    def plot_waveform(self, unit_id,color='k'):
        
        wv = self.waveforms[unit_id, :, :]
        
        peak_channel = np.argmax(np.max(wv, 1) - 
                                 np.min(wv,1))
        
        low_channel = np.max([peak_channel-10,0])
        high_channel = np.min([peak_channel+10,383])
        
        for i in range(low_channel, high_channel):
            
            channel_data = wv[i,:]
            x = np.linspace(0,1,len(channel_data)) + self.xlocs[i]
            y = channel_data + self.ylocs[i] * 10
            plt.plot(x,y,color=color)
            
        plt.ylim([self.ylocs[low_channel] * 10-20, self.ylocs[low_channel] * 10+110])
 
    def plot_template(self, unit_id):
        
        peak_loc = np.searchsorted(self.channel_map, self.labels.loc[unit_id].depth)

        if peak_loc - 15 < 0:
            peak_loc -= (peak_loc-15)
            
        if peak_loc + 15 > len(self.channel_map):
            peak_loc -= (15 - (len(self.channel_map) - peak_loc))

        channel_range = np.arange(peak_loc-15,peak_loc+15)
    
        template = self.templates[unit_id,:,channel_range]
    
        plt.imshow(template ,
                       origin='lower',
                       cmap='cividis',
                       aspect='auto')
        
        plt.axis('off')

# %%