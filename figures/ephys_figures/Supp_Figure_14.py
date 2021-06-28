import numpy as np
import os
import glob

import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from manual_noise_unit_ids import get_manual_noise_unit_ids
from ks_results import ks_results

data_directory = os.path.join('..','..','data','local_large_data','kilosort_results')

directories = glob.glob(os.path.join(data_directory, '*probe*'))

cache_dir = '/mnt/nvme0/ecephys_cache_dir' 

manifest_path = os.path.join(cache_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

import warnings
warnings.filterwarnings("ignore")

sessions = cache.get_session_table()

# %%


from scipy.ndimage.filters import gaussian_filter1d

def smoothed_hist(data, bins, color, window_size=2, vertical=False):
    
    h,b = np.histogram(data, bins)
    
    h = gaussian_filter1d(h.astype('float'), window_size)
    
    if vertical:
        plt.barh(b[:-1],h,height=np.mean(np.diff(b)), color=color, alpha=0.2)
        plt.plot(h, b[:-1], color=color, linewidth=3)
        plt.ylim([np.min(bins), np.max(bins)])
    else:
       plt.bar(b[:-1],h,width=np.mean(np.diff(b)), color=color, alpha=0.2)
       plt.plot(b[:-1], h, color=color, linewidth=3)
       plt.xlim([np.min(bins), np.max(bins)])
        
    
    axis = plt.gca()    
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    
    
    
    
def plot_boundaries(channels, vertical_extent=15, vertical=False):
    sorted_channels = channels.sort_values(by='probe_vertical_position')
    
    boundaries = np.where((sorted_channels.ecephys_structure_acronym.values[1:] ==
                          sorted_channels.ecephys_structure_acronym.values[:-1]) == False)[0]
    
    
    valid_structures = np.where(np.diff(boundaries) > 1)
    
    valid_boundaries = sorted_channels.index.values[boundaries[valid_structures] + 1]

    top_of_ctx = np.max(np.where(sorted_channels.ecephys_structure_acronym.str.find('VIS') == 0)[0])


    valid_boundaries = np.concatenate((valid_boundaries,[sorted_channels.index.values[top_of_ctx]]))
    
    for idx, bound in enumerate(valid_boundaries):
        
        if idx > 0:
            middle = np.mean([valid_boundaries[idx-1], valid_boundaries[idx]]).astype('int')
        else:
            middle = (valid_boundaries[idx] / 2).astype('int')
        structure = sorted_channels.iloc[np.searchsorted(sorted_channels.index.values, middle)].ecephys_structure_acronym
        
        if vertical:
            plt.plot([0,vertical_extent],[bound, bound],'--k')
            plt.text(vertical_extent*0.66, middle-10, structure)
        else:
            plt.plot([bound,bound],[0,vertical_extent],'--k')
            plt.text(middle-10, vertical_extent*0.66, structure)

# %%

plt.figure(199, figsize=(20,5))
plt.clf()

area_groups = {'cortex' : ['VISp','VISrl','VIS','VISal','VISl'],
               'CA1' : ['CA1'],
               'CA3': ['CA3'],
               'DG' : ['DG'],
               'subiculum' : ['PRE', 'POST', 'SUB', 'ProS'],
               'thalamus' : ['LP', 'LGd', 'LGv', 'IGL', 'POL'],
               'midbrain' : ['APN', 'MRN', 'SGN']}

statistics = {}

for key in area_groups.keys():
    
    statistics[key] = {'processed' : {'firing_rate' : [],
              'waveform_amplitude' : [],
              'waveform_duration': [],
              'waveform_spread' : []},
                       'original': {'firing_rate' : [],
              'waveform_amplitude' : [],
              'waveform_duration': [],
              'waveform_spread' : []}}


for directory_idx, directory in enumerate(directories):
    
    session_id = int(os.path.basename(directory).split('_')[0])
    probe_name = os.path.basename(directory).split('_')[3]
    
    print(session_id)
    print(directory)
    
    session = cache.get_session_data(session_id)
    
    probe_id = session.probes[session.probes.description == probe_name].index.values[0]
    
    original = ks_results(os.path.join(directory, 'original'),get_manual_noise_unit_ids(), offset=30)
    processed = ks_results(os.path.join(directory, 'processed'), get_manual_noise_unit_ids())

    sampling_rate = session.probes.loc[probe_id].sampling_rate
    
    channels = session.channels[session.channels.probe_id == probe_id]
    channels = channels.set_index(channels.local_index)
    
    plt.subplot(1,10,directory_idx+1)
    
    processed_ids = processed.labels[(processed.labels.label == 'good')].index.values
    original_ids = original.labels[(original.labels.label == 'good')].index.values
    
    processed_qc = processed.metrics[(processed.metrics.isi_viol < 0.5) &
                                     (processed.metrics.amplitude_cutoff < 0.1) &
                                     (processed.metrics.presence_ratio > 0.95)].index.values
    original_qc = original.metrics[(original.metrics.isi_viol < 0.5) &
                                     (original.metrics.amplitude_cutoff < 0.1) &
                                     (original.metrics.presence_ratio > 0.95)].index.values
    
    original_ids = np.intersect1d(original_ids, original_qc)
    processed_ids = np.intersect1d(processed_ids, processed_qc)
    
    smoothed_hist(original.metrics.loc[original_ids].peak_channel,
                  np.arange(0,384,2),
                  'darkgrey', vertical=True)
    smoothed_hist(processed.metrics.loc[processed_ids].peak_channel,
                  np.arange(0,384,2),
                  'darkred', vertical=True)

    plot_boundaries(channels, 5, vertical=True)
    
    plt.title('Original: ' + str(len(original_ids)) + '\nDenoised: ' + str(len(processed_ids)))

    for area_group in area_groups.keys():
        
        for area in area_groups[area_group]:
            
            valid_channels = channels[channels.ecephys_structure_acronym == area].index.values
            
            if len(valid_channels) > 0:
                
                for T in ('original', 'processed'):
                    
                    if T == 'original':
                        df = original.metrics.loc[original_ids]
                    else:
                        df = processed.metrics.loc[processed_ids]
                        
                    sub_df = df[df.peak_channel.isin(valid_channels)]

                    statistics[area_group][T]['firing_rate'].extend(list(sub_df.firing_rate.values))
                    statistics[area_group][T]['waveform_amplitude'].extend(list(sub_df.amplitude.values))
                    statistics[area_group][T]['waveform_spread'].extend(list(sub_df.spread.values))
                    statistics[area_group][T]['waveform_duration'].extend(list(sub_df.duration.values))

plt.show()

    # %%

plt.figure(1111, figsize=(13,20))
plt.clf()

bins = [np.arange(-1,2,0.05),
        np.arange(0,500,10),
        np.arange(0,1,0.02),
        np.arange(0,250,5)]
    
for area_idx, area_group in enumerate(area_groups.keys()):
    
    plt.subplot(7,5,area_idx*5+1)
    plt.bar(0, len(statistics[area_group]['original']['firing_rate']),color='darkgrey')
    plt.bar(1, len(statistics[area_group]['processed']['firing_rate']),color='darkred')
    plt.xlim([-1,2])
    plt.ylabel(area_group)
    axis = plt.gca()    
    [axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
    plt.title('Yield')
    
    for stat_idx, stat in enumerate(['firing_rate', 
                                     'waveform_amplitude',
                                     'waveform_duration',
                                     'waveform_spread']):
        
        plt.subplot(7,5,area_idx*5+stat_idx+2)
        
        if stat_idx == 0:
            s_proc = np.log10(statistics[area_group]['processed'][stat])
            s_orig = np.log10(statistics[area_group]['original'][stat])
        else:
            s_proc = statistics[area_group]['processed'][stat]
            s_orig = statistics[area_group]['original'][stat]

        smoothed_hist(s_proc,
                      bins=bins[stat_idx],
                      color='darkred')
        
        
        smoothed_hist(s_orig,
                      bins=bins[stat_idx],
                      color='darkgrey')
        
        if area_idx == 0:
            plt.title(stat)
            
plt.tight_layout()

plt.show()