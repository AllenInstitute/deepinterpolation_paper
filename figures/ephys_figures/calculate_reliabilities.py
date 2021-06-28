import numpy as np
import os
import pandas as pd
from scipy.stats import ks_2samp
import glob

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.stimulus_analysis import NaturalMovies

from manual_noise_unit_ids import get_manual_noise_unit_ids
from ks_results import ks_results

data_directory = os.path.join('..','..','data','local_large_data','kilosort_results')


directories = glob.glob(os.path.join(data_directory ,'*probe*'))
directories.sort()

# %%

from scipy.stats import pearsonr

def get_response_matrix(stim_table, ks_results, unit_id, num_bins=100, num_trials=25):

    response_matrix = np.zeros((num_trials, num_bins))
    
    for trial_idx, start_time in enumerate(stim_table.iloc[30:55].start_time):
        
        selection = ks_results.clusters == unit_id
        times = ks_results.spike_times[selection]
        
        in_range = (times > (start_time)) & (times < (start_time + 30))
        t = ((times[in_range] - start_time) * (num_bins / 30)).astype('int')
        response_matrix[trial_idx, :], b = np.histogram(t, bins=np.arange(num_bins+1))
        
    return response_matrix

def reliability(response_matrix):
    
    num_trials = response_matrix.shape[0]
    
    corr_matrix = np.zeros((num_trials,num_trials))

    for i in range(num_trials):
        for j in range(i+1,num_trials):
            r,p = pearsonr(response_matrix[i,:], 
                              response_matrix[j,:])
            corr_matrix[i,j] = r

    inds = np.triu_indices(num_trials, k=1)
    reliability = np.nanmean(corr_matrix[inds[0],inds[1]])
    
    return reliability

def lifetime_sparseness(responses):
    """Computes the lifetime sparseness for one unit. See Olsen & Wilson 2008.

    Parameters
    ----------
    responses : array of floats
        An array of a unit's spike-counts over the duration of multiple trials within a given session

    Returns
    -------
    lifetime_sparsness : float
        The lifetime sparseness for one unit
    """
    if len(responses) <= 1:
        # Unable to calculate, return nan
        warnings.warn('responses array must contain at least two or more values to calculate.')
        return np.nan

    coeff = 1.0/len(responses)
    return (1.0 - coeff*((np.power(np.sum(responses), 2)) / (np.sum(np.power(responses, 2))))) / (1.0 - coeff)


# %%


cache_dir = '/mnt/nvme0/ecephys_cache_dir' 

manifest_path = os.path.join(cache_dir, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

import warnings
warnings.filterwarnings("ignore")

sessions = cache.get_session_table()

# %%

for directory_idx, directory in enumerate(directories):
    
    session_id = int(os.path.basename(directory).split('_')[0])
    probe_name = os.path.basename(directory).split('_')[3]
    
    print(session_id)
    print(directory)
    
    session = cache.get_session_data(session_id)
    
    probe_id = session.probes[session.probes.description == probe_name].index.values[0]
    
    print (np.unique(session.units[session.units.probe_id == probe_id].ecephys_structure_acronym))
    
    original = ks_results(os.path.join(directory, 'original'),
                          get_manual_noise_unit_ids(), 
                          offset=30)
    processed = ks_results(os.path.join(directory, 'processed'), 
                           get_manual_noise_unit_ids())

    sampling_rate = session.probes.loc[probe_id].sampling_rate
    
    channels = session.channels[session.channels.probe_id == probe_id]
    channels = channels.set_index(channels.local_index) 
    
    nm = NaturalMovies(session, stimulus_key = 'natural_movie_one_more_repeats')
    stim_table = nm.stim_table[nm.stim_table.frame == 0]
    
    start_time = stim_table.iloc[-1].start_time - 3600
    
    print(session_id)
    print(probe_id)
    print(start_time)

    original.spike_times = original.indices / sampling_rate + start_time
    
    processed.spike_times = (processed.indices + 30) / sampling_rate + start_time

    print(' ')

    original.labels['ecephys_structure_acronym'] = \
        channels.loc[original.labels.depth].ecephys_structure_acronym.values
        
    processed.labels['ecephys_structure_acronym'] = \
        channels.loc[processed.labels.depth].ecephys_structure_acronym.values
        
    in_cortex_processed = processed.labels[(processed.labels.ecephys_structure_acronym.str.match('VIS')) &
                                 (processed.labels.label == 'good')].index.values
    
    in_cortex_original = original.labels[(original.labels.ecephys_structure_acronym.str.match('VIS')) &
                                 (original.labels.label == 'good')].index.values
    
    num_bins = 300
    
    def calc_reliability_metrics(matrix):
        
        shuffled_matrix = np.zeros(matrix.shape)
        for i in range(25):
            shuffled_matrix[i,:] = matrix[i,np.random.permutation(num_bins)]
            
        responses = np.mean(matrix,0)
        shuffled_responses = np.mean(shuffled_matrix,0)
        
        R = reliability(matrix)
        stat, P = ks_2samp(responses, shuffled_responses)
        LS = lifetime_sparseness(responses)
        
        return R, P, LS
    
    RR = []
    PP = []
    LSLS = []
    
    print('Calculating processed reliabilities')
    for unit_id in in_cortex_processed:

        matrix = get_response_matrix(stim_table, processed, unit_id, num_bins)
        R, P, LS = calc_reliability_metrics(matrix)
        
        RR.append(R)
        PP.append(P)
        LSLS.append(LS)
        
    df_processed = pd.DataFrame(index=in_cortex_processed,
                           data ={'reliability' : RR,
                                  'p_value' : PP,
                                  'lifetime_sparseness' : LSLS})
        
    RR = []
    PP = []
    LSLS = []
    
    print('Calculating original reliabilities')
    for unit_id in in_cortex_original:

        matrix = get_response_matrix(stim_table, original, unit_id, num_bins)
        R, P, LS = calc_reliability_metrics(matrix)
        
        RR.append(R)
        PP.append(P)
        LSLS.append(LS)
        
    df_orig = pd.DataFrame(index=in_cortex_original,
                           data ={'reliability' : RR,
                                  'p_value' : PP,
                                  'lifetime_sparseness' : LSLS})
    
    df_orig.to_csv(directory + '/original_reliability_v2.csv')
    df_processed.to_csv(directory + '/processed_reliability_v2.csv')
        

