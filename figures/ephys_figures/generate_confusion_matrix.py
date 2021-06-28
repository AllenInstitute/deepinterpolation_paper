import numpy as np
import os
import glob

from manual_noise_unit_ids import get_manual_noise_unit_ids
from ks_results import ks_results

data_directory = os.path.join('..','..','data','local_large_data','kilosort_results')

directories = glob.glob(os.path.join(data_directory,'*probe*'))
directories.sort()


# %%

def calculate_metrics(t_gt, t_k, thresh=0.0005):
    
    n_gt = len(t_gt)
    n_k = len(t_k)
    
    nearest_inds = np.searchsorted(t_k, t_gt)
    
    t_gt_prime = t_gt[nearest_inds < n_k]
    t_k_nearest = t_k[nearest_inds[nearest_inds < n_k]]
    
    n_k_match = np.sum(np.abs(t_k_nearest - t_gt_prime) <= thresh)
    
    precision = n_k_match / n_k
    recall = n_k_match / n_gt
    accuracy = n_k_match / (n_k + n_gt - n_k_match)
    
    return precision, recall, accuracy

# %%

for directory_idx, directory in enumerate(directories):
    
    print(directory)

    original = ks_results(os.path.join(directory, 'original'), get_manual_noise_unit_ids(), offset=30)
    processed = ks_results(os.path.join(directory, 'processed'), get_manual_noise_unit_ids())
    
    R = original.labels
    original_ids = R[R.label == 'good'].sort_values(by='depth').index.values
    
    R = processed.labels
    processed_ids = R[R.label == 'good'].sort_values(by='depth').index.values
 
    confusion_matrix  = np.zeros((len(original_ids), len(processed_ids), 3))
    
    print(confusion_matrix.shape)
    
    print('Generating confusion matrix')

    for idx1, unit_id1 in enumerate(original_ids):
        
        for_unit1 = original.clusters == unit_id1
        spike_times1 = original.times[for_unit1]
        depth1 = original.labels.loc[unit_id1].depth
        
        for idx2, unit_id2 in enumerate(processed_ids):
            
            depth2 = processed.labels.loc[unit_id2].depth
            
            if np.abs(depth1 - depth2) <= 4:
            
                for_unit2  = processed.clusters == unit_id2
                spike_times2 = processed.times[for_unit2]
            
                precision, recall, accuracy = calculate_metrics(spike_times1, 
                                                            spike_times2)
            
                confusion_matrix[idx1, idx2, 0] = precision
                confusion_matrix[idx1, idx2, 1] = recall
                confusion_matrix[idx1, idx2, 2] = accuracy
            
    np.save(os.path.join(directory + '/confusion_matrix.npy'),
            confusion_matrix)
    print('Saved data.')

 
