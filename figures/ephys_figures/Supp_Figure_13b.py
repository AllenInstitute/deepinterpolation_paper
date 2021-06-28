import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

from manual_noise_unit_ids import get_manual_noise_unit_ids
from ks_results import ks_results

data_directory = os.path.join('..','..','data','local_large_data','kilosort_results')

directories = glob.glob(os.path.join(data_directory, '*probe*'))

# %%

df_original = []
df_processed = []

for directory_idx, directory in enumerate(directories):
    
    session_id = int(os.path.basename(directory).split('_')[0])
    probe_name = os.path.basename(directory).split('_')[3]
    
    print(session_id)
    print(directory)

    original = ks_results(os.path.join(directory, 'original'),get_manual_noise_unit_ids(), offset=30)
    processed = ks_results(os.path.join(directory, 'processed'), get_manual_noise_unit_ids())
    
    processed_ids = processed.labels[(processed.labels.label == 'good')].index.values
    original_ids = original.labels[(original.labels.label == 'good')].index.values
    
    df_original.append(original.metrics.loc[original_ids])
    df_processed.append(processed.metrics.loc[processed_ids])


# %%    
   
df_original = pd.concat(df_original)
df_processed = pd.concat(df_processed)
    
# %%
    
plt.figure(1988, figsize=(9,5))
plt.clf()

Npoints = 50

plt.subplot(1,3,1)

isi_cut = np.logspace(-4,1,Npoints)

data = [np.sum(df_original.isi_viol <= isi) for isi in isi_cut]
plt.plot(isi_cut, data, 'darkgrey')

data = [np.sum(df_processed.isi_viol <= isi) for isi in isi_cut]
plt.semilogx(isi_cut, data, 'darkred')
plt.ylabel("Units passing quality threshold")

plt.ylim([0,6000])
axis = plt.gca()    
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
plt.xticks(ticks=[1e-3, 1e-1, 1e1], labels=['0.001', '0.1', '10'])
plt.xlabel('ISI violations score')

plt.plot([0.5,0.5],[0,6000],'--k')

plt.subplot(1,3,2)

amp_cut = np.logspace(-4,np.log(0.6),Npoints)

data = [np.sum(df_original.amplitude_cutoff <= amp) for amp in amp_cut]
plt.plot(amp_cut, data, 'darkgrey')

data = [np.sum(df_processed.amplitude_cutoff <= amp) for amp in amp_cut]
plt.semilogx(amp_cut, data, 'darkred')

plt.ylim([0,6000])
axis = plt.gca()    
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
plt.xticks(ticks=[1e-4, 1e-3, 1e-2, 1e-1], labels=['0.0001', '0.001', '0.01', '0.1'])
plt.xlabel('Amplitude cutoff')
plt.plot([0.1,0.1],[0,6000],'--k')

plt.subplot(1,3,3)

pres_rat = np.linspace(0,0.99,Npoints)

data = [np.sum(df_original.presence_ratio >= pr) for pr in pres_rat]
plt.plot(pres_rat, data, 'darkgrey')

data = [np.sum(df_processed.presence_ratio >= pr) for pr in pres_rat]
plt.plot(pres_rat, data, 'darkred')

plt.ylim([0,6000])
axis = plt.gca()    
[axis.spines[loc].set_visible(False) for loc in ['top', 'right']] 
plt.plot([0.95,0.95],[0,6000],'--k')

plt.xlabel('Presence ratio')

plt.tight_layout()

plt.show()

# %%

