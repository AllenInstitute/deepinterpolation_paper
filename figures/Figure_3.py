import matplotlib.pylab as plt
import pandas as pd
import os
import numpy as np
import glob
from scripts.plotting_helpers import placeAxesOnGrid
import h5py
import matplotlib
import pathlib
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
      
matplotlib.rcParams.update({"font.size": 8})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['pdf.fonttype'] = 42

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.path_to_transfer_training_loss = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "transfer_training_loss"
        )

        self.path_to_transfer_training_evaluation = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "transfer_training_evaluation"
        )

        self.path_meta_data = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "2021-01-06-characterize_cross_conditions_loss.csv"
        )

        self.local_comp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_657391625.h5",
        )

        self.transfer_example_base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "fine_tuning",
            "ai93-603576132.h5",
        )

        self.transfer_example_to_self = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "fine_tuning",
            "603576132-603576132.h5",
        )

        self.transfer_example_to_other = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "fine_tuning",
            "501474098-603576132.h5",
        )

        self.np_file_loss = "transfer_val_loss.npy"

    def load_data(self):
        self.load_data_transfer_evaluation()
        self.load_meta_data()
        self.load_data_training()

        with h5py.File(self.local_comp, "r") as file_handle:
            self.raw_dat = file_handle["data_raw"]

        file_handle = h5py.File(self.transfer_example_base, "r") 
        self.transfer_example_base_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_to_self, "r") 
        self.transfer_example_to_self_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_to_other, "r") 
        self.transfer_example_to_other_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_base, "r") 
        self.transfer_example_raw_movie = file_handle["raw"]


    def load_meta_data(self):
        self.local_meta_data = pd.read_csv(self.path_meta_data)

    def load_data_training(self):
                
        folders = os.listdir(self.path_to_transfer_training_loss)

        list_improvement = {}
        list_time = {}
        changing_traces = {}
        for indiv_folder in folders:
            local_folder = os.path.join(self.path_to_transfer_training_loss, indiv_folder)
            list_files = glob.glob(os.path.join(local_folder, "*losses.png"))
            if len(list_files)>0:
                init_loss = np.load(os.path.join(local_folder,list_files[0][0:-11]+"init_val_loss.npy"))
                data_array = np.load(os.path.join(local_folder,list_files[0][0:-11]+"_val_loss.npy"))
                data_array = np.concatenate([np.array([init_loss]), data_array])
                data_array_smooth = np.convolve(data_array, 1/5*np.ones(5), 'valid')
                list_improvement[indiv_folder] = np.min((data_array)/init_loss)
                std_end_loss = np.std(data_array_smooth[-20:])
                end_loss = np.min(data_array)

                try:
                    first_time = np.argwhere(np.abs(data_array_smooth-end_loss)<0.1*(init_loss-end_loss))[0][0]
                except:
                    first_time = 0
                    
                list_time[indiv_folder] = first_time
                final_curve = (data_array)/init_loss

                local_meta = self.local_meta_data[self.local_meta_data['exp id']==int(indiv_folder)]
                if local_meta['true rig'].values[0]=='nikon' and not(local_meta['cre'].values[0]=='Cux2'):
                    if 'Same conditions' in changing_traces.keys(): 
                        changing_traces['Same conditions'] = np.vstack([changing_traces['Same conditions'], final_curve])
                    else:
                        changing_traces['Same conditions'] = final_curve

                if local_meta['true rig'].values[0]=='nikon' and local_meta['cre'].values[0]=='Vip':
                    if 'Exc. to Inh.' in changing_traces.keys(): 
                        changing_traces['Exc. to Inh.'] = np.vstack([changing_traces['Exc. to Inh.'], final_curve])
                    else:
                        changing_traces['Exc. to Inh.'] = final_curve

                if local_meta['true rig'].values[0]=='scientifica' and not(local_meta['cre'].values[0]=='Vip'):
                    if 'Changing rig' in changing_traces.keys(): 
                        changing_traces['Changing rig'] = np.vstack([changing_traces['Changing rig'], final_curve])
                    else:
                        changing_traces['Changing rig'] = final_curve
                
                if local_meta['true rig'].values[0]=='piezo':
                    if 'To new imaging protocol' in changing_traces.keys(): 
                        changing_traces['To new imaging protocol'] = np.vstack([changing_traces['To new imaging protocol'], final_curve])
                    else:
                        changing_traces['To new imaging protocol'] = final_curve

                if local_meta['cre'].values[0]=='Ntsr1':
                    if 'To layer 6' in changing_traces.keys(): 
                        changing_traces['To layer 6'] = np.vstack([changing_traces['To layer 6'], final_curve])
                    else:
                        changing_traces['To layer 6'] = final_curve

        self.changing_loss_traces = changing_traces

    def load_data_transfer_evaluation(self):
        # We get the original folder to parse
        folder_to_parse = self.path_to_transfer_training_evaluation
        list_folder = os.listdir(folder_to_parse)
        np_file = self.np_file_loss

        self.list_trained_on = []
        self.list_evaluated_on = []
        for indiv_folder in list_folder:
            folder_data = indiv_folder.split("-")
            trained_on = folder_data[0]
            evaluated_on = folder_data[1]
            if not(trained_on in self.list_trained_on):
                self.list_trained_on.append(trained_on)
            if not(evaluated_on in self.list_evaluated_on):
                self.list_evaluated_on.append(evaluated_on)

        self.list_trained_on.sort()
        self.list_evaluated_on.sort()

        self.raw_img_evaluation = np.zeros([len(self.list_trained_on), len(self.list_evaluated_on)])
        for indiv_folder in list_folder:
            full_folder = os.path.join(folder_to_parse, indiv_folder)

            transfer_val_loss = np.load(os.path.join(full_folder, np_file))
            folder_data = indiv_folder.split("-")
            trained_on = folder_data[0]
            evaluated_on = folder_data[1]
            
            index_trained = self.list_trained_on.index(trained_on)
            index_evaluated = self.list_evaluated_on.index(evaluated_on)

            self.raw_img_evaluation[index_trained, index_evaluated] = transfer_val_loss

        index_baseline = self.list_trained_on.index('Ai93')
        self.norm_img_evaluation = self.raw_img_evaluation.copy()
        for index_training in np.arange(self.raw_img_evaluation.shape[0]):
            self.norm_img_evaluation[index_training, :] = self.raw_img_evaluation[index_training, :] / self.raw_img_evaluation[index_baseline, :]

    def get_error_simulation(self, Nsamples=100, function='L1 loss', proportion_bad_frames = 0.1):

        self.mean_img = np.mean(self.raw_dat, axis=0)
        shape = self.mean_img.shape
        # We generate a poisson noise version of it
        list_peak_photons = [0.1, 0.5, 0.75, 1, 2, 3, 4, 5, 7.5, 10, 25, 50, 100, 150]
        loss_list = []
        self.img_with_poissons = []

        # We z-score to mimick our preprocessing in the neuronal net.
        z_mean_img = self.mean_img.flatten()
        #z_mean_img = z_mean_img - np.mean(z_mean_img)
        #z_mean_img = z_mean_img / np.std(z_mean_img)

        number_img_sim = Nsamples
        local_peak = list_peak_photons.copy()
        for index, PEAK in enumerate(local_peak):
            print(PEAK)
            poisson_sim = np.zeros([shape[0], shape[1], number_img_sim])
            for index_img in np.arange(number_img_sim):
                poissonNoise = (
                    np.random.poisson(
                        self.mean_img / np.max(self.mean_img.flatten()) * PEAK
                    )
                    / PEAK
                    * np.max(self.mean_img.flatten())
                )

                poisson_sim[:,:,index_img] = poissonNoise

            total_bad_frames = int(np.round(proportion_bad_frames*number_img_sim))
            poisson_sim[:,:,0:total_bad_frames]=0

            if function == 'L2 loss':
                pred_img = np.mean(poisson_sim, axis=2) 
            elif function == 'L1 loss':
                pred_img = np.median(poisson_sim, axis=2) 

            list_peak_photons[index] = np.mean(self.mean_img / np.max(self.mean_img.flatten()) * PEAK)

            # We z-score to mimick our preprocessing in the neuronal net.
            pred_img = pred_img.flatten()
            #median_img = median_img - np.mean(median_img)
            #median_img = median_img / np.std(median_img)
            loss_list.append(np.mean(100*np.abs(pred_img.flatten() - z_mean_img)/z_mean_img))

        return [list_peak_photons, loss_list] 
    
    def plot_example_fine_tuning(self, ax):
        def zoomed_in(img, ax):
            axins = zoomed_inset_axes(ax, 2, loc=1)
            axins.imshow(img, cmap='gray', clim=[vmin, vmax])
            axins.set_xlim(200, 300)
            axins.set_ylim(300, 200)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.sca(ax[3])
        plt.imshow(self.transfer_example_to_self_movie[100, :, :, 0], cmap='gray')
        vmin, vmax = plt.gci().get_clim()
        plt.axis('off')
        plt.title('Fine-tuning Ai93 model\nwith layer 6 data')
        zoomed_in(self.transfer_example_to_self_movie[100, :, :, 0], ax[3])

        plt.sca(ax[1])
        plt.imshow(self.transfer_example_base_movie[100, :, :, 0], cmap='gray')
        plt.clim(vmin = vmin, vmax = vmax) 
        plt.axis('off')
        plt.title('Using Ai93 model')
        zoomed_in(self.transfer_example_base_movie[100, :, :, 0], ax[1])

        plt.sca(ax[2])
        plt.imshow(self.transfer_example_to_other_movie[100, :, :, 0], cmap='gray')
        plt.clim(vmin = vmin, vmax = vmax) 
        plt.axis('off')
        plt.title('Fine-tuning Ai93 model\non non-layer 6 data')
        zoomed_in(self.transfer_example_to_other_movie[100, :, :, 0], ax[2])

        plt.sca(ax[0])
        plt.imshow(self.transfer_example_raw_movie[100, :, :, 0], cmap='gray')
        plt.clim(vmin = vmin, vmax = vmax) 
        plt.axis('off')
        plt.title('Raw data')

        rectangle_length = 100 * 512 / 400
        rect = matplotlib.patches.Rectangle(
            [15, 512 - 30],
            rectangle_length,
            15,
            angle=0.0,
            color="w",
        )
        plt.gca().add_patch(rect)

    def plot_loss_photon_peak(self):
        losses = ['L1 loss', 'L2 loss']
        samples = 150
        proportion_list = [0.01, 0.05, 0.1]
        for function in losses:

            for proportion_bad_frames in proportion_list:
                if function == 'L1 loss':
                    color = 'indianred'
                else:
                    color = 'cornflowerblue'  

                [list_peak_photons, loss_list] = self.get_error_simulation(Nsamples=samples, function=function, proportion_bad_frames=proportion_bad_frames)

                
                plt.plot(
                    list_peak_photons,
                    loss_list, label = function+', '+str(100*proportion_bad_frames)+'% bad frames', color = color, alpha=proportion_bad_frames*10
                )

        plt.legend(frameon=False, prop={'size': 8})
        plt.xlabel("Average photon count per pixel and dwell time", fontsize=8)
        plt.ylabel("Average absolute error (%)", fontsize=8)

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        ax = plt.gca()
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]+ ax.get_xticklabels()+ ax.get_yticklabels():
            item.set_fontsize(8)

    def plot_transfer_loss(self):

        ax = plt.gca()
        color = sns.color_palette("rocket")

        for index, indiv_trace_keys in enumerate(self.changing_loss_traces.keys()):
            local_traces = self.changing_loss_traces[indiv_trace_keys]
            mean_traces = np.mean(local_traces, axis=0)
            std_traces = np.std(local_traces, axis=0)
            ax.plot(mean_traces, label=indiv_trace_keys, color=color[index])
            ax.fill_between(np.arange(len(mean_traces)), mean_traces - std_traces, mean_traces + std_traces, color=color[index], alpha=0.4)

        plt.legend(frameon=False, prop={'size': 8}, bbox_to_anchor=(0.5, 0.4))

        plt.ylabel('training validation loss over initial loss', fontsize=8)
        plt.xlabel('training samples (k)', fontsize=8)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)


    def plot_evaluation_img(self):
        list_labels_trained_on = []

        # replace string in list with metadata    
        for indiv_id in self.list_trained_on: 
            row = self.local_meta_data[self.local_meta_data['exp id'].astype(str) == str(indiv_id)]
            try:
                row=row[['rig', 'cre', 'depth', 'area']]

                local_string = row.iloc[0,:].to_string(header=False, index=False).split('\n')        
                local_string = [indiv_str.strip() for indiv_str in local_string]
                local_string = [indiv_str.rjust(7, ' ') for indiv_str in local_string]

                local_string = ' '.join(local_string)
                list_labels_trained_on.append(local_string)
            except:
                if "Ai93" in indiv_id:
                    list_labels_trained_on.append('Broadly trained Ai93 model')
                else:
                    list_labels_trained_on.append(indiv_id)

        list_labels_evaluated_on = []
        for indiv_id in self.list_evaluated_on: 
            row = self.local_meta_data[self.local_meta_data['exp id'].astype(str) == str(indiv_id)]
            try:
                row=row[['rig', 'cre', 'depth', 'area']]

                local_string = row.iloc[0,:].to_string(header=False, index=False).split('\n')        
                local_string = [indiv_str.strip() for indiv_str in local_string]
                local_string = [indiv_str.rjust(7, ' ') for indiv_str in local_string]
                local_string = ' '.join(local_string)

                list_labels_evaluated_on.append(local_string)
            except:
                if "Ai93" in indiv_id:
                    list_labels_trained_on.append('Broadly trained Ai93 model')
                else:
                    list_labels_trained_on.append(indiv_id )

        # Get norm img and sort it per changing imaging conditions
        norm_img = self.norm_img_evaluation

        reindex = np.argsort(list_labels_evaluated_on)
        list_labels_evaluated_on = np.array(list_labels_evaluated_on)
        norm_img = norm_img[:, reindex]
        list_labels_evaluated_on = list_labels_evaluated_on[reindex]
        list_labels_evaluated_on = ['Exp. '+format(index, '02d') for index, local_str in enumerate(list_labels_evaluated_on)]
        list_labels_trained_on = np.array(list_labels_trained_on)
        norm_img[:-1,:] = norm_img[reindex, :]
        list_labels_trained_on[0:-1] = list_labels_trained_on[reindex]
        list_labels_trained_on[0:-1] = ['Exp. '+format(index, '02d')+' '+local_str for index, local_str in enumerate(list_labels_trained_on[0:-1])]

        # We remove the last line as a line with 1 is confusing to folks
        norm_img = norm_img[:-1,:]
        list_labels_trained_on = list_labels_trained_on[:-1]

        plt.imshow(100*(norm_img-1), cmap='coolwarm', clim=[-25, 25])
        y_ticks = np.arange(norm_img.shape[0])
        x_ticks = np.arange(norm_img.shape[1])
        plt.yticks(y_ticks, fontfamily='monospace', fontsize=8)
        plt.gca().set_yticklabels(list_labels_trained_on)
        plt.xticks(x_ticks, list_labels_evaluated_on, rotation=45, ha="right", fontfamily='monospace', fontsize=8)
        cbar = plt.colorbar()
        cbar.set_label('Change in reconstruction loss\nfrom broadly trained Ai93 model (%)', rotation=90, fontsize=8)
        plt.xlabel('Model evaluated on')
        plt.ylabel('Model fine-tuned on')


    def make_figure(self):

        self.fig = plt.figure(figsize=(15, 15))

        ax = placeAxesOnGrid(self.fig, xspan=[0, 0.1], yspan=[0.05, 0.1])
        plt.text(
             0, 0.9, "A", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 1],
            xspan=[0.05, 0.4],
            yspan=[0.1, 0.35],
        )
        self.plot_transfer_loss()
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        ax = placeAxesOnGrid(self.fig, xspan=[0.45, 0.5], yspan=[0.05, 0.1])
        plt.text(
            0, 0.9, "B", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 1],
            xspan=[0.65, 0.95],
            yspan=[0.05, 0.4],
        )
        
        plt.text(
            -0.25, 1.2, "Fine tuning with an L1-loss", fontsize=10, weight="normal", transform=ax.transAxes,
        )

        plt.text(
            -0.55, 1.015, "Rig #    Cre-line   Depth", fontsize=8, weight="bold", transform=ax.transAxes,
        )
        self.plot_evaluation_img()
        
        ax = placeAxesOnGrid(self.fig, dim=[1, 4], xspan=[0.05, 0.95], yspan=[0.4, 0.7])        
        plt.sca(ax[0])
        plt.text(
            0, 1.2, "C", fontsize=20, weight="bold", transform=ax[0].transAxes,
        )
        plt.axis('off')
        
        self.plot_example_fine_tuning(ax)

        """
        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 1],
            xspan=[0.65, 0.95],
            yspan=[0.475, 0.625],
        ) 

        plt.sca(ax)

        self.plot_loss_photon_peak()
        """

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Figure 3 - transfer training.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
