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
      

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

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
            "l1_l2",
            "movie_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450_model.h5",
        )

        self.transfer_example_l1 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "l1_l2",
            "movie_2021_01_10_16_09_transfer_mean_absolute_error_2021_01_10_16_09_model.h5",
        )

        self.transfer_example_l2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "l1_l2",
            "movie_2021_02_01_07_06_transfer_mean_squared_error_2021_02_01_07_06-0004-0.7439.h5",
        )


    def load_data(self):
        with h5py.File(self.local_comp, "r") as file_handle:
            self.raw_dat = file_handle["data_raw"][:,:,:]

        file_handle = h5py.File(self.transfer_example_base, "r") 
        self.transfer_example_base_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_l1, "r") 
        self.transfer_example_l1_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_l2, "r") 
        self.transfer_example_l2_movie = file_handle["data"]

        file_handle = h5py.File(self.transfer_example_l1, "r") 
        self.transfer_example_raw_movie = file_handle["raw"]


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
            img = cut_transpose(img)

            axins.imshow(img, cmap='gray', clim=[vmin, vmax], origin="lower")
            axins.set_xlim(100, 200)
            axins.set_ylim(100, 200)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        def cut_transpose(img):
            img = np.transpose(img)
            img = img[0:410,:]

            return img

        def plot_img(local_img, vmin, vmax):
            local_img = cut_transpose(local_img)
            plt.imshow(local_img, cmap='gray', origin="lower")
            plt.clim(vmin = vmin, vmax = vmax) 
            plt.axis('off')

        frame_nb = 200
        plt.sca(ax[3])
        local_img = self.transfer_example_l2_movie[frame_nb, :, :, 0]
        vmin = np.percentile(local_img.flatten(), 10)
        vmax = np.percentile(local_img.flatten(), 98)
        plot_img(local_img, vmin, vmax)
        plt.title('Fine-tuning Ai93 model\nusing L2 loss')
        zoomed_in(local_img, ax[3])

        plt.sca(ax[1])
        local_img = self.transfer_example_base_movie[frame_nb, :, :, 0]
        plot_img(local_img, vmin, vmax)
        plt.title('Using Ai93 model')
        zoomed_in(local_img, ax[1])

        plt.sca(ax[2])
        local_img = self.transfer_example_l1_movie[frame_nb, :, :, 0]
        plot_img(local_img, vmin, vmax)
        plt.title('Fine-tuning Ai93 model\nusing L1 loss')
        zoomed_in(local_img, ax[2])

        plt.sca(ax[0])
        local_img = self.transfer_example_raw_movie[frame_nb, :, :, 0]
        plot_img(local_img, vmin, vmax)
        plt.title('Raw data with low photon count\nfrom VIP cells')
        zoomed_in(local_img, ax[0])

        rectangle_length = 100 * 512 / 400
        rect = matplotlib.patches.Rectangle(
            [15,15],
            rectangle_length,
            15,
            angle=0.0,
            color="w",
        )
        ax[0].add_patch(rect)


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


    def make_figure(self):

        self.fig = plt.figure(figsize=(15, 15))

        ax = placeAxesOnGrid(self.fig, xspan=[0, 0.1], yspan=[0.05, 0.1])
        plt.text(
             0, 0.75, "A", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 4],
            xspan=[0.05, 0.95],
            yspan=[0.05, 0.45],
        )
        self.plot_example_fine_tuning(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0, 0.1], yspan=[0.35, 0.5])
        plt.text(
             0, 0.75, "B", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 1],
            xspan=[0.05, 0.5],
            yspan=[0.37, 0.75],
        )
        self.plot_loss_photon_peak()
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Ext Figure 6 - L1 vs L2.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
