import matplotlib.pylab as plt
import pandas as pd
import os
import numpy as np
import glob
from scripts.plotting_helpers import placeAxesOnGrid
import matplotlib
import pathlib
import seaborn as sns
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pickle
from sys import path
from suite2p.extraction import preprocess
import h5py as h5
from numba import jit, prange
from scipy.ndimage import filters
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['pdf.fonttype'] = 42

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.local_o_ephys = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "o-ephys"
        )

        self.example_img = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "o-ephys",
            "102932",
            "102932_cropped_all.h5"
        )

        self.local_o_ephys_buff = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "o_ephys_dict.npy"
        )

    def upsample_trace(self, modelx, modely, newx):
        interp_model = interp1d(
            modelx, modely, bounds_error=False, fill_value=0)

        array_F_deep = interp_model(newx)

        return array_F_deep

    def load_data(self):

        with h5.File(self.example_img, 'r') as file_handle:
            self.raw_img = file_handle['raw'][100, :, :]
            self.deepi_img = file_handle['data'][100, :, :]

        if os.path.isfile(self.local_o_ephys_buff):
            self.all_cells_data = np.load(
                self.local_o_ephys_buff, allow_pickle=True).item()
        else:
            list_ephys_cells = os.listdir(self.local_o_ephys)
            self.all_cells_data = {}

            for indiv_folder in list_ephys_cells:
                long_local_folder = os.path.join(
                    self.local_o_ephys, indiv_folder)

                path_to_ophys_F_deep = os.path.join(
                    long_local_folder, "out_movie", "deepinterp", "suite2p", "plane0",  "F.npy")
                path_to_ophys_F_deep_neu = os.path.join(
                    long_local_folder, "out_movie", "deepinterp", "suite2p", "plane0",  "Fneu.npy")
                path_to_ophys_cell_deep = os.path.join(
                    long_local_folder, "out_movie", "deepinterp", "suite2p", "plane0",  "iscell.npy")

                path_to_ophys_F_raw = os.path.join(
                    long_local_folder, "out_movie", "raw", "suite2p", "plane0",  "F.npy")
                path_to_ophys_F_raw_neu = os.path.join(
                    long_local_folder, "out_movie", "raw", "suite2p", "plane0",  "Fneu.npy")
                path_to_ophys_cell_raw = os.path.join(
                    long_local_folder, "out_movie", "raw", "suite2p", "plane0",  "iscell.npy")

                path_to_ephys = os.path.join(
                    long_local_folder, "out_movie", "ephys_processed.h5")

                # Load ephys events. Those are at 40,000khz
                gt_ephys = h5.File(path_to_ephys, 'r')['spk'][:]
                vm_ephys = h5.File(path_to_ephys, 'r')[
                    'Vm'][0, 0:len(gt_ephys[0])]

                # index of ophys frames in ephys
                gt_ephys_indexes = h5.File(path_to_ephys, 'r')[
                    'iFrames'][0, 30:-30]

                # Get index of cell
                array_cell_deep = np.load(path_to_ophys_cell_deep)
                cell_index_deep = np.where(array_cell_deep == 1)[0][0]
                array_cell_raw = np.load(path_to_ophys_cell_raw)
                cell_index_raw = np.where(array_cell_raw == 1)[0][0]

                # Get F and neuropil
                array_F_deep = np.load(path_to_ophys_F_deep)[
                    cell_index_deep, :]
                array_F_neu_deep = np.load(path_to_ophys_F_deep_neu)[
                    cell_index_deep, :]
                array_F_raw = np.load(path_to_ophys_F_raw)[
                    cell_index_raw, 30:-30]
                array_F_neu_raw = np.load(path_to_ophys_F_raw_neu)[
                    cell_index_raw, 30:-30]

                # We upsample ophys to be on the same time scale as ephys
                full_gt_ephys_indexes = np.arange(0, len(gt_ephys[0]))

                array_F_deep = self.upsample_trace(
                    gt_ephys_indexes, array_F_deep, full_gt_ephys_indexes)
                array_F_neu_deep = self.upsample_trace(
                    gt_ephys_indexes, array_F_neu_deep, full_gt_ephys_indexes)
                array_F_raw = self.upsample_trace(
                    gt_ephys_indexes, array_F_raw, full_gt_ephys_indexes)
                array_F_neu_raw = self.upsample_trace(
                    gt_ephys_indexes, array_F_neu_raw, full_gt_ephys_indexes)

                # We downsample both ephys and ophys to 150Hz with simple averaging. Ephys is kept in units of spikes per bin
                ephys_ground_truth = np.array(
                    self.rebin(gt_ephys.flatten(), int(266)))
                vm_ground_truth = np.array(self.rebin(
                    vm_ephys.flatten(), int(266)))/266
                array_F_deep = np.array(self.rebin(
                    array_F_deep.flatten(), int(266)))/266
                array_F_neu_deep = np.array(self.rebin(
                    array_F_neu_deep.flatten(), int(266)))/266
                array_F_raw = np.array(self.rebin(
                    array_F_raw.flatten(), int(266)))/266
                array_F_neu_raw = np.array(self.rebin(
                    array_F_neu_raw.flatten(), int(266)))/266

                frame_rate = 40000/266.
                tau_tetO = 0.9622924335325923
                tau = np.exp(-1/(tau_tetO * frame_rate))

                nndv_spikes_deep = self.nndv(
                    tau, frame_rate, array_F_deep, Fneu=array_F_neu_deep)
                nndv_spikes_raw = self.nndv(
                    tau, frame_rate, array_F_raw, Fneu=array_F_neu_raw)

                self.all_cells_data[indiv_folder] = ({'nndv_spikes_deep': nndv_spikes_deep, 'nndv_spikes_raw': nndv_spikes_raw, 'ephys_ground_truth': ephys_ground_truth,
                                                     'array_F_deep': array_F_deep, 'array_F_raw': array_F_raw, 'vm_ground_truth': vm_ground_truth, 'full_vm': vm_ephys})

                np.save(self.local_o_ephys_buff, self.all_cells_data)

    def oasis_trace(self, F, v, w, t, l, s, tau, fs):
        """ spike deconvolution on a single neuron """
        NT = F.shape[0]
        g = -1./(tau * fs)

        it = 0
        ip = 0

        while it < NT:
            v[ip], w[ip], t[ip], l[ip] = F[it], 1, it, 1
            while ip > 0:
                if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                    # violation of the constraint means merging pools
                    f1 = np.exp(g * l[ip-1])
                    f2 = np.exp(2 * g * l[ip-1])
                    wnew = w[ip-1] + w[ip] * f2
                    v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip] * f1) / wnew
                    w[ip-1] = wnew
                    l[ip-1] = l[ip-1] + l[ip]
                    ip -= 1
                else:
                    break
            it += 1
            ip += 1

        s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])
        return(s)

    # function for computing the non-regularized deconvolution
    # baseline operation

    def nndv(self, tau, fs, F, Fneu):
        # tau = 1.0 # timescale of indicator
        neucoeff = 0.8  # neuropil coefficient
        # for computing and subtracting baseline
        # take the running max of the running min after smoothing with gaussian
        baseline = 'maximin'
        sig_baseline = 5*10.0  # in bins, standard deviation of gaussian with which to smooth
        win_baseline = 60.0  # in seconds, window in which to compute max/min filters

        ops = {'tau': tau, 'fs': fs, 'neucoeff': neucoeff,
               'baseline': baseline, 'sig_baseline': sig_baseline, 'win_baseline': win_baseline, 'batch_size': 200}

        # load traces and subtract neuropil
        #F = f_cell['f_cell'][()]
        #Fneu = f_cell['f_np'][()]
        Fc = F - ops['neucoeff'] * Fneu

        ops['prctile_baseline'] = 8.0
        Fc = np.reshape(Fc, [1, -1])

        # Fc = preprocess(Fc, ops)

        Fc = Fc.ravel()

        NT = Fc.shape[0]
        Fc = Fc.astype(np.float32)

        v = np.zeros(NT, dtype=np.float32)
        w = np.zeros(NT, dtype=np.float32)
        t = np.zeros(NT, dtype=np.int64)
        l = np.zeros(NT, dtype=np.float32)
        s = np.zeros(NT, dtype=np.float32)

        spikes_nndv = self.oasis_trace(
            F=Fc, v=v, w=w, t=t, l=l, s=s, tau=ops['tau'], fs=ops['fs'])
        return spikes_nndv

    # helper function that rebins the events and the ground truth spikes by summation across non-overlapping window of width winsize
    def rebin(self, vector, winsize):  # winsize must be integer
        new_vector = np.zeros(vector.shape[0] // winsize)
        for n, value in enumerate(new_vector):
            new_vector[n] = np.sum(vector[n*winsize:(n+1)*winsize])
        return new_vector

    def make_figure(self):

        self.fig = plt.figure(figsize=(15, 15))

        ax = placeAxesOnGrid(self.fig, xspan=[0.05, 0.1], yspan=[0.05, 0.1])
        plt.text(
            0, 0.75, "A", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(self.fig, xspan=[0.17, 0.3], yspan=[0.07, 0.12])
        plt.axis('off')

        plt.text(
            0, 0, "Dual calcium imaging\nand electrophysiological recording\n(cell 102932)", horizontalalignment='center', fontsize=11, weight="normal", transform=ax.transAxes
        )

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, 2],
            xspan=[0.05, 0.3],
            yspan=[0.15, 0.30],
        )
        plt.sca(ax[0])
        plt.title('Raw imaging data', fontsize=10)
        plt.imshow(self.raw_img[:-9, :], cmap='gray')
        plt.axis('off')

        plt.sca(ax[1])
        plt.title('DeepInterpolation', fontsize=10, color="#8A181A")
        plt.imshow(self.deepi_img[:-9, :], cmap='gray')
        plt.axis('off')

        ax = placeAxesOnGrid(self.fig, xspan=[0.3, 0.35], yspan=[0.05, 0.1])
        plt.text(
            0, 0.75, "B", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[6, 1],
            xspan=[0.5, 0.95],
            yspan=[0.05, 0.30],
        )
        current_cell = '102932'
        local_cell_dict = self.all_cells_data[current_cell]
        frame_rate = 40000/266.
        kept_x_min = int(90*frame_rate)
        kept_x_max = int(120*frame_rate)

        frame_rate_full = 40000
        kept_x_min_full = int(90*frame_rate_full)
        kept_x_max_full = int(120*frame_rate_full)

        nndv_spikes_deep = local_cell_dict['nndv_spikes_deep'][kept_x_min:kept_x_max]
        nndv_spikes_raw = local_cell_dict['nndv_spikes_raw'][kept_x_min:kept_x_max]
        ephys_ground_truth = local_cell_dict['ephys_ground_truth'][kept_x_min:kept_x_max]
        array_F_deep = local_cell_dict['array_F_deep'][kept_x_min:kept_x_max]
        array_F_raw = local_cell_dict['array_F_raw'][kept_x_min:kept_x_max]
        full_vm = local_cell_dict['full_vm'][kept_x_min_full:kept_x_max_full]

        xaxis = 1/frame_rate*np.arange(0, len(ephys_ground_truth))
        xaxis_full = 1/frame_rate_full*np.arange(0, len(full_vm))

        plt.sca(ax[0])
        ax[0].plot(xaxis_full, full_vm)

        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["bottom"].set_visible(False)
        plt.ylabel('\nVoltage (mV)', rotation=0, labelpad=35)
        ax[0].yaxis.set_label_coords(-0.23, 0.4)
        plt.xticks([], [])

        plt.sca(ax[1])
        ax[1].plot(xaxis, ephys_ground_truth)
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        plt.ylabel('\n\nDetected spikes\nin electrophysiology',
                   rotation=0, labelpad=50)
        ax[1].yaxis.set_label_coords(-0.17, 0.2)
        plt.xticks([], [])

        plt.sca(ax[2])
        plt.text(5, 150, 'Raw data', color='black')
        ax[2].plot(xaxis, 100*(array_F_raw /
                   np.mean(array_F_raw)-1), color='black')
        ax[2].spines["right"].set_visible(False)
        ax[2].spines["top"].set_visible(False)
        ax[2].spines["bottom"].set_visible(False)
        plt.ylabel('\nΔF/F (%)', rotation=0, labelpad=30)
        ax[2].yaxis.set_label_coords(-0.15, 0.4)
        plt.xticks([], [])

        plt.sca(ax[3])
        ax[3].plot(xaxis, nndv_spikes_raw, color='black')
        ax[3].spines["right"].set_visible(False)
        ax[3].spines["top"].set_visible(False)
        ax[3].spines["bottom"].set_visible(False)
        ax[3].set_ylabel(
            '\n\nDetected events\nwith OASIS\n(A.U.)', rotation=0, labelpad=50)
        ax[3].yaxis.set_label_coords(-0.17, -0.15)
        plt.xticks([], [])

        plt.sca(ax[4])
        plt.text(5, 150, 'DeepInterpolation', color="#8A181A")
        ax[4].plot(xaxis, 100*(array_F_deep /
                   np.mean(array_F_deep)-1), color="#8A181A")
        ax[4].spines["right"].set_visible(False)
        ax[4].spines["top"].set_visible(False)
        ax[4].spines["bottom"].set_visible(False)
        plt.ylabel('\nΔF/F (%)', rotation=0, labelpad=30)
        ax[4].yaxis.set_label_coords(-0.15, 0.4)
        plt.xticks([], [])

        plt.sca(ax[5])
        ax[5].plot(xaxis, nndv_spikes_deep, color="#8A181A")
        ax[5].spines["right"].set_visible(False)
        ax[5].spines["top"].set_visible(False)
        yvar = plt.ylabel('Detected events\nwith OASIS\n(A.U.)',
                          rotation=0, labelpad=50)
        ax[5].yaxis.set_label_coords(-0.17, -0.15)
        plt.xlabel('Time (s)')

        ax = placeAxesOnGrid(self.fig, xspan=[0.05, 0.1], yspan=[0.32, 0.37])
        plt.text(
            0, 0.75, "C", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, len(self.all_cells_data.keys())],
            xspan=[0.05, 0.95],
            yspan=[0.37, 0.55],
        )
        factor = int(5)

        for index_key, indiv_cell in enumerate(self.all_cells_data.keys()):

            local_cell_dict = self.all_cells_data[indiv_cell]

            nndv_spikes_deep = local_cell_dict['nndv_spikes_deep']
            nndv_spikes_raw = local_cell_dict['nndv_spikes_raw']
            ephys_ground_truth = local_cell_dict['ephys_ground_truth']
            array_F_raw = local_cell_dict['array_F_raw']
            array_F_deep = local_cell_dict['array_F_deep']

            # These are the things I've been comparing. Adapt to the names of your dictionaries.
            methods = {'black': array_F_raw, '#8A181A': array_F_deep}

            for color, method in methods.items():
                plt.sca(ax[index_key])
                # compute ROC curves
                # rebin ground truth according to factor
                gt = self.rebin(ephys_ground_truth.ravel(), factor)
                spike_nb = np.unique(gt)
                # gt = gt/np.max(gt) #normalize to max

                test = method.ravel()  # load the events you are comparing
                test[np.isnan(test)] = 0  # if any entries are nan, set to 0
                # rebin loaded events according to factor
                test = self.rebin(test, factor)
                f0 = np.mean(test[gt == 0])

                dff = 100*(test/f0-1)
                data = [dff[gt == unique_spk] for unique_spk in spike_nb]

                if color == '#8A181A':
                    delta = 0.25
                else:
                    delta = -0.25

                violin_parts = ax[index_key].violinplot(data, spike_nb+delta, points=20, widths=0.3,
                                                        showmeans=True, showextrema=True, showmedians=True)
                plt.xlabel('Nb of spikes')

                if index_key == 0:
                    plt.ylabel('ΔF/F (%)')

                    # Make all the violin statistics marks red:
                for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                    vp = violin_parts[partname]
                    vp.set_edgecolor(color)

                for pc in violin_parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_color(color)

                plt.title('Cell '+str(indiv_cell))
                ax[index_key].spines["right"].set_visible(False)
                ax[index_key].spines["top"].set_visible(False)
                plt.xlim(-1, 5)
                plt.xticks(np.arange(0, 5, step=1))

        ax = placeAxesOnGrid(self.fig, xspan=[0.05, 0.1], yspan=[0.6, 0.65])
        plt.text(
            0, 0.75, "D", fontsize=20, weight="bold", transform=ax.transAxes,
        )
        plt.axis('off')

        # These are binning factors.
        factors = np.array([1, 2, 5, 15]).astype('int')

        ax = placeAxesOnGrid(
            self.fig,
            dim=[1, len(factors)],
            xspan=[0.05, 0.95],
            yspan=[0.65, 0.85],
        )

        axins = []
        for k, factor in enumerate(factors):
            ax[k].set_title(
                'bin:' + str(np.round(factor * 1000./(40000./266))) + ' ms')

            # sub region of the original plot
            axins.append(ax[k].inset_axes([0.4, 0.1, 0.5, 0.45]))
            # axins.imshow(Z2, extent=extent, interpolation="nearest",
            #        origin="lower")

            ax[k].indicate_inset_zoom(axins[k])

            for index_key, indiv_cell in enumerate(self.all_cells_data.keys()):

                local_cell_dict = self.all_cells_data[indiv_cell]

                nndv_spikes_deep = local_cell_dict['nndv_spikes_deep']
                nndv_spikes_raw = local_cell_dict['nndv_spikes_raw']
                ephys_ground_truth = local_cell_dict['ephys_ground_truth']

                # These are the things I've been comparing. Adapt to the names of your dictionaries.
                methods = {'#8A181A': nndv_spikes_deep, 'k': nndv_spikes_raw}
                for color, method in methods.items():

                    # compute ROC curves
                    # rebin ground truth according to factor
                    gt = self.rebin(ephys_ground_truth.ravel(), factor)
                    gt = gt/np.max(gt)  # normalize to max
                    test = method.ravel()  # load the events you are comparing
                    # if any entries are nan, set to 0
                    test[np.isnan(test)] = 0
                    # rebin loaded events according to factor
                    test = self.rebin(test, factor)
                    # normalize events after rebinning
                    test = test/np.max(test)
                    # compute the ROC curve - does not work is ground truth is non-binary
                    fpr, tpr, _ = roc_curve(gt > 0, test)

                    # plot average of interpolated ROC curves for each cell
                    ax[k].plot(fpr, tpr, color, alpha=1, linewidth=0.2)
                    axins[k].plot(fpr, tpr, color, alpha=1, linewidth=0.2)
            if k == 0:
                ax[k].set_ylabel('True Positive Rate')

            ax[k].set_xlabel('False Positive Rate')

            ax[k].spines["right"].set_visible(False)
            ax[k].spines["top"].set_visible(False)

            x1, x2, y1, y2 = 0, 0.05, 0, 0.8
            axins[k].set_xlim(x1, x2)
            axins[k].set_ylim(y1, y2)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Ext Figure 3 - ophys vs ephys.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
