import h5py
import numpy as np
import matplotlib.pylab as plt
import os
import matplotlib
from scripts.plotting_helpers import placeAxesOnGrid
from matplotlib.patches import Rectangle
import pathlib
from read_roi import read_roi_zip
import cv2
import colorsys
from scipy.fft import fft, fftfreq
from scipy.stats import sem
import scipy.signal as signal
import scipy.io as scio
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.path_to_raw_clean = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_clean.npy",
        )
        self.path_to_raw_noisy = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_noisy.npy",
        )
        self.path_to_di = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_di_3.npy",
        )
        self.path_to_pmd = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_pmd.npy",
        )

        self.path_to_ground_traces = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "2020-03-03-spikes_and_neurons.mat",
        )

        self.path_to_gaussian = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_gaussian.npy",
        )
        

    def load_data(self):

        self.raw_movie_clean = np.load(self.path_to_raw_clean)[:,:,30:-30]
        self.raw_movie_noisy = np.load(self.path_to_raw_noisy)[:,:,30:-30]
        self.raw_movie_gaussian = np.load(self.path_to_gaussian)[:,:,30:-30]

        self.di_movie = np.load(self.path_to_di)[:,:,:]
        self.pmd_movie = np.load(self.path_to_pmd)[:,:,30:-30]
        data_mat = scio.loadmat(self.path_to_ground_traces)
        self.ground_images = data_mat['images'][0]
        self.ground_spikes = data_mat['traces'][0]
        self.fill_rois()

    def fill_rois(self):
        self.final_roi = dict()
        for index, roi in enumerate(np.arange(len(self.ground_images))):
            self.final_roi[roi] = self.ground_images[roi][0:256, 0:256]

    def get_index_spike_trig(self, trace_time, trace_spike, spike_count_range, spike_period = 3):
        spike_count_array = np.convolve(trace_spike, np.ones(spike_period), mode = 'same')
        
        peaks, __ = find_peaks(spike_count_array)

        spike_count_indexes = []
        all_peaks_value = spike_count_array[peaks]
        for spike_count in spike_count_range:
            
            local_indexes =  np.where((all_peaks_value>=spike_count-0.5) & (all_peaks_value<(spike_count+0.5)))[0]
            spike_count_indexes.append(peaks[local_indexes])
        
        return spike_count_indexes
    
    def get_dff_comparison(self):
        
        spike_count_range = np.arange(1, 15, 1)
        average = True

        list_index =  np.arange(200) # len(X.ground_images)) 
        dff_list_clean = []
        dff_list_di = []
        dff_list_gaussian = []
        area_list_clean = []
        area_list_di = []
        area_list_gaussian = []

        for index in list_index:
            cell_id = index
            local_roi = self.final_roi[cell_id]
            local_img = self.get_cell_truth_image(cell_id)

            if np.size(local_img[local_img>0])>100:
                start_mov = True


                calcium_trace_di = self.get_av_trace_plot(self.di_movie, local_roi)
                calcium_trace_clean = self.get_av_trace_plot(self.raw_movie_noisy, local_roi)

                calcium_trace_gaussian = gaussian_filter1d(calcium_trace_clean, 3)

                trace_spike = self.ground_spikes[cell_id][0][60:60+len(calcium_trace_clean)]

                single_spike = np.min(trace_spike[trace_spike>0])
                trace_spike = trace_spike/single_spike

                share_time = np.arange(0, len(trace_spike))
                all_calcium_average_di, n_averages_di = self.get_spike_triggered_average(share_time, calcium_trace_di, trace_spike, spike_count_range = spike_count_range, average=average)
                all_calcium_average_clean, n_averages_clean = self.get_spike_triggered_average(share_time, calcium_trace_clean, trace_spike, spike_count_range = spike_count_range, average=average)
                all_calcium_average_gaussian, n_averages_gaussian = self.get_spike_triggered_average(share_time, calcium_trace_gaussian, trace_spike, spike_count_range = spike_count_range, average=average)

                for index_av, indiv_average_clean in enumerate(all_calcium_average_clean):
                    dff_max_clean = np.max(indiv_average_clean)
                    dff_max_di = np.max(all_calcium_average_di[index_av])
                    dff_max_gaussian = np.max(all_calcium_average_gaussian[index_av])

                    dff_area_clean = np.sum(indiv_average_clean[indiv_average_clean>0])
                    dff_area_di = np.sum(all_calcium_average_di[index_av][all_calcium_average_di[index_av]>0])
                    dff_area_gaussian = np.sum(all_calcium_average_gaussian[index_av][all_calcium_average_gaussian[index_av]>0])

                    dff_list_clean.append(dff_max_clean)
                    dff_list_di.append(dff_max_di)
                    dff_list_gaussian.append(dff_max_gaussian)

                    area_list_clean.append(dff_area_clean)
                    area_list_di.append(dff_area_di)
                    area_list_gaussian.append(dff_area_gaussian)

        return area_list_clean, area_list_di, area_list_gaussian, dff_list_clean, dff_list_di, dff_list_gaussian

    def plot_dff_comparison(self, dff_list_clean, dff_list_di, dff_list_gaussian):
        plt.plot(100*np.array(dff_list_clean), 100*np.array(dff_list_clean), 'k-', markersize=1.5, label='Ground truth + Noise')        
        plt.plot(100*np.array(dff_list_clean), 100*np.array(dff_list_gaussian), '.', color='cornflowerblue', markersize=1.5, label='Gaussian kernel')
        plt.plot(100*np.array(dff_list_clean), 100*np.array(dff_list_di), '.', color = "#8A181A",  markersize=1.5, label='DeepInterpolation')

        plt.xlabel(u"ΔF/F event ampliture (%) in ground truth")
        plt.ylabel(u"ΔF/F event ampliture (%)\nafter DeepInterpolation")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.legend(prop={"size": 6}, frameon=False)

    def plot_dff_area_comparison(self, area_list_clean, area_list_di, area_list_gaussian):
        plt.plot(area_list_clean, area_list_clean, 'k-', markersize=1.5)        
        plt.plot(area_list_clean, area_list_gaussian, '.', color='cornflowerblue', markersize=1.5)
        plt.plot(area_list_clean, area_list_di, '.', color = "#8A181A", markersize=1.5)

        plt.xlabel("Area under calcium events in ground truth (AU)")
        plt.ylabel("Area under calcium events\nafter DeepInterpolation (AU)")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

    def get_spike_triggered_average(self, share_time, calcium_trace, trace_spike, spike_count_range = np.arange(1, 10, 2), average=True):
        
        list_spike_index = self.get_index_spike_trig(share_time, trace_spike, spike_count_range, spike_period = 3)

        average_before = 10
        average_after = 30

        all_calcium_average = []
        n_averages = []
        for indiv_spike_indexes in list_spike_index:
            index_average = 0
            for indiv_index in indiv_spike_indexes:
                if (indiv_index-average_before)>0 and (indiv_index+average_after)<len(calcium_trace):
                    if index_average == 0:
                        local_trace = calcium_trace[indiv_index-average_before:indiv_index+average_after]
                    else:
                        local_trace = local_trace + calcium_trace[indiv_index-average_before:indiv_index+average_after]
                    index_average = index_average + 1
                if not(average) and index_average==1:
                    break
            n_averages.append(index_average)


            local_trace = local_trace / index_average
            all_calcium_average.append(local_trace)

        return all_calcium_average, n_averages

    def get_cell_truth_image(self, cell_id):
        data = self.final_roi[cell_id]
        self.ground_images
        return data

    def plot_cell_spike_map(self, cell_id, list_mov, local_axis, spike_count_range = np.arange(1, 10, 2), average=True, display_y=False):
        local_roi = self.final_roi[cell_id]

        start_mov = True
        for key, local_mov in list_mov.items():
            calcium_trace = self.get_av_trace_plot(local_mov, local_roi)
            
            trace_spike = self.ground_spikes[cell_id][0][60:60+len(calcium_trace)]
            single_spike = np.min(trace_spike[trace_spike>0])
            trace_spike = trace_spike/single_spike

            share_time = np.arange(0, len(trace_spike))
            all_calcium_average, n_averages = self.get_spike_triggered_average(share_time, calcium_trace, trace_spike, spike_count_range = spike_count_range, average=average)

            for index_av, indiv_average in enumerate(all_calcium_average):
                plt.sca(local_axis[index_av])
                plt.title(str(spike_count_range[index_av])+ " spike(s), N="+str(n_averages[index_av]), fontsize=8)
                if "Deep" in key:
                    plt.plot(1/11*np.arange(len(indiv_average)), 100*indiv_average, label=key, color = "#8A181A", linewidth=1.0)
                elif "Gaussian" in key:
                    plt.plot(1/11*np.arange(len(indiv_average)), 100*indiv_average, label=key, color = "cornflowerblue", linewidth=1.0)
                else:
                    plt.plot(1/11*np.arange(len(indiv_average)), 100*indiv_average, label=key, color = "k", linewidth=1.0)

                if index_av < len(all_calcium_average)-1:
                    plt.gca().spines['bottom'].set_visible(False)
                    plt.xticks([])

                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                if not(display_y):
                    plt.gca().spines['left'].set_visible(False)
                    plt.yticks([])

                plt.ylim([-10, 150])
            
            start_mov = False

    def get_av_trace_plot(self, input_mov, input_roi, dff=True):
        local_trace = []
        for index_frame in range(input_mov.shape[2]):
            local_img_mov = input_mov[:, :, index_frame]
            local_trace.append(np.mean(local_img_mov[input_roi == 1.0].flatten()))

        if dff:
            return (local_trace - np.mean(local_trace)) / np.mean(local_trace)
        else:
            return local_trace

    def get_error_comparison(self):
        list_mov = {
            "ground_truth": self.raw_movie_clean,
            "noisy": self.raw_movie_noisy,
            "pmd": self.pmd_movie,
            "deep interpolation": self.di_movie,
        }

        l2_dict = {}
        for index, local_key in enumerate(self.final_roi.keys()):
            local_roi = self.final_roi[local_key]
            print(local_roi)
            local_mov = list_mov["ground_truth"]
            local_trace_ground = self.get_av_trace_plot(local_mov, local_roi)

            for mov_key in list_mov.keys():
                if mov_key != "ground_truth":
                    local_mov = list_mov[mov_key]
                    local_trace = self.get_av_trace_plot(local_mov, local_roi)
                    local_l2 = np.sqrt(np.sum((local_trace - local_trace_ground) ** 2))
                    try:
                        l2_dict[mov_key].append(local_l2)
                    except:
                        l2_dict[mov_key] = [local_l2]
        self.l2_dict = l2_dict

    def plot_error_comparison(self, norm=False):
        if norm == False:
            w = 0.8
            plt.bar(
                x=["noisy", "pmd", "deep interpolation"],
                height=[
                    np.mean(self.l2_dict["noisy"]),
                    np.mean(self.l2_dict["pmd"]),
                    np.mean(self.l2_dict["deep interpolation"]),
                ],
                width=w,
                color=(0, 0, 0, 0),
                edgecolor="black",
            )
            for index, key in enumerate(self.l2_dict.keys()):
                # distribute scatter randomly across whole width of bar
                plt.gca().scatter(
                    index + np.random.random(len(self.l2_dict[key])) * w - w / 2,
                    self.l2_dict[key],
                    color="black",
                )

            plt.ylabel("L2 norm of error with ground truth")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
        else:
            plt.plot(
                self.l2_dict["noisy"],
                np.array(self.l2_dict["pmd"])
                / np.array(self.l2_dict["deep interpolation"]),
                "k.",
            )
            plt.xlabel("Raw data: L2 norm\nerror with ground truth")
            plt.ylabel("Improvement of\nDeepInterpolation\nover PMD")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

    def plot_example_movie_roi(self, current_mov, number_traces=False, scale_bar=False):

        frame_index = 200
        list_roi = np.arange(len(self.final_roi)) #[0, 1, 2, 3, 4, 5, 6, 7, 8 ]

        # We first create the ROI + image of the figure
        local_img = current_mov[:, :, frame_index]

        local_img = self.convert_gray_rgb(local_img)

        list_color = 50*np.ones(list_roi.shape)# [50, 50, 50, 50, 50, 50, 50, 50, 50]
        index_trace = 0

        for index_key, local_key in enumerate(self.final_roi.keys()):
            if index_key in list_roi:
                index_trace = index_trace + 1

                local_roi = self.final_roi[local_key]
                color = list_color[index_key]
                if number_traces:
                    [location_roi_x, location_roi_y] = np.where(local_roi > 0)
                    x_mean = np.mean(location_roi_x)
                    y_mean = np.mean(location_roi_y)

                    plt.text(y_mean, x_mean + 15, str(index_trace), color="red")

                local_img = self.colorized(local_img, local_roi, color)

        plt.imshow(local_img)
        plt.gca().axis("off")

        if scale_bar:
            rectangle_length = 100 * current_mov.shape[0] / 400
            rect = matplotlib.patches.Rectangle(
                [20, 230], rectangle_length, 10, angle=0.0, color="w"
            )
            plt.gca().add_patch(rect)

    def plot_example_roi_traces(
        self, ax, current_mov, ground_mov, number_traces=False, time_axis=True
    ):

        list_mov = {"extracted trace": current_mov, "ground truth trace": ground_mov}
        start_trace = 200
        list_roi = [0, 1, 2, 3, 4, 5, 6, 7, 8 ]
        index_trace = 0
        for index, local_key in enumerate(self.final_roi.keys()):
            if index in list_roi:
                plt.sca(ax[index_trace])

                index_trace = index_trace + 1
                local_roi = self.final_roi[local_key]
                for index_key, mov_key in enumerate(list_mov.keys()):
                    local_mov = list_mov[mov_key]
                    local_trace = self.get_av_trace_plot(local_mov, local_roi)

                    plt.plot(100 * local_trace, label=mov_key, linewidth=0.5)

                local_trace = self.ground_spikes[index][0][60:60+len(local_trace)]
                plt.plot(25*local_trace/np.max(local_trace), label='Spikes', linewidth=0.25)
                plt.gca().spines["bottom"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.ylim(-15, 25)
                plt.xlim(start_trace, 2000)

                if number_traces:
                    plt.text(start_trace - 60, 0, str(index_trace), color="red")

                #if number_traces and index_trace == 4:
                #    plt.legend(loc="lower right", prop={"size": 6})

        if time_axis:
            rect = matplotlib.patches.Rectangle(
                [start_trace + 1, -10], 300, 1, angle=0.0, color="k",
            )
            plt.gca().add_patch(rect)
            rect = matplotlib.patches.Rectangle(
                [start_trace + 1, -10], 1, 25, angle=0.0, color="k",
            )

            plt.gca().add_patch(rect)
            plt.text(
                0.04, -0.175, "10s ", fontsize=6, transform=plt.gca().transAxes,
            )
            plt.text(
                -0.20,
                0.05,
                "25 %\n$\Delta$F/F (%)",
                fontsize=6,
                transform=plt.gca().transAxes,
            )

    def make_figure(self):

        self.fig = plt.figure(figsize=(12, 7.5))


        local_axis = placeAxesOnGrid(self.fig, xspan=(0.0, 0.1), yspan=(0.0, 0.1))
        plt.axis('off')
        plt.text(-0.2, 1.05, "A", fontsize=15, weight="bold", transform=local_axis.transAxes)

        list_index = [181, 290]#, 320] #2,3,4,13,14,15,16,19,20,25,26,29,30,39,42,44,45,47,48,51,55,56,57,64,66,67,75,80,82,83,85,89,92,96]

        # for index in list_index:
        cell_id = 320

        local_axis = placeAxesOnGrid(self.fig, dim=[5, 1], xspan=(0.05, 0.20), yspan=(0, 9), wspace=0)            
        self.plot_cell_spike_map(cell_id, {'Gaussian kernel' : self.raw_movie_gaussian, 'Ground Truth + Noise': self.raw_movie_noisy, 'DeepInterpolation': self.di_movie}, local_axis = local_axis, spike_count_range = np.arange(1, 10, 2), average=False, display_y=True)  
        plt.sca(local_axis[2])
        plt.ylabel(u'ΔF/F (%)')
        plt.sca(local_axis[4])
        plt.xlabel(u'Time (s)')

        local_axis = placeAxesOnGrid(self.fig, dim=[5, 1], xspan=(0.25, 0.40), yspan=(0, 9), wspace=0)
        self.plot_cell_spike_map(cell_id, {'Gaussian kernel' : self.raw_movie_gaussian, 'Ground Truth + Noise': self.raw_movie_clean, 'DeepInterpolation': self.di_movie}, local_axis = local_axis, spike_count_range = np.arange(1, 10, 2), average=True, display_y=False)  
        plt.sca(local_axis[1])

        plt.legend(prop={"size": 6}, frameon=False)

        plt.sca(local_axis[4])
        plt.xlabel(u'Time (s)')

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.55, 0.95), yspan=(0.05, 0.45))
        area_list_clean, area_list_di, area_list_gaussian, dff_list_clean, dff_list_di, dff_list_gaussian = self.get_dff_comparison()

        plt.text(-0.2, 1.05, "B", fontsize=15, weight="bold", transform=local_axis.transAxes)
        self.plot_dff_comparison(dff_list_clean, dff_list_di, dff_list_gaussian)

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.55, 0.95), yspan=(0.55, 0.95))
        plt.text(-0.2, 1.05, "C", fontsize=15, weight="bold", transform=local_axis.transAxes)
        self.plot_dff_area_comparison(area_list_clean, area_list_di, area_list_gaussian)


    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 4.pdf",
    )

    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
