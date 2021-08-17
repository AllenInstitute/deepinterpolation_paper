from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import h5py
import numpy as np
import matplotlib.pylab as plt
import os
import matplotlib
from scipy.stats import mode
import scipy
import pathlib

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['pdf.fonttype'] = 42

class Analyze_corr:
    def __init__(self, lims_id, stim):
        self.exp_id = lims_id
        self.path_to_denoised_movie = (
            r"Z:\\"
            + str(lims_id)
            + "\movie_"
            + str(lims_id)
            + "_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai148-0450.h5"
        )
        self.path_to_denoised_traces = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            str(lims_id) + "-denoised_traces_original_roi.npy",
        )
        self.path_to_sample_movie = (
            r"Z:\\" + str(lims_id) + "\comp\comp_" + str(lims_id) + ".h5"
        )
        self.stim = stim

    def load_data(self):
        # We first gather needed data
        self.get_segmented_raw()
        self.get_segmented_denoised()
        self.get_reorg_sweeps()

    def get_signal_noise_corr(self, input_array, plot=True):
        nb_repeat = input_array.shape[1]
        # For signal corr, we first average across trials
        natural_movie_denf_signal = np.mean(input_array, axis=1)
        signal_corr_array = np.corrcoef(natural_movie_denf_signal)

        # For noise corr, we first remove the signal
        natural_movie_denf_noise_tmp = input_array
        for local_repeat in range(0, nb_repeat):
            natural_movie_denf_noise_tmp[:, local_repeat, :] = (
                natural_movie_denf_noise_tmp[:, local_repeat, :]
                - natural_movie_denf_signal
            )

        local_shape = natural_movie_denf_noise_tmp.shape
        natural_movie_denf_noise_tmp = natural_movie_denf_noise_tmp.reshape(
            [local_shape[0], -1]
        )

        noise_corr_array = np.corrcoef(natural_movie_denf_noise_tmp)
        full_sign_corr = signal_corr_array.copy()
        full_noise_corr = noise_corr_array.copy()
        signal_corr_array = signal_corr_array[
            np.triu_indices(signal_corr_array.shape[0], k=1)
        ]
        noise_corr_array = noise_corr_array[
            np.triu_indices(noise_corr_array.shape[0], k=1)
        ]

        return [signal_corr_array, noise_corr_array, full_sign_corr, full_noise_corr]

    def get_segmented_denoised(self):

        if os.path.isfile(self.path_to_denoised_traces):
            self.denoised_f = np.load(self.path_to_denoised_traces)
        else:
            nb_mask = len(self.list_masks)
            with h5py.File(self.path_to_denoised_movie, "r") as h5_handle:
                local_data = h5_handle["data"]
                nb_frames = local_data.shape[0]

                self.denoised_f = np.zeros([nb_mask, nb_frames])
                for local_frame_index in np.arange(0, nb_frames):
                    local_frame = local_data[local_frame_index, :, :]
                    print(local_frame_index)
                    for index_mask, local_mask in enumerate(self.list_masks):
                        img = local_mask.get_mask_plane()
                        local_average = np.mean(
                            local_frame[img == True].flatten())
                        self.denoised_f[index_mask,
                                        local_frame_index] = local_average

            np.save(self.path_to_denoised_traces, self.denoised_f)

    def get_segmented_raw(self):

        # We get the metadata from the sdk
        boc = BrainObservatoryCache()
        exp = boc.get_ophys_experiment_data(self.exp_id)

        self.raw_f = exp.get_fluorescence_traces()[1]
        self.list_masks = exp.get_roi_mask()

        stim_table = exp.get_stimulus_table(
            self.stim
        )  # Remember to take into account the 30 frames offset in there too (start and end frames)

        list_frame = stim_table["frame"].unique()
        self.list_repeat = stim_table["repeat"].unique()

        list_frame.sort()
        start_frame = list_frame[1]
        end_frame = list_frame[-2]

        list_rows_end = stim_table["frame"] == end_frame
        list_rows_start = stim_table["frame"] == start_frame
        self.all_start_frames = stim_table[list_rows_start]
        self.all_end_frames = stim_table[list_rows_end]

    def get_reorg_sweeps(self):

        nb_neurons = self.denoised_f.shape[0]

        # Reorganization of data
        for neuron_nb in np.arange(0, nb_neurons):
            for index_repeat, indiv_repeat in enumerate(self.list_repeat):
                list_rows_start2 = self.all_start_frames["repeat"] == indiv_repeat
                list_rows_end2 = self.all_end_frames["repeat"] == indiv_repeat

                list_rows_start_selected = self.all_start_frames[list_rows_start2]
                list_rows_end_selected = self.all_end_frames[list_rows_end2]

                local_start = int(list_rows_start_selected["start"])
                local_end = int(list_rows_end_selected["end"])

                local_trace_raw = self.raw_f[neuron_nb, local_start:local_end]
                local_trace_den = self.denoised_f[
                    neuron_nb, local_start - 30: local_end - 30
                ]

                if neuron_nb == 0 and index_repeat == 0:
                    natural_movie_denf = np.zeros(
                        [nb_neurons, len(self.list_repeat),
                         len(local_trace_den)]
                    )
                    natural_movie_rawf = np.ones(
                        [nb_neurons, len(self.list_repeat),
                         len(local_trace_raw)]
                    )

                end_trace = np.min(
                    [len(local_trace_den), natural_movie_denf.shape[2]])
                natural_movie_denf[
                    neuron_nb, index_repeat, 0:end_trace
                ] = local_trace_den[0:end_trace]
                natural_movie_rawf[
                    neuron_nb, index_repeat, 0:end_trace
                ] = local_trace_raw[0:end_trace]

        self.natural_movie_denf = natural_movie_denf
        self.natural_movie_rawf = natural_movie_rawf


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file
        local_path = pathlib.Path(__file__).parent.absolute()

        self.path_to_sample_movie = os.path.join(
            local_path, "../data/local_large_data/comp_637998955.h5"
        )

        self.list_id_denoised_traces = [
            637998955, 505811062, 509904120, 657391625]
        self.list_id_denoised_traces_stim = [
            "natural_movie_three",
            "natural_movie_one",
            "natural_movie_one",
            "natural_movie_one",
        ]

    def load_data(self):

        self.list_id_obj = []

        # We first gather needed data
        for index, local_id in enumerate(self.list_id_denoised_traces):
            local_obj_id = Analyze_corr(
                local_id, self.list_id_denoised_traces_stim[index]
            )

            local_obj_id.get_segmented_raw()
            local_obj_id.get_segmented_denoised()
            local_obj_id.get_reorg_sweeps()

            self.list_id_obj.append(local_obj_id)

        print(self.list_id_obj)
        self.get_sample_frames()

    def get_sample_frames(self):
        with h5py.File(self.path_to_sample_movie, "r") as file_handle:
            self.denoised_sample = file_handle["data_proc"][:, :, :]
            self.raw_sample = file_handle["data_raw"][:, :, :]

    def make_figure(self):
        self.fig = plt.figure(figsize=(7, 12))
        global_grid = self.fig.add_gridspec(
            nrows=8,
            ncols=2,
            height_ratios=[4, 2, 2, 2, 1, 2, 1, 4],
            wspace=0.3,
            hspace=0.1,
        )

        ax = self.fig.add_subplot(global_grid[0, 0])
        plt.text(-0.35, 1.1, "A", fontsize=15,
                 weight="bold", transform=ax.transAxes)

        for index, local_cell_masks in enumerate(self.list_id_obj):
            print(
                "Number of cell masks in exp "
                + str(index)
                + " : "
                + str(len(local_cell_masks.list_masks))
            )

        self.plot_filters_on_top_image(
            ax,
            self.list_id_obj[0].list_masks,
            self.raw_sample[100, :, :].astype("float"),
        )
        plt.title("Raw movie")
        rectangle_length = 100 * 512 / 400
        rect = matplotlib.patches.Rectangle(
            [20, 512 - 40], rectangle_length, 15, angle=0.0, color="w"
        )
        plt.gca().add_patch(rect)
        plt.arrow(
            1.05,
            0.4,
            0.35,
            0,
            head_width=0.075,
            transform=ax.transAxes,
            clip_on=False,
            color="k",
            facecolor="k",
        )
        plt.text(
            1.05,
            0.5,
            "Apply ROI\nfound\non raw movie",
            fontsize=10,
            transform=ax.transAxes,
        )

        ax = self.fig.add_subplot(global_grid[0, 1])
        self.plot_filters_on_top_image(
            ax,
            self.list_id_obj[0].list_masks,
            self.denoised_sample[100, :, :].astype("float"),
        )
        plt.title("with DeepInterpolation")

        for index, neuron_nb in enumerate([35, 65, 84]):
            ax = self.fig.add_subplot(global_grid[1 + index, 1])
            color_den = (138 / 255.0, 24 / 255.0, 26 / 255.0)
            color_raw = (74 / 255.0, 72 / 255.0, 74 / 255.0)

            plt.text(
                -0.65,
                0.5,
                "Example\ncell " + str(index),
                fontsize=12,
                transform=ax.transAxes,
            )

            self.plot_example_reliability(
                ax, self.list_id_obj[0].natural_movie_denf, neuron_nb, color=color_den
            )
            local_ylim = ax.get_ylim()
            if index < 2:
                ax.get_xaxis().set_visible(False)
                ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = self.fig.add_subplot(global_grid[1 + index, 0])
            if index == 0:
                plt.text(
                    -0.2, 1, "B", fontsize=15, weight="bold", transform=ax.transAxes
                )

            self.plot_example_reliability(
                ax, self.list_id_obj[0].natural_movie_rawf, neuron_nb, color=color_raw
            )
            ax.set_ylim(local_ylim)

            if index < 2:
                ax.get_xaxis().set_visible(False)
                ax.spines["bottom"].set_visible(False)
            else:
                plt.xlabel("Time from start of stim (s)")
                plt.subplots_adjust(wspace=0.3)

        sig_corr_raw = np.array([])
        sig_corr_den = np.array([])
        noise_corr_raw = np.array([])
        noise_corr_den = np.array([])

        for index, local_obj in enumerate(self.list_id_obj):
            [
                signal_corr_array_raw,
                noise_corr_array_raw,
                full_sign_corr_raw,
                full_noise_corr_raw,
            ] = local_obj.get_signal_noise_corr(
                local_obj.natural_movie_rawf, plot=False
            )
            [
                signal_corr_array_di,
                noise_corr_array_di,
                full_sign_corr_den,
                full_noise_corr_den,
            ] = local_obj.get_signal_noise_corr(
                local_obj.natural_movie_denf, plot=False
            )
            if index == 0:
                example_noise_corr_raw = full_noise_corr_raw
                example_noise_corr_di = full_noise_corr_den

            sig_corr_raw = np.append(
                sig_corr_raw, signal_corr_array_raw.flatten(), 0)
            sig_corr_den = np.append(
                sig_corr_den, signal_corr_array_di.flatten(), 0)
            noise_corr_raw = np.append(
                noise_corr_raw, noise_corr_array_raw.flatten(), 0
            )
            noise_corr_den = np.append(
                noise_corr_den, noise_corr_array_di.flatten(), 0)

        ax = self.fig.add_subplot(global_grid[5, 0])
        plt.text(-0.18, 1, "C", fontsize=15,
                 weight="bold", transform=ax.transAxes)
        self.plot_hist_overlaid(sig_corr_raw, sig_corr_den)
        plt.xlabel("signal correlation")
        plt.text(
            0.7,
            0.8,
            "N=" + str(len(sig_corr_raw)) + " pairs",
            fontsize=8,
            transform=ax.transAxes,
        )

        mean_raw = np.mean(sig_corr_raw)
        mean_den = np.mean(sig_corr_den)
        std_raw = np.std(sig_corr_raw)
        std_den = np.std(sig_corr_den)
        mode_raw = mode(sig_corr_raw)[0]
        mode_den = mode(sig_corr_den)[0]
        kurtosis_raw = np.mean(
            np.power(((sig_corr_raw - mean_raw) / std_raw), 4))
        kurtosis_den = np.mean(
            np.power(((sig_corr_den - mean_den) / std_den), 4))
        p_test_signal = scipy.stats.ttest_rel(sig_corr_raw, sig_corr_den)

        print("Mean of raw sign corr: " + str(mean_raw))
        print("Mean of den sign corr: " + str(mean_den))
        print("SEM of raw sign corr: " + str(scipy.stats.sem(sig_corr_raw)))
        print("SEM of den sign corr: " + str(scipy.stats.sem(sig_corr_den)))
        print("N: " + str(len(sig_corr_den)))
        print("Mode of raw signal: " + str(mode_raw))
        print("Mode of den signal: " + str(mode_den))
        print("Raw signal pearson skewness: " +
              str((mean_raw - mode_raw) / std_raw))
        print("Den signal pearson skewness: " +
              str((mean_den - mode_den) / std_den))
        print("Raw signal kurtosis: " + str(kurtosis_raw))
        print("Den signal kurtosis: " + str(kurtosis_den))
        print("student t-test, pvalue : " + str(p_test_signal[1]))

        ax = self.fig.add_subplot(global_grid[5, 1])
        self.plot_hist_overlaid(noise_corr_raw, noise_corr_den)
        plt.xlabel("noise correlation")
        plt.legend(frameon=False, prop={"size": 8})

        mean_raw = np.mean(noise_corr_raw)
        mean_den = np.mean(noise_corr_den)
        std_raw = np.std(noise_corr_raw)
        std_den = np.std(noise_corr_den)
        mode_raw = mode(noise_corr_raw)[0]
        mode_den = mode(noise_corr_den)[0]
        kurtosis_raw = np.mean(
            np.power(((noise_corr_raw - mean_raw) / std_raw), 4))
        kurtosis_den = np.mean(
            np.power(((noise_corr_den - mean_den) / std_den), 4))
        p_test_noise = scipy.stats.ttest_rel(noise_corr_raw, noise_corr_den)

        print("Mean of raw noise corr: " + str(mean_raw))
        print("Mean of den noise corr: " + str(mean_den))
        print("SEM of raw noise corr: " + str(scipy.stats.sem(noise_corr_raw)))
        print("SEM of den noise corr: " + str(scipy.stats.sem(noise_corr_den)))
        print("N: " + str(len(noise_corr_den)))
        print("Mode of raw noise: " + str(mode_raw))
        print("Mode of den noise: " + str(mode_den))
        print("Raw noise pearson skewness: " +
              str((mean_raw - mode_raw) / std_raw))
        print("Den noise pearson skewness: " +
              str((mean_den - mode_den) / std_den))
        print("Raw noise kurtosis: " + str(kurtosis_raw))
        print("Den noise kurtosis: " + str(kurtosis_den))
        print(p_test_noise)
        print("student t-test, pvalue : " + str(p_test_noise[1]))

        ax = self.fig.add_subplot(global_grid[7, 0])
        plt.text(-0.35, 1, "D", fontsize=15,
                 weight="bold", transform=ax.transAxes)

        plt.text(
            0.4,
            1.04,
            "Distribution of strong pairwise noise correction (> 0.4)",
            fontsize=8,
            weight="bold",
            transform=ax.transAxes,
        )

        self.plot_filters_on_top_image(
            ax,
            self.list_id_obj[0].list_masks,
            np.ones([512, 512]),
            percentile_off=True,
            axis_box=True,
        )
        self.plot_roi_connection(
            ax, self.list_id_obj[0].list_masks, example_noise_corr_raw, color=color_raw
        )

        rectangle_length = 100 * 512 / 400
        rect = matplotlib.patches.Rectangle(
            [20, 512 - 40], rectangle_length, 15, angle=0.0, color="k"
        )
        plt.gca().add_patch(rect)

        ax = self.fig.add_subplot(global_grid[7, 1])
        self.plot_filters_on_top_image(
            ax,
            self.list_id_obj[0].list_masks,
            np.ones([512, 512]),
            percentile_off=True,
            axis_box=True,
        )
        self.plot_roi_connection(
            ax, self.list_id_obj[0].list_masks, example_noise_corr_di, color=color_den
        )

    def plot_filters_on_top_image(
        self,
        ax,
        cell_masks,
        twop_image,
        top_percentile=98,
        bottom_percentile=2,
        percentile_off=False,
        axis_box=False,
    ):

        if not (percentile_off):
            top_limit = np.percentile(twop_image, top_percentile)
            bottom_limit = np.percentile(twop_image, bottom_percentile)

            twop_image[twop_image < bottom_limit] = bottom_limit
            twop_image[twop_image > top_limit] = top_limit

            twop_image = (twop_image - bottom_limit) / \
                (top_limit - bottom_limit)

        hue = np.zeros((512, 512))
        sat = np.zeros((512, 512))
        val = np.zeros((512, 512))

        for x in range(512):
            for y in range(512):
                [hue[x, y], sat[x, y], val[x, y]] = matplotlib.colors.rgb_to_hsv(
                    [twop_image[x, y], twop_image[x, y], twop_image[x, y]]
                )

        for index_mask, local_mask in enumerate(cell_masks):

            img = local_mask.get_mask_plane()
            # All ROI should be the same color to avoid confustion with colors in following plots
            hue[img > 0] = 0  # index_mask/len(self.list_masks)
            sat[img > 0] = 0.75

        final_rgb = np.zeros((512, 512, 3))
        for x in range(512):
            for y in range(512):
                final_rgb[x, y, :] = matplotlib.colors.hsv_to_rgb(
                    [hue[x, y], sat[x, y], val[x, y]]
                )

        plt.imshow(final_rgb)
        if not (axis_box):
            plt.axis("off")
        else:
            plt.gca().set_yticklabels([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_xticks([])

    def plot_hist_overlaid(self, raw_dat, den_dat):
        bins_on = np.arange(-0.5, 1, 0.05)
        [den_bins, final_bins] = np.histogram(np.array(den_dat), bins=bins_on)
        [raw_bins, final_bins] = np.histogram(np.array(raw_dat), bins=bins_on)

        plt.plot(bins_on[:-1], raw_bins, "#4A484A", label="Raw", linewidth=1)
        plt.fill_between(
            bins_on[:-1], raw_bins, color=(209 /
                                           255.0, 209 / 255.0, 209 / 255.0)
        )

        plt.plot(
            bins_on[:-1], den_bins, "#8A181A", label="DeepInterpolation", linewidth=1
        )

        plt.fill_between(
            bins_on[:-1], den_bins, color=(231 /
                                           255.0, 209 / 255.0, 210 / 255.0)
        )

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def plot_roi_connection(self, ax, cell_masks, connection, color):
        plt.sca(ax)
        nb_pairs = 0
        for input_mask in np.arange(len(self.list_id_obj[0].list_masks)):
            for output_mask in np.arange(
                input_mask + 1, len(self.list_id_obj[0].list_masks)
            ):

                local_conn = connection[input_mask, output_mask]
                if abs(local_conn) >= 0.4:
                    input_img = (
                        self.list_id_obj[0].list_masks[input_mask].get_mask_plane(
                        )
                    )
                    x_input, y_input = np.where(input_img > 0)
                    x_input = np.mean(x_input)
                    y_input = np.mean(y_input)

                    output_img = (
                        self.list_id_obj[0].list_masks[output_mask].get_mask_plane(
                        )
                    )
                    x_output, y_output = np.where(output_img > 0)
                    x_output = np.mean(x_output)
                    y_output = np.mean(y_output)

                    """
                    if local_conn>0:
                        color = np.array([81, 249, 50])/255.0
                    else:
                        color = np.array([252, 10, 238])/255.0
                    """
                    nb_pairs = nb_pairs + 1
                    plt.plot(
                        [y_input, y_output],
                        [x_input, x_output],
                        alpha=abs(local_conn),
                        linewidth=0.5,
                        color=color,
                    )
        print("Nb significant pairs is " + str(nb_pairs))

    def plot_example_reliability(self, ax, input_array, neuron_nb, color):

        local_shape = input_array.shape

        frame_rate = 30
        local_time = np.arange(0, input_array.shape[2]).astype(
            "float") * 1 / frame_rate

        nb_repeat = local_shape[1]

        f0 = np.mean(input_array[neuron_nb, :, :].flatten())
        hsv_color = matplotlib.colors.rgb_to_hsv(color)
        for index, local_repeat in enumerate(np.arange(nb_repeat - 1, -1, -1)):
            local_color = hsv_color

            local_color[1] = local_color[1] / 3 * (3 - index / nb_repeat)

            rgb_color = matplotlib.colors.hsv_to_rgb(local_color)
            local_trace = 100 * \
                (input_array[neuron_nb, local_repeat, :] - f0) / f0

            plt.plot(
                local_time,
                local_trace + 15 * local_repeat,
                label="Repeat " + str(local_repeat),
                color=rgb_color,
            )

        plt.xlabel("Time from start of stim (s)")
        plt.ylabel("$\Delta$F/F (%)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xlim([0, 10])


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 6 - signal_noise.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
