import matplotlib
import h5py
import numpy as np
import os
import scipy.stats
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import pathlib
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import rastermap
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scripts.plotting_helpers import placeAxesOnGrid
from matplotlib.patches import Rectangle
from PIL import Image
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import scipy.stats.stats as stats
from scipy.stats import mode


class LocalAnalysisSNR:
    def __init__(self, path_to_raw_h5, exp_id, suite2p_folder="plane0"):
        self.path_to_raw_h5 = path_to_raw_h5
        self.path_to_segmentation = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            str(exp_id),
            "suite2p",
            suite2p_folder,
        )
        self.exp_id = exp_id
        self.path_to_raw_traces = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            str(self.exp_id) + "-denoised_filters_on_raw.npy",
        )

    def load_data(self):
        self.get_segmented_denoised()
        self.get_cell_image_filters()
        self.get_segmented_raw()
        self.get_stim_table()
        self.get_reorg_sweeps()

    def plot_filters_overlaid(
        self, axis_box=False,
    ):

        hue = 255 * np.ones((512, 512))
        sat = 255 * np.ones((512, 512))
        val = 0 * np.ones((512, 512))

        for index_mask in np.arange(self.all_cells.shape[2]):

            img = self.all_cells[:, :, index_mask]
            # All ROI should be the same color to avoid confustion with colors in following plots
            hue[img > 0] = 0 * np.random.rand(1)  # index_mask/len(self.list_masks)
            sat[img > 0] = 0.75
            val[img > 0] = 255 * img[img > 0]

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

    def plot_signal_corr_overlaid(self, connection, threshold=0.4, color="red"):
        nb_connect = 0
        for input_mask in np.arange(self.all_cells.shape[2]):
            for output_mask in np.arange(input_mask + 1, self.all_cells.shape[2]):

                local_conn = connection[input_mask, output_mask]
                if abs(local_conn) >= threshold:
                    nb_connect = nb_connect + 1
                    input_img = self.all_cells[:, :, input_mask]
                    x_input, y_input = np.where(input_img > 0)
                    x_input = np.mean(x_input)
                    y_input = np.mean(y_input)

                    output_img = self.all_cells[:, :, output_mask]
                    x_output, y_output = np.where(output_img > 0)
                    x_output = np.mean(x_output)
                    y_output = np.mean(y_output)

                    plt.plot(
                        [y_input, y_output],
                        [x_input, x_output],
                        alpha=abs(local_conn),
                        linewidth=0.5,
                        color=color,
                    )
        return nb_connect

    def get_stim_table(self):

        # We get the metadata from the sdk
        boc = BrainObservatoryCache()
        exp = boc.get_ophys_experiment_data(self.exp_id)
        st = exp.get_stimulus_table("natural_movie_one")

        stim_table = exp.get_stimulus_table(
            "natural_movie_one"
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

        print("Downloaded stim table for " + str(self.exp_id))

    def get_reliability_neurons(self, input_array):
        nb_repeat = input_array.shape[1]
        nb_neurons = input_array.shape[0]

        reliability = np.zeros([nb_neurons])
        for index_neuron in range(nb_neurons):
            corr_matrix = np.zeros((nb_repeat, nb_repeat))
            for i in range(nb_repeat):
                for j in range(i + 1, nb_repeat):
                    r, p = stats.pearsonr(
                        input_array[index_neuron, i, :],
                        input_array[index_neuron, j, :],
                    )
                    corr_matrix[i, j] = r
            inds = np.triu_indices(nb_repeat, k=1)
            reliability[index_neuron] = np.nanmean(corr_matrix[inds[0], inds[1]])

        return reliability

    def get_reorg_sweeps(self):

        nb_neurons = self.denoised_f.shape[0]
        cell_index = np.where(self.is_cell)[0]
        # Reorganization of data
        for index, neuron_nb in enumerate(cell_index):
            for index_repeat, indiv_repeat in enumerate(self.list_repeat):
                list_rows_start2 = self.all_start_frames["repeat"] == indiv_repeat
                list_rows_end2 = self.all_end_frames["repeat"] == indiv_repeat

                list_rows_start_selected = self.all_start_frames[list_rows_start2]
                list_rows_end_selected = self.all_end_frames[list_rows_end2]

                local_start = int(list_rows_start_selected["start"])
                local_end = int(list_rows_end_selected["end"])

                local_trace_den = self.denoised_f[
                    neuron_nb, local_start - 30 : local_end - 30
                ]
                local_trace_raw = self.raw_f[neuron_nb, local_start:local_end]

                if index == 0 and index_repeat == 0:
                    natural_movie_denf = np.zeros(
                        [len(cell_index), len(self.list_repeat), len(local_trace_den)]
                    )
                    natural_movie_rawf = np.zeros(
                        [len(cell_index), len(self.list_repeat), len(local_trace_raw)]
                    )

                if len(local_trace_den) > natural_movie_denf.shape[2]:
                    local_trace_den = local_trace_den[0 : natural_movie_denf.shape[2]]
                if len(local_trace_raw) > natural_movie_rawf.shape[2]:
                    local_trace_raw = local_trace_raw[0 : natural_movie_rawf.shape[2]]

                natural_movie_denf[
                    index, index_repeat, 0 : len(local_trace_den)
                ] = local_trace_den
                natural_movie_rawf[
                    index, index_repeat, 0 : len(local_trace_raw)
                ] = local_trace_raw

        self.natural_movie_denf = natural_movie_denf
        self.natural_movie_rawf = natural_movie_rawf

    def get_segmented_denoised(self):
        raw_traces = "F.npy"
        sorting = "iscell.npy"
        cell_stat = "stat.npy"

        traces_path = os.path.join(self.path_to_segmentation, raw_traces)
        sorting_path = os.path.join(self.path_to_segmentation, sorting)
        cell_path = os.path.join(self.path_to_segmentation, cell_stat)

        self.denoised_f = np.load(traces_path)
        self.is_cell = np.load(sorting_path)[:, 0] == 1
        self.cell_filters = np.load(cell_path, allow_pickle=True)
        self.cell_filters = self.cell_filters[self.is_cell]

    def get_cell_image_filters(self):
        # We recreate all cells filters
        self.all_cells = np.zeros((512, 512, len(self.cell_filters)))

        for neuron_nb in range(len(self.cell_filters)):
            list_x = self.cell_filters[neuron_nb]["xpix"]
            list_y = self.cell_filters[neuron_nb]["ypix"]
            weight = self.cell_filters[neuron_nb]["lam"]
            self.all_cells[list_y, list_x, neuron_nb] = weight

    def get_snr_traces(self, local_traces):
        interval_noise = 100
        max_frames = local_traces.shape[1]

        list_blocks = np.arange(0, max_frames - interval_noise, interval_noise)
        list_noise = np.zeros([local_traces.shape[0], len(list_blocks)])

        for index, start_range in enumerate(list_blocks):
            short_mov = local_traces[:, start_range : (start_range + interval_noise)]
            local_noise_di = np.std(
                short_mov, axis=1
            )  # scipy.stats.median_absolute_deviation

            list_noise[:, index] = local_noise_di

        current_noise_di = np.median(list_noise, axis=1)

        local_signal_di = np.median(local_traces, axis=1)
        snr_di = local_signal_di / current_noise_di

        # remove nan snr
        snr_di = snr_di[snr_di > 0]
        return [snr_di, local_signal_di, current_noise_di]

    def get_snr_di_raw(self):
        [snr_cell_pop_raw, local_signal_raw, current_noise_raw] = self.get_snr_traces(
            self.raw_f[self.is_cell, :]
        )
        [snr_cell_pop_di, local_signal_di, current_noise_di] = self.get_snr_traces(
            self.denoised_f[self.is_cell, :]
        )

        return [snr_cell_pop_raw, snr_cell_pop_di]

    def plot_snr_comparison(self):
        [snr_cell_pop_raw, local_signal_raw, current_noise_raw] = self.get_snr_traces(
            self.raw_f[self.is_cell, :]
        )
        [snr_cell_pop_di, local_signal_di, current_noise_di] = self.get_snr_traces(
            self.denoised_f[self.is_cell, :]
        )

        plt.plot(snr_cell_pop_raw, snr_cell_pop_di, ".")
        plt.plot(snr_cell_pop_raw, snr_cell_pop_raw, "r-")

        plt.xlabel("SNR from ROI - raw")
        plt.ylabel("SNR from ROI - DeepInterpolation")

    def get_signal_noise_corr(self, input_array):
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

    def get_segmented_raw(self):

        if os.path.isfile(self.path_to_raw_traces):
            self.raw_f = np.load(self.path_to_raw_traces)
        else:
            nb_mask = self.all_cells.shape[2]
            with h5py.File(self.path_to_raw_h5, "r") as h5_handle:
                local_data = h5_handle["data"]
                nb_frames = local_data.shape[0]
                reshape_cell = self.all_cells
                reshape_cell = np.reshape(
                    reshape_cell,
                    [self.all_cells.shape[0] * self.all_cells.shape[1], -1],
                )
                self.raw_f = np.zeros([nb_mask, nb_frames])
                block = 100
                for local_frame_index in np.arange(0, nb_frames, block):
                    final_index = np.min([local_frame_index + block, nb_frames])
                    local_frame = local_data[local_frame_index:final_index, :, :]
                    local_frame = local_frame.reshape(
                        [-1, self.all_cells.shape[0] * self.all_cells.shape[1]]
                    )

                    self.raw_f[:, local_frame_index:final_index] = np.transpose(
                        np.dot(local_frame, reshape_cell)
                    )

            np.save(self.path_to_raw_traces, self.raw_f)


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.list_ids = [637998955, 505811062, 510021399]
        self.list_raw_folders = [
            r"\\allen\programs\braintv\production\neuralcoding\prod33\specimen_606651548\ophys_session_637923516\ophys_experiment_637998955\processed\concat_31Hz_0.h5",
            r"\\allen\programs\braintv\production\neuralcoding\prod12\specimen_501139891\ophys_experiment_505811062\processed\concat_31Hz_0.h5",
            r"\\allen\programs\braintv\production\neuralcoding\prod6\specimen_502185594\ophys_experiment_510021399\processed\concat_31Hz_0.h5",
        ]

        self.path_raster_traces = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "526912861 - F.npy",
        )

        self.path_raster_is_cell = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "526912861  - iscell.npy",
        )

        self.raster_somas = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "526912861-somas.png",
        )

        self.raster_else = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "526912861-everything_else.png",
        )

        self.list_path_segmentation_examples = [
            os.path.join(
                os.path.dirname(__file__), "..", "data", "505811062_projection.png"
            ),
            os.path.join(
                os.path.dirname(__file__), "..", "data", "510021399_projection.png"
            ),
            os.path.join(
                os.path.dirname(__file__), "..", "data", "637998955_projection.png"
            ),
        ]

        self.list_path_segmentation_sorted = [
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "local_large_data",
                "714778152-somas",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "local_large_data",
                "714778152-proximal",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "local_large_data",
                "714778152-apicals",
            ),
        ]

        self.path_to_rois_example_folder = os.path.join(
            os.path.dirname(__file__), "..", "data", "examples_rois"
        )

    def load_data(self):
        self.get_snr_rel_rois()
        self.get_raster_data()

    def get_path_segmentation(self, root_path):
        F_path = os.path.join(root_path, "F.npy")
        is_cell_path = os.path.join(root_path, "iscell.npy")
        png_path = os.path.join(root_path, "projection.png")

        return (F_path, is_cell_path, png_path)

    def get_raster_data(self):
        self.raster_traces = np.load(self.path_raster_traces)
        self.raster_is_cell = np.load(self.path_raster_is_cell)

    def get_snr_rel_rois(self):
        self.list_snr_cell_pop_raw = [None] * len(self.list_ids)
        self.list_snr_cell_pop_di = [None] * len(self.list_ids)

        self.list_rel_cell_pop_raw = [None] * len(self.list_ids)
        self.list_rel_cell_pop_di = [None] * len(self.list_ids)

        # All cells
        for index, local_id in enumerate(self.list_ids):
            local_path = self.list_raw_folders[index]
            X = LocalAnalysisSNR(local_path, local_id)
            X.load_data()
            [
                self.list_snr_cell_pop_raw[index],
                self.list_snr_cell_pop_di[index],
            ] = X.get_snr_di_raw()

            self.list_rel_cell_pop_raw[index] = X.get_reliability_neurons(
                X.natural_movie_rawf
            )
            self.list_rel_cell_pop_di[index] = X.get_reliability_neurons(
                X.natural_movie_denf
            )

        # Somas
        self.list_snr_cell_pop_raw_somas = [None] * 3
        self.list_snr_cell_pop_di_somas = [None] * 3

        self.list_rel_cell_pop_raw_somas = [None] * 3
        self.list_rel_cell_pop_di_somas = [None] * 3

        for index, local_id in enumerate(self.list_ids):
            local_path = self.list_raw_folders[index]
            X = LocalAnalysisSNR(local_path, local_id, suite2p_folder="plane0 - somas")
            X.load_data()
            [
                self.list_snr_cell_pop_raw_somas[index],
                self.list_snr_cell_pop_di_somas[index],
            ] = X.get_snr_di_raw()

            self.list_rel_cell_pop_raw_somas[index] = X.get_reliability_neurons(
                X.natural_movie_rawf
            )
            self.list_rel_cell_pop_di_somas[index] = X.get_reliability_neurons(
                X.natural_movie_denf
            )

        self.snr_cell_pop_raw_all = np.concatenate(self.list_snr_cell_pop_raw)
        self.snr_cell_pop_di_all = np.concatenate(self.list_snr_cell_pop_di)

        self.snr_cell_pop_raw_somas = np.concatenate(self.list_snr_cell_pop_raw_somas)
        self.snr_cell_pop_di_somas = np.concatenate(self.list_snr_cell_pop_di_somas)

        self.rel_cell_pop_raw_all = np.concatenate(self.list_rel_cell_pop_raw)
        self.rel_cell_pop_di_all = np.concatenate(self.list_rel_cell_pop_di)

        self.rel_cell_pop_raw_somas = np.concatenate(self.list_rel_cell_pop_raw_somas)
        self.rel_cell_pop_di_somas = np.concatenate(self.list_rel_cell_pop_di_somas)

    def plot_traces_example(self, ax, path_F, path_iscell, time_axis=True):
        traces = np.load(path_F)
        is_cell = np.load(path_iscell)

        index_keep = np.where(is_cell[:, 0])[0]

        filtered_traces = traces[index_keep, :]

        nb_traces = filtered_traces.shape[0]
        max_plotted = 25
        frame_rate = 30
        local_time = (
            np.arange(0, filtered_traces.shape[1]).astype("float") * 1 / frame_rate
        )

        for local_index in np.arange(max_plotted, 0, -1):
            local_trace = filtered_traces[local_index, :]
            f0 = np.mean(local_trace)
            local_trace = 100 * (local_trace - f0) / f0
            plt.plot(local_time, local_trace + 100 * local_index, linewidth=1)

        if time_axis:
            rect = matplotlib.patches.Rectangle(
                [-2, 0], 10, 0.02, angle=0.0, color="k",
            )
            plt.gca().add_patch(rect)
            rect = matplotlib.patches.Rectangle(
                [-2, 0], 0.2, 500, angle=0.0, color="k",
            )
            plt.gca().add_patch(rect)
            plt.text(
                0.04, -0.04, "10s ", fontsize=8, transform=plt.gca().transAxes,
            )
            plt.text(
                -0.15,
                0.05,
                "500 %\n$\Delta$F/F (%)",
                fontsize=8,
                transform=plt.gca().transAxes,
            )

        plt.xlabel("Time (s)")
        plt.ylabel("$\Delta$F/F (%)")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        if time_axis == False:
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().spines["bottom"].set_visible(False)
        plt.axis("off")
        plt.xlim([-4, 100])

    def plot_projection_img(
        self, ax, path_to_img, zoom=[0, 1, 0, 1], rect_zoom=[], scale_bar=False
    ):
        local_img = plt.imread(path_to_img)
        local_shape = local_img.shape
        left = int(np.round(local_shape[0] * zoom[0]))
        right = int(np.round(local_shape[0] * zoom[1]))
        bottom = int(np.round(local_shape[1] * zoom[2]))
        top = int(np.round(local_shape[1] * zoom[3]))
        zoomed_img = local_img[left:right, bottom:top]

        plt.imshow(zoomed_img)

        local_shape_raw = local_shape
        local_shape = zoomed_img.shape

        if scale_bar:
            rectangle_length = 100 * local_shape_raw[0] / 400
            rect = matplotlib.patches.Rectangle(
                [40, local_shape_raw[0] - 80],
                rectangle_length,
                30,
                angle=0.0,
                color="w",
            )
            plt.gca().add_patch(rect)

        if len(rect_zoom) > 0:
            zoom_left = int(np.round(local_shape[0] * rect_zoom[0]))
            zoom_right = int(np.round(local_shape[0] * rect_zoom[1]))
            zoom_bottom = int(np.round(local_shape[1] * rect_zoom[2]))
            zoom_top = int(np.round(local_shape[1] * rect_zoom[3]))
            rect = Rectangle(
                (zoom_left, zoom_bottom),
                zoom_right - zoom_left,
                zoom_top - zoom_bottom,
                fill=False,
                color="red",
                facecolor="None",
                linestyle="dashed",
                clip_on=False,
            )
            ax.add_patch(rect)
        plt.axis("off")

    def plot_traces_raster_example(self, traces_F):
        # Keep the good ones
        out = traces_F

        # We z-score each unit
        out_z = zscore(out, axis=1)

        # remove annoying nans to detect
        list_column = np.max(out_z, axis=1) > 0
        out_z = out_z[list_column, :]

        # Run the embedding
        model = rastermap.mapping.Rastermap(n_components=1, n_X=50).fit(out_z)
        isort = np.argsort(model.embedding[:, 0])

        # Project on the new base
        X = out_z[isort, :]

        # this is for visualization purposes
        Sm = gaussian_filter1d(X.T, np.minimum(3, int(X.shape[0] * 0.005)), axis=1)
        d = Sm.T

        frame_rate = 30
        local_time = np.arange(0, X.shape[1]).astype("float") * 1 / frame_rate

        plt.imshow(
            X,
            "gray",
            clim=[-5, 10],
            aspect="auto",
            extent=[0, local_time[-1], X.shape[0], 0],
        )
        plt.xlabel("Time (s)")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlim([0, 300])

    def plot_snr_densities(self, ax):

        plt.plot(
            self.snr_cell_pop_raw_all,
            self.snr_cell_pop_di_all,
            ".",
            color=np.array([182, 182, 182]) / 255,
            label="All ROIs",
        )
        plt.plot(
            self.snr_cell_pop_raw_somas,
            self.snr_cell_pop_di_somas,
            ".",
            color=np.array([255, 116, 77]) / 255,
            label="Somatic ROIs",
        )
        plt.plot(
            self.snr_cell_pop_raw_all, self.snr_cell_pop_raw_all, "k--",
        )

        plt.legend(frameon=False, prop={"size": 9})
        plt.xlabel("SNR\n of raw traces")
        plt.ylabel("SNR after\nDeepInterpolation")
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 60])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        print(
            "raw all ROI snr, mean:{}, sem:{}, N={}".format(
                np.mean(self.snr_cell_pop_raw_all),
                scipy.stats.sem(self.snr_cell_pop_raw_all),
                len(self.snr_cell_pop_raw_all),
            )
        )

        print(
            "DI all ROI snr, mean:{}, sem:{}, N={}".format(
                np.mean(self.snr_cell_pop_di_all),
                scipy.stats.sem(self.snr_cell_pop_di_all),
                len(self.snr_cell_pop_di_all),
            )
        )

        print(
            "raw all ROI snr, mean:{}, sem:{}, N={}".format(
                np.mean(self.snr_cell_pop_raw_somas),
                scipy.stats.sem(self.snr_cell_pop_raw_somas),
                len(self.snr_cell_pop_raw_somas),
            )
        )

        print(
            "DI all ROI snr, mean:{}, sem:{}, N={}".format(
                np.mean(self.snr_cell_pop_di_somas),
                scipy.stats.sem(self.snr_cell_pop_di_somas),
                len(self.snr_cell_pop_di_somas),
            )
        )

    def plot_nb_roi_threshold(self, ax):
        list_raw = []
        list_di = []
        for exp in range(3):
            nb_raw_units_higherSNR = len(
                np.where(self.list_snr_cell_pop_raw[exp] > 20)[0]
            )
            nb_di_units_higherSNR = len(
                np.where(self.list_snr_cell_pop_di[exp] > 20)[0]
            )
            list_raw.append(nb_raw_units_higherSNR)
            list_di.append(nb_di_units_higherSNR)

            plt.plot(
                ["Raw", "DeepInterpolation"],
                [nb_raw_units_higherSNR, nb_di_units_higherSNR],
                "k",
            )

        print(
            "raw nb units>20, mean:{}, sem:{}, N={}".format(
                np.mean(list_raw), scipy.stats.sem(list_raw), len(list_raw)
            )
        )

        print(
            "di nb units >20, mean:{}, sem:{}, N={}".format(
                np.mean(list_di), scipy.stats.sem(list_di), len(list_di)
            )
        )

        plt.ylabel("# ROI with SNR\nabove 20")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

    def plot_roi_profile(self, ax, path_to_roi, scale_bar=False):
        im = Image.open(path_to_roi)
        imarray = np.array(im)
        orig_shape = imarray.shape

        # We locate the ROI center
        value = np.max(imarray.flatten())
        square_size = 100
        [x, y] = np.where(imarray == value)

        xmin = np.max([0, x - square_size / 2])
        xmax = np.min([orig_shape[0], xmin + square_size])

        if (xmax - xmin) < square_size:
            xmax = np.min([orig_shape[0], x + square_size / 2])
            xmin = np.max([0, xmax - square_size])

        ymin = np.max([0, y - square_size / 2])
        ymax = np.min([orig_shape[1], ymin + square_size])

        if (ymax - ymin) < square_size:
            ymax = np.min([orig_shape[0], y + square_size / 2])
            ymin = np.max([0, ymax - square_size])

        final_img = imarray[int(xmin) : int(xmax), int(ymin) : int(ymax)]
        cmin = np.percentile(final_img.flatten(), 0)
        cmax = np.percentile(final_img.flatten(), 99.97)
        plt.imshow(final_img, cmap="gray", clim=[cmin, cmax])  # inferno")
        plt.axis("off")

        if scale_bar:
            rectangle_length = 50 * imarray.shape[0] / 400
            rect = matplotlib.patches.Rectangle(
                [5, final_img.shape[0] - 15], rectangle_length, 5, angle=0.0, color="w",
            )
            ax.add_patch(rect)

    def plot_reliability_comparison(self):
        plt.plot(
            self.rel_cell_pop_raw_all,
            self.rel_cell_pop_di_all,
            ".",
            color=np.array([182, 182, 182]) / 255,
            label="All ROIs",
        )
        plt.plot(
            self.rel_cell_pop_raw_somas,
            self.rel_cell_pop_di_somas,
            ".",
            color=np.array([255, 116, 77]) / 255,
            label="Somatic ROIs",
        )

        plt.plot(
            self.rel_cell_pop_raw_somas, self.rel_cell_pop_raw_somas, "k--",
        )

        plt.xlabel("Response reliability to natural movie in raw data")
        plt.ylabel("Response reliability to natural movie\nafter DeepInterpolation")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend(frameon=False, prop={"size": 9})

        non_nan_roi = np.logical_not(np.isnan(self.rel_cell_pop_di_all))
        print(
            "reliability all ROI raw , mean:{}, sem:{}, N={}".format(
                np.mean(self.rel_cell_pop_raw_all[non_nan_roi]),
                scipy.stats.sem(self.rel_cell_pop_raw_all[non_nan_roi]),
                len(self.rel_cell_pop_raw_all[non_nan_roi]),
            )
        )

        print(
            "reliability all ROI di , mean:{}, sem:{}, N={}".format(
                np.mean(self.rel_cell_pop_di_all[non_nan_roi]),
                scipy.stats.sem(self.rel_cell_pop_di_all[non_nan_roi]),
                len(self.rel_cell_pop_di_all[non_nan_roi]),
            )
        )

        print(
            "reliability somas ROI raw , mean:{}, sem:{}, N={}".format(
                np.mean(self.rel_cell_pop_raw_somas),
                scipy.stats.sem(self.rel_cell_pop_raw_somas),
                len(self.rel_cell_pop_raw_somas),
            )
        )

        print(
            "reliability somas ROI di , mean:{}, sem:{}, N={}".format(
                np.mean(self.rel_cell_pop_di_somas),
                scipy.stats.sem(self.rel_cell_pop_di_somas),
                len(self.rel_cell_pop_di_somas),
            )
        )

    def plot_corr_overlaid_filters(
        self, type="denoised", color="red", somas=False, scale_bar=False
    ):
        local_path = self.list_raw_folders[0]
        local_id = 637998955
        if somas:
            X = LocalAnalysisSNR(local_path, local_id, suite2p_folder="plane0 - somas")
        else:
            X = LocalAnalysisSNR(local_path, local_id)

        X.load_data()

        if type == "denoised":
            [
                signal_corr_array,
                noise_corr_array,
                full_sign_corr,
                full_noise_corr,
            ] = X.get_signal_noise_corr(X.natural_movie_denf)
        else:
            [
                signal_corr_array,
                noise_corr_array,
                full_sign_corr,
                full_noise_corr,
            ] = X.get_signal_noise_corr(X.natural_movie_rawf)

        X.plot_filters_overlaid()
        nb_connect = X.plot_signal_corr_overlaid(full_noise_corr, color=color)
        print(
            "Nb significant connection :"
            + str(nb_connect)
            + " "
            + type
            + "-somas:"
            + str(somas)
        )
        if scale_bar:
            rectangle_length = 100 * 512 / 400
            rect = matplotlib.patches.Rectangle(
                [10, 512 - 20], rectangle_length, 10, angle=0.0, color="w",
            )
            plt.gca().add_patch(rect)

    def get_batch_sign_noise_corr(self):

        sig_corr_raw = np.array([])
        sig_corr_den = np.array([])
        noise_corr_raw = np.array([])
        noise_corr_den = np.array([])

        for index, local_id in enumerate(self.list_ids):

            local_obj = LocalAnalysisSNR(self.list_raw_folders[index], local_id)

            local_obj.load_data()

            [
                signal_corr_array_raw,
                noise_corr_array_raw,
                full_sign_corr_raw,
                full_noise_corr_raw,
            ] = local_obj.get_signal_noise_corr(local_obj.natural_movie_rawf)
            [
                signal_corr_array_di,
                noise_corr_array_di,
                full_sign_corr_den,
                full_noise_corr_den,
            ] = local_obj.get_signal_noise_corr(local_obj.natural_movie_denf)
            if index == 0:
                example_noise_corr_raw = full_noise_corr_raw
                example_noise_corr_di = full_noise_corr_den

            sig_corr_raw = np.append(sig_corr_raw, signal_corr_array_raw.flatten(), 0)
            sig_corr_den = np.append(sig_corr_den, signal_corr_array_di.flatten(), 0)
            noise_corr_raw = np.append(
                noise_corr_raw, noise_corr_array_raw.flatten(), 0
            )
            noise_corr_den = np.append(noise_corr_den, noise_corr_array_di.flatten(), 0)

        return [sig_corr_raw, sig_corr_den, noise_corr_raw, noise_corr_den]

    def plot_hist_overlaid(self, raw_dat, den_dat):
        bins_on = np.arange(-0.5, 1, 0.05)
        [den_bins, final_bins] = np.histogram(np.array(den_dat), bins=bins_on)
        [raw_bins, final_bins] = np.histogram(np.array(raw_dat), bins=bins_on)

        plt.plot(bins_on[:-1], raw_bins, "#4A484A", label="Raw", linewidth=1)
        plt.fill_between(
            bins_on[:-1], raw_bins, color=(209 / 255.0, 209 / 255.0, 209 / 255.0)
        )

        plt.plot(
            bins_on[:-1], den_bins, "#8A181A", label="DeepInterpolation", linewidth=1
        )

        plt.fill_between(
            bins_on[:-1], den_bins, color=(231 / 255.0, 209 / 255.0, 210 / 255.0)
        )

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])

    def make_figure(self):

        self.fig = plt.figure(figsize=(25, 19))

        ax = placeAxesOnGrid(
            self.fig,
            dim=[2, 3],
            xspan=[0.005, 0.35],
            yspan=[0.005, 0.35],
            wspace=0.1,
            hspace=0.1,
        )

        for local_index, local_path in enumerate(self.list_path_segmentation_examples):
            plt.sca(ax[0][local_index])
            if local_index == 0:
                scale_bar = True
                plt.text(
                    -0.15,
                    1,
                    "A",
                    fontsize=20,
                    weight="bold",
                    transform=ax[local_index][0].transAxes,
                )

            else:
                scale_bar = False
            self.plot_projection_img(
                ax[0][local_index],
                local_path,
                rect_zoom=[0.25, 0.75, 0.25, 0.75],
                scale_bar=scale_bar,
            )

        for local_index, local_path in enumerate(self.list_path_segmentation_examples):
            plt.sca(ax[1][local_index])
            self.plot_projection_img(
                ax[1][local_index],
                local_path,
                zoom=[0.25, 0.75, 0.25, 0.75],
                rect_zoom=[0, 1, 0, 1],
            )

        ax = placeAxesOnGrid(self.fig, xspan=[0.40, 0.6], yspan=[0.38, 0.58])
        plt.text(
            0, 1.1, "D", fontsize=20, weight="bold", transform=ax.transAxes,
        )

        self.plot_snr_densities(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.65, 0.75], yspan=[0.38, 0.58])
        self.plot_nb_roi_threshold(ax)

        ax = placeAxesOnGrid(
            self.fig,
            dim=[3, 2],
            xspan=[0.025, 0.35],
            yspan=[0.37, 0.825],
            wspace=0.1,
            hspace=0.2,
        )

        for index_local, local_path_root in enumerate(
            self.list_path_segmentation_sorted
        ):
            plt.sca(ax[index_local][1])
            (F_path, iscell_path, png_path) = self.get_path_segmentation(
                local_path_root
            )
            if index_local == 2:
                self.plot_traces_example(
                    ax[index_local][1], F_path, iscell_path, time_axis=True
                )
            else:
                self.plot_traces_example(
                    ax[index_local][1], F_path, iscell_path, time_axis=False
                )

            plt.sca(ax[index_local][0])

            if index_local == 0:
                plt.text(
                    -0.25,
                    1,
                    "C",
                    fontsize=20,
                    weight="bold",
                    transform=ax[index_local][0].transAxes,
                )
            if index_local == 2:
                self.plot_projection_img(ax, png_path, scale_bar=True)
            else:
                self.plot_projection_img(ax, png_path)

            if index_local == 0:
                plt.title("Somas")
            elif index_local == 1:
                plt.title("horizontal dendrites/axons")
            elif index_local == 2:
                plt.title("vertical dendrites/axons")

        ax = placeAxesOnGrid(
            self.fig, dim=[2, 6], xspan=[0.385, 0.975], yspan=[0.025, 0.35]
        )
        index_row = -1
        offset_column = -1

        for index_local, local_folder in enumerate(
            os.listdir(self.path_to_rois_example_folder)
        ):
            if not ("DS_Store" in local_folder):
                index_row += 1
                list_files = os.listdir(
                    os.path.join(self.path_to_rois_example_folder, local_folder)
                )
                # We reset row if we reach the bottom
                if index_row == 2:
                    index_row = 0
                    offset_column = 2
                
                index_column = offset_column                

                for index_file, local_file in enumerate(list_files):
                    if not ("DS_Store" in local_file):
                        index_column += 1       
                                 
                        print(index_row, index_column)

                        plt.sca(ax[index_row][index_column])
                        if index_row == 0 and index_column == 0:
                            plt.text(
                                -0.15,
                                1.2,
                                "B",
                                fontsize=20,
                                weight="bold",
                                transform=ax[index_row][index_column].transAxes,
                            )

                        if index_row == 0 and index_column == 2:
                            plt.text(
                                0,
                                1.2,
                                "Automatically segmented filters",
                                fontsize=15,
                                transform=ax[index_row][index_column].transAxes,
                            )

                        if index_column == 1 or index_column == 4:
                            plt.title(local_folder)
                        if index_column == 0 and index_row == 1:
                            scale_bar = True
                        else:
                            scale_bar = False

                        self.plot_roi_profile(
                            ax[index_row][index_column],
                            os.path.join(
                                self.path_to_rois_example_folder,
                                local_folder,
                                local_file,
                            ),
                            scale_bar=scale_bar,
                        )

        ax = placeAxesOnGrid(self.fig, dim=[1, 1], xspan=[0.8, 0.95], yspan=[0.38, 0.58])

        plt.text(
            -0.32, 1.1, "E", fontsize=20, weight="bold", transform=ax.transAxes,
        )

        self.plot_reliability_comparison()

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 2], xspan=[0.40, 0.65], yspan=[0.65, 0.825]
        )
        plt.text(
            -0.1, 0.95, "F", fontsize=20, weight="bold", transform=ax[0].transAxes,
        )
        [
            sig_corr_raw,
            sig_corr_den,
            noise_corr_raw,
            noise_corr_den,
        ] = self.get_batch_sign_noise_corr()

        plt.sca(ax[0])
        self.plot_hist_overlaid(sig_corr_raw, sig_corr_den)
        plt.xlabel("signal correlation")
        plt.text(
            0.7,
            0.8,
            "N=" + str(len(sig_corr_raw)) + " pairs",
            fontsize=8,
            transform=ax[0].transAxes,
        )

        plt.sca(ax[1])
        self.plot_hist_overlaid(noise_corr_raw, noise_corr_den)
        plt.xlabel("noise correlation")

        plt.legend(frameon=False, prop={"size": 9})

        ax = placeAxesOnGrid(self.fig, dim=[1, 2], xspan=[0.7, 0.95], yspan=[0.65, 0.825])
        plt.sca(ax[0])
        plt.text(
            -0.4, 1.0, "G", fontsize=20, weight="bold", transform=ax[0].transAxes,
        )

        plt.text(
            0.15,
            1.15,
            "Distribution of strong pairwise noise correction (> 0.4)",
            fontsize=10,
            transform=ax[0].transAxes,
        )
        local_figure.plot_corr_overlaid_filters(
            scale_bar=True, type="raw", color=np.array([182, 182, 182]) / 255
        )
        plt.title("All ROIs - Raw", fontsize=8)
        plt.sca(ax[1])
        local_figure.plot_corr_overlaid_filters(
            type="denoised", color=np.array([182, 182, 182]) / 255
        )
        plt.title("All ROIs - DeepInterpolation", fontsize=8)

        sig_corr_den = sig_corr_den[~np.isnan(sig_corr_raw)]
        sig_corr_raw = sig_corr_raw[~np.isnan(sig_corr_raw)]

        sig_corr_raw = sig_corr_raw[~np.isnan(sig_corr_den)]
        sig_corr_den = sig_corr_den[~np.isnan(sig_corr_den)]

        noise_corr_den = noise_corr_den[~np.isnan(noise_corr_raw)]
        noise_corr_raw = noise_corr_raw[~np.isnan(noise_corr_raw)]

        noise_corr_raw = noise_corr_raw[~np.isnan(noise_corr_den)]
        noise_corr_den = noise_corr_den[~np.isnan(noise_corr_den)]

        mean_raw = np.mean(sig_corr_raw)
        mean_den = np.mean(sig_corr_den)
        std_raw = np.std(sig_corr_raw)
        std_den = np.std(sig_corr_den)
        mode_raw = mode(sig_corr_raw)[0]
        mode_den = mode(sig_corr_den)[0]

        kurtosis_raw = np.mean(np.power(((sig_corr_raw - mean_raw) / std_raw), 4))
        kurtosis_den = np.mean(np.power(((sig_corr_den - mean_den) / std_den), 4))
        p_test_signal = scipy.stats.ttest_rel(sig_corr_raw, sig_corr_den)

        print("Mean of raw sign corr: " + str(mean_raw))
        print("Mean of den sign corr: " + str(mean_den))
        print("SEM of raw sign corr: " + str(scipy.stats.sem(sig_corr_raw)))
        print("SEM of den sign corr: " + str(scipy.stats.sem(sig_corr_den)))
        print("N: " + str(len(sig_corr_den)))
        print("Mode of raw signal: " + str(mode_raw))
        print("Mode of den signal: " + str(mode_den))
        print("Raw signal pearson skewness: " + str((mean_raw - mode_raw) / std_raw))
        print("Den signal pearson skewness: " + str((mean_den - mode_den) / std_den))
        print("Raw signal kurtosis: " + str(kurtosis_raw))
        print("Den signal kurtosis: " + str(kurtosis_den))
        print("student t-test, pvalue : " + str(p_test_signal[1]))

        mean_raw = np.mean(noise_corr_raw)
        mean_den = np.mean(noise_corr_den)
        std_raw = np.std(noise_corr_raw)
        std_den = np.std(noise_corr_den)
        mode_raw = mode(noise_corr_raw)[0]
        mode_den = mode(noise_corr_den)[0]
        kurtosis_raw = np.mean(np.power(((noise_corr_raw - mean_raw) / std_raw), 4))
        kurtosis_den = np.mean(np.power(((noise_corr_den - mean_den) / std_den), 4))
        p_test_noise = scipy.stats.ttest_rel(noise_corr_raw, noise_corr_den)

        print("Mean of raw noise corr: " + str(mean_raw))
        print("Mean of den noise corr: " + str(mean_den))
        print("SEM of raw noise corr: " + str(scipy.stats.sem(noise_corr_raw)))
        print("SEM of den noise corr: " + str(scipy.stats.sem(noise_corr_den)))
        print("N: " + str(len(noise_corr_den)))
        print("Mode of raw noise: " + str(mode_raw))
        print("Mode of den noise: " + str(mode_den))
        print("Raw noise pearson skewness: " + str((mean_raw - mode_raw) / std_raw))
        print("Den noise pearson skewness: " + str((mean_den - mode_den) / std_den))
        print("Raw noise kurtosis: " + str(kurtosis_raw))
        print("Den noise kurtosis: " + str(kurtosis_den))
        print(p_test_noise)
        print("student t-test, pvalue : " + str(p_test_noise[1]))

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Figure 2 - impact on segmentation.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
