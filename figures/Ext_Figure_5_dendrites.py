from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import h5py
import numpy as np
import matplotlib.pylab as plt
import rastermap
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
import scipy.stats.stats as stats
import os
import matplotlib
from scripts.plotting_helpers import placeAxesOnGrid


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

    def load_data(self):
        # We first gather needed data
        self.exp_id = 637998955
        self.path_to_segmentation = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "637998955",
            "suite2p",
            "plane0 - somas - old",
        )

        self.stim_frame_rate = 60

        self.get_segmented_denoised()

        self.get_segmented_raw()

        self.get_reorg_sweeps()

        (self.max_frame, self.max_amplitude) = self.get_max_frame_response(
            self.natural_movie_denf
        )

        self.aver_corr_all_cells = self.get_aver_corr(self.natural_movie_denf)

        self.aver_corr_non_cell = self.aver_corr_all_cells[self.is_cell == False]
        self.aver_corr_cell = self.aver_corr_all_cells[self.is_cell]

        self.all_tracesall_cells = self.get_cell_image_filters()

        self.neuron_soma = self.all_cells[:, :, np.where(self.is_cell)[0]]
        self.neuron_dend = self.all_cells[:, :,
                                          np.where(self.is_cell == False)[0]]

    def init_layout(self):
        # We create the panels
        self.fig = plt.figure(figsize=(25, 10))

    def fill_layout(self):

        ax = placeAxesOnGrid(self.fig, dim=[2, 2], xspan=[
                             0, 0.25], yspan=[0.05, 0.95])
        plt.sca(ax[0][0])
        plt.text(
            -0.4, 1.1, "A", fontsize=20, weight="bold", transform=ax[0][0].transAxes
        )
        self.plot_example_reliability(ax[0][0], self.natural_movie_denf, 21)
        plt.xlabel("")
        ax[0][0].get_xaxis().set_visible(False)
        ax[0][0].spines["bottom"].set_visible(False)
        plt.title("soma roi 1")

        plt.sca(ax[1][0])
        self.plot_example_reliability(ax[1][0], self.natural_movie_denf, 239)
        plt.title("non-somatic roi 1")

        plt.sca(ax[0][1])
        self.plot_example_reliability(ax[0][1], self.natural_movie_denf, 10)
        plt.xlabel("")
        ax[0][1].get_xaxis().set_visible(False)
        ax[0][1].spines["bottom"].set_visible(False)
        plt.ylabel("")
        plt.title("soma roi 2")

        plt.sca(ax[1][1])
        self.plot_example_reliability(ax[1][1], self.natural_movie_denf, 252)
        plt.ylabel("")
        plt.title("non-somatic roi 2")

        ax = placeAxesOnGrid(self.fig, xspan=[0.3, 0.5], yspan=[0, 0.02])
        plt.text(-0.15, 2, "B", fontsize=20,
                 weight="bold", transform=ax.transAxes)
        self.plot_preferred_frame_time_colorbar(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.525, 0.725], yspan=[0, 0.02])
        plt.text(-0.15, 2, "C", fontsize=20,
                 weight="bold", transform=ax.transAxes)
        self.plot_preferred_frame_amplitude_colorbar(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.75, 0.95], yspan=[0, 0.02])
        plt.text(-0.10, 2, "D", fontsize=20,
                 weight="bold", transform=ax.transAxes)
        self.plot_reliability_colorbar(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.3, 0.5], yspan=[0.05, 0.575])
        self.plot_preferred_frame_time_map(ax)

        ax = placeAxesOnGrid(
            self.fig, xspan=[0.525, 0.725], yspan=[0.05, 0.575])
        self.plot_preferred_frame_amplitude_map(ax)

        ax = placeAxesOnGrid(
            self.fig, xspan=[0.750, 0.95], yspan=[0.05, 0.575])
        self.plot_reliability_map(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.35, 0.5], yspan=[0.6, 0.95])
        self.plot_distribution_preferred_time(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.575, 0.725], yspan=[0.6, 0.95])
        self.plot_distribution_amplitude(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.8, 0.95], yspan=[0.6, 0.95])
        self.plot_distribution_reliability(ax)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def get_segmented_denoised(self):
        raw_traces = "F.npy"
        sorting = "iscell.npy"
        cell_stat = "stat.npy"

        traces_path = os.path.join(self.path_to_segmentation, raw_traces)
        sorting_path = os.path.join(self.path_to_segmentation, sorting)
        cell_path = os.path.join(self.path_to_segmentation, cell_stat)

        self.all_traces = np.load(traces_path)
        self.is_cell = np.load(sorting_path)[:, 0] == 1
        self.cell_filters = np.load(cell_path, allow_pickle=True)

    def get_segmented_raw(self):
        # We get the metadata from the sdk
        boc = BrainObservatoryCache()
        exp = boc.get_ophys_experiment_data(self.exp_id)
        st = exp.get_stimulus_table("natural_movie_three")

        stim_table = exp.get_stimulus_table(
            "natural_movie_three"
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

        nb_neurons = self.all_traces.shape[0]

        # Reorganization of data
        for neuron_nb in np.arange(0, nb_neurons):
            for index_repeat, indiv_repeat in enumerate(self.list_repeat):
                list_rows_start2 = self.all_start_frames["repeat"] == indiv_repeat
                list_rows_end2 = self.all_end_frames["repeat"] == indiv_repeat

                list_rows_start_selected = self.all_start_frames[list_rows_start2]
                list_rows_end_selected = self.all_end_frames[list_rows_end2]

                local_start = int(list_rows_start_selected["start"])
                local_end = int(list_rows_end_selected["end"])

                local_trace_den = self.all_traces[
                    neuron_nb, local_start - 30: local_end - 30
                ]

                if neuron_nb == 0 and index_repeat == 0:
                    natural_movie_denf = np.zeros(
                        [nb_neurons, len(self.list_repeat),
                         len(local_trace_den)]
                    )

                natural_movie_denf[
                    neuron_nb, index_repeat, 0: len(local_trace_den)
                ] = local_trace_den

        self.natural_movie_denf = natural_movie_denf

    def get_max_frame_response(self, input_array):

        average_trial = np.mean(input_array, axis=1)
        max_frame = np.argmax(average_trial, axis=1)
        max_amplitude = np.zeros(input_array.shape[0])
        for index_neuron in range(input_array.shape[0]):
            f0 = np.mean(input_array[index_neuron, :, :].flatten())
            max_amplitude[index_neuron] = (
                100 * (average_trial[index_neuron,
                       max_frame[index_neuron]] - f0) / f0
            )

        return (max_frame, max_amplitude)

    def get_aver_corr(self, input_array):
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
            reliability[index_neuron] = np.nanmean(
                corr_matrix[inds[0], inds[1]])

        return reliability

    def get_cell_image_filters(self):
        # We recreate all cells filters
        self.all_cells = np.zeros((512, 512, len(self.cell_filters)))

        for neuron_nb in range(len(self.cell_filters)):
            list_x = self.cell_filters[neuron_nb]["xpix"]
            list_y = self.cell_filters[neuron_nb]["ypix"]
            weight = self.cell_filters[neuron_nb]["lam"]
            self.all_cells[list_x, list_y, neuron_nb] = weight
            local_img = self.all_cells[:, :, neuron_nb]
            self.all_cells[:, :, neuron_nb] = local_img / \
                np.max(local_img.flatten())

    def plot_example_reliability(self, ax, input_array, neuron_nb):

        local_shape = input_array.shape
        frame_rate = 30
        local_time = np.arange(0, input_array.shape[2]).astype(
            "float") * 1 / frame_rate

        nb_repeat = local_shape[1]

        f0 = np.mean(input_array[neuron_nb, :, :].flatten())

        for local_repeat in np.arange(nb_repeat - 1, -1, -1):
            local_trace = 100 * \
                (input_array[neuron_nb, local_repeat, :] - f0) / f0
            plt.plot(
                local_time,
                local_trace + 15 * local_repeat,
                label="Repeat " + str(local_repeat),
            )

        plt.xlabel("Time from\nstart of movie (s)")
        plt.ylabel("$\Delta$F/F (%)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.xlim([0, 10])

    def plot_reliability_map(self, ax):

        hue = np.zeros((512, 512))
        sat = np.ones((512, 512))
        val = np.max(self.all_cells, axis=2)

        for neuron_nb in range(len(self.cell_filters)):
            local_img = self.all_cells[:, :, neuron_nb]
            hue[local_img > 0] = (
                self.aver_corr_all_cells[neuron_nb] -
                np.min(self.aver_corr_all_cells)
            ) / np.max(self.aver_corr_all_cells)

        final_rgb = np.zeros((512, 512, 3))
        for x in range(512):
            for y in range(512):
                final_rgb[x, y, :] = matplotlib.colors.hsv_to_rgb(
                    [hue[x, y], sat[x, y], val[x, y]]
                )

        plt.imshow(final_rgb)
        plt.axis("off")

    def plot_reliability_colorbar(self, ax):

        cmap = matplotlib.cm.hsv
        norm = matplotlib.colors.Normalize(
            vmin=np.min(self.aver_corr_all_cells), vmax=np.max(self.aver_corr_all_cells)
        )

        cb1 = matplotlib.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        plt.title("Response reliability")

    def plot_distribution_reliability(self, ax):

        n, bins, patches = plt.hist(
            self.aver_corr_non_cell, label="non-somatic", alpha=0.4, density=True
        )
        plt.hist(self.aver_corr_cell, label="somas",
                 alpha=0.4, bins=bins, density=True)
        plt.xlabel("Reliability")
        plt.ylabel("Density of units")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

    def plot_preferred_frame_time_map(self, ax):

        hue = np.zeros((512, 512))
        sat = np.ones((512, 512))
        val = np.max(self.all_cells, axis=2)

        max_max_frame = np.max(self.max_frame)

        for neuron_nb in range(len(self.cell_filters)):
            local_img = self.all_cells[:, :, neuron_nb]
            hue[local_img > 0] = self.max_frame[neuron_nb] / max_max_frame

        final_rgb = np.zeros((512, 512, 3))
        for x in range(512):
            for y in range(512):
                final_rgb[x, y, :] = matplotlib.colors.hsv_to_rgb(
                    [hue[x, y], sat[x, y], val[x, y]]
                )

        plt.imshow(final_rgb)
        plt.axis("off")

        rectangle_length = 100 * final_rgb.shape[0] / 400
        rect = matplotlib.patches.Rectangle(
            [20, final_rgb.shape[0] - 40], rectangle_length, 15, angle=0.0, color="w"
        )
        ax.add_patch(rect)

    def plot_preferred_frame_amplitude_map(self, ax):
        hue = np.zeros((512, 512))
        sat = np.ones((512, 512))
        val = np.max(self.all_cells, axis=2)

        max_max_amplitude = np.max(self.max_amplitude)

        for neuron_nb in range(len(self.cell_filters)):
            local_img = self.all_cells[:, :, neuron_nb]
            hue[local_img > 0] = self.max_amplitude[neuron_nb] / max_max_amplitude

        final_rgb = np.zeros((512, 512, 3))
        for x in range(512):
            for y in range(512):
                final_rgb[x, y, :] = matplotlib.colors.hsv_to_rgb(
                    [hue[x, y], sat[x, y], val[x, y]]
                )

        plt.imshow(final_rgb)
        plt.axis("off")

    def plot_preferred_frame_time_colorbar(self, ax):
        max_max_frame = np.max(self.max_frame)
        cmap = matplotlib.cm.hsv
        norm = matplotlib.colors.Normalize(
            vmin=0, vmax=max_max_frame / self.stim_frame_rate
        )

        cb1 = matplotlib.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        plt.title("Preferred Frame time (s)")

    def plot_distribution_preferred_time(self, ax):
        max_frame_som = self.max_frame[self.is_cell]
        max_frame_dend = self.max_frame[self.is_cell == False]

        n, bins, patches = plt.hist(
            max_frame_dend / self.stim_frame_rate,
            label="non-somatic",
            alpha=0.4,
            density=True,
        )

        plt.hist(
            max_frame_som / self.stim_frame_rate,
            label="somas",
            alpha=0.4,
            bins=bins,
            density=True,
        )
        plt.xlabel("Preferred Frame time (s)")
        plt.ylabel("Density of units")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend()

    def plot_preferred_frame_amplitude_colorbar(self, ax):
        max_max_amplitude = np.max(self.max_amplitude)

        cmap = matplotlib.cm.hsv
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_max_amplitude)

        cb1 = matplotlib.colorbar.ColorbarBase(
            ax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        plt.title("Preferred frame response amplitude (%$\Delta$F/F)")

    def plot_distribution_amplitude(self, ax):
        max_amplitude_som = self.max_amplitude[self.is_cell]
        max_amplitude_dend = self.max_amplitude[self.is_cell == False]

        n, bins, patches = plt.hist(
            max_amplitude_dend, label="non-somatic", alpha=0.4, density=True
        )
        plt.hist(max_amplitude_som, label="somas",
                 alpha=0.4, bins=bins, density=True)
        plt.xlabel("Preferred frame response amplitude (%$\Delta$F/F)")
        plt.ylabel("Density of units")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 10 - dendrite_soma_analysis.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)
    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.init_layout()
    local_figure.fill_layout()
    local_figure.save_figure()
