import h5py
import numpy as np
import matplotlib.pylab as plt
import os
import matplotlib
from scipy.stats import mode
import scipy
from scripts.plotting_helpers import placeAxesOnGrid
from matplotlib.patches import Rectangle
from pyitlib import discrete_random_variable as drv
import pathlib
import random


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file
        self.path_to_comp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)
                            ), "..", "data", "local_large_data"
        )
        self.path_to_sample_movie = lambda local_id: f"comp_{local_id}.h5"
        self.list_movies = [637998955, 505811062, 509904120, 657391625]
        self.path_to_mutual_info_anal = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "sup_fig4_intermediate.npy",
        )

        local_path = pathlib.Path(__file__).parent.absolute()

        self.path_schematic = os.path.join(
            local_path, "Supp Fig 5 - diagram.png")

    def load_data(self):

        self.list_id_obj = []

        # We first gather needed data

        self.list_movies_data = {}

        for index, local_id in enumerate(self.list_movies):
            local_path = self.path_to_sample_movie(local_id)
            [raw_sample, denoised_sample] = self.load_sample_movies(
                os.path.join(self.path_to_comp, local_path)
            )

            self.list_movies_data[local_id] = [raw_sample, denoised_sample]

    def load_sample_movies(self, local_path):
        with h5py.File(local_path, "r") as file_handle:
            denoised_sample = file_handle["data_proc"][:, :, :]
            raw_sample = file_handle["data_raw"][:, :, :]

        return [raw_sample, denoised_sample]

    def get_corr(self, array, list_pixel):
        array_dim = array.shape
        reshape_array = np.transpose(
            np.reshape(array, [array_dim[0], array_dim[1] * array_dim[2]])
        )
        corr_array = np.corrcoef(reshape_array[list_pixel, :])

        return corr_array

    def get_temporal_relationship_array(
        self,
        raw_data,
        proc_data,
        list_deltas=[-50, -10, -5, -2, -1, 0, 1, 2, 5, 10, 50],
        computation_frames=1000000,
        min_intensity=200,
    ):

        original_size = raw_data.shape

        full_range = np.max(np.abs(list_deltas))

        init_trace = raw_data[full_range:-full_range, 0, 0]
        length_trace = len(init_trace)

        average_frame = np.mean(raw_data, axis=2)
        [x_list, y_list] = np.where(average_frame > min_intensity)

        den_input_to_mutual = np.zeros(
            (len(list_deltas), length_trace * len(x_list)), dtype="uint16"
        )

        raw_input_to_mutual = np.zeros(
            (len(list_deltas), length_trace * len(x_list)), dtype="uint16"
        )

        for index_delta, delta in enumerate(list_deltas):
            start_index = 0
            print(index_delta)
            for index_list, dim1 in enumerate(x_list):
                dim2 = y_list[index_list]

                if -full_range + delta == 0:
                    local_trace_to_compare_den = proc_data[
                        full_range + delta:, dim1, dim2
                    ]
                    local_trace_to_compare_raw = raw_data[
                        full_range + delta:, dim1, dim2
                    ]
                else:
                    local_trace_to_compare_den = proc_data[
                        full_range + delta: -full_range + delta, dim1, dim2
                    ]
                    local_trace_to_compare_raw = raw_data[
                        full_range + delta: -full_range + delta, dim1, dim2
                    ]

                den_input_to_mutual[
                    index_delta, start_index: start_index + length_trace
                ] = local_trace_to_compare_den.astype("uint16")
                raw_input_to_mutual[
                    index_delta, start_index: start_index + length_trace
                ] = local_trace_to_compare_raw.astype("uint16")

                start_index = start_index + length_trace

        local_input_den = den_input_to_mutual[:,
                                              0:computation_frames].astype("uint16")
        local_input_raw = raw_input_to_mutual[:,
                                              0:computation_frames].astype("uint16")

        return [local_input_raw, local_input_den]

    def get_spatial_relationship_array(
        self,
        raw_data,
        proc_data,
        list_deltas=np.arange(-10, 10, 1),
        computation_frames=1000000,
    ):
        original_size = raw_data.shape

        average_frame = np.mean(raw_data, axis=2)
        [x_list, y_list] = np.where(average_frame > 200)

        to_keep = np.where(y_list < original_size[2] - np.max(list_deltas))[0]
        x_list = x_list[to_keep]
        y_list = y_list[to_keep]

        to_keep = np.where(y_list > -np.min(list_deltas))[0]
        x_list = x_list[to_keep]
        y_list = y_list[to_keep]

        den_input_to_mutual = np.zeros(
            (len(list_deltas), original_size[0] * len(x_list)), dtype="uint16"
        )

        raw_input_to_mutual = np.zeros(
            (len(list_deltas), original_size[0] * len(x_list)), dtype="uint16"
        )

        for index_delta, delta in enumerate(list_deltas):
            start_index = 0
            print(index_delta)
            for index_list, dim1 in enumerate(x_list):
                dim2 = y_list[index_list]
                if dim2 + delta < original_size[2]:
                    local_trace_to_compare_den = proc_data[:,
                                                           dim1, dim2 + delta]
                    local_trace_to_compare_raw = raw_data[:,
                                                          dim1, dim2 + delta]

                den_input_to_mutual[
                    index_delta, start_index: start_index + original_size[0]
                ] = local_trace_to_compare_den.astype("uint16")
                raw_input_to_mutual[
                    index_delta, start_index: start_index + original_size[0]
                ] = local_trace_to_compare_raw.astype("uint16")

                start_index = start_index + original_size[0]

        local_input_den = den_input_to_mutual[:,
                                              0:computation_frames].astype("uint16")
        local_input_raw = raw_input_to_mutual[:,
                                              0:computation_frames].astype("uint16")

        return [local_input_raw, local_input_den]

    def plot_corr_distrib(self, ax, list_corr_den, list_corr_raw):

        bins_on = np.arange(-1, 1, 0.01)
        [den_bins, final_bins] = np.histogram(
            np.array(list_corr_den), bins=bins_on)
        [raw_bins, final_bins] = np.histogram(
            np.array(list_corr_raw), bins=bins_on)

        plt.plot(bins_on[:-1], raw_bins, "#4A484A", label="Raw", linewidth=1)
        plt.fill_between(
            bins_on[:-1], raw_bins, color=(209 /
                                           255.0, 209 / 255.0, 209 / 255.0)
        )
        plt.plot(
            bins_on[:-1], den_bins, "#8A181A", label="Deep interpolation", linewidth=1
        )
        plt.fill_between(
            bins_on[:-1], den_bins, color=(231 /
                                           255.0, 209 / 255.0, 210 / 255.0)
        )

        plt.xlabel("Pearson correlation of individual pixels")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.legend(frameon=False, prop={"size": 8})

    def make_figure(self):
        self.fig = plt.figure(figsize=(18, 16))

        local_label = placeAxesOnGrid(
            self.fig, xspan=(0.01, 0.02), yspan=(0.01, 0.02))
        local_label.text(0, 0, "A", fontsize=15, weight="bold")
        local_label.axis("off")
        local_label.set_xlim(0, 0.01)
        local_label.set_ylim(0, 0.01)

        ax = placeAxesOnGrid(
            self.fig, dim=[2, 2], xspan=[0.05, 0.45], yspan=[0.05, 0.45]
        )

        [raw_sample, denoised_sample] = self.list_movies_data[637998955]

        list_pixel = np.random.randint(0, 512 * 512, size=10000)
        local_corr_raw = self.get_corr(raw_sample, list_pixel)
        local_corr_den = self.get_corr(denoised_sample, list_pixel)

        plt.sca(ax[0][0])
        plt.imshow(raw_sample[100, :, :], cmap="gray", clim=[100, 350])
        plt.title("Raw")
        vmin, vmax = plt.gci().get_clim()
        local_shape_raw = [512, 512]
        rectangle_length = 100 * local_shape_raw[0] / 400
        rect = matplotlib.patches.Rectangle(
            [30, local_shape_raw[0] - 60], rectangle_length, 30, angle=0.0, color="w"
        )
        ax[0][0].add_patch(rect)
        plt.axis("off")

        plt.sca(ax[0][1])
        plt.title("DeepInterpolation")
        plt.imshow(denoised_sample[100, :, :], cmap="gray", clim=[100, 350])
        plt.axis("off")

        plt.sca(ax[1][0])
        im_ax = plt.imshow(local_corr_raw, clim=[-0.5, 0.5])
        plt.xlabel("Random selection\nof 10,000 pixels")
        plt.ylabel("Random selection\nof 10,000 pixels")

        plt.sca(ax[1][1])
        plt.imshow(local_corr_den, clim=[-0.5, 0.5])
        plt.xlabel("Random selection\nof 10,000 pixels")
        plt.gca().get_yaxis().set_visible(False)
        plt.gca().get_yaxis().set_ticks([])

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.455, 0.46], yspan=[0.27, 0.48]
        )

        cb = plt.colorbar(im_ax, cax=ax)
        cb.set_label("Pearson correlation")

        local_label = placeAxesOnGrid(
            self.fig, xspan=(0.515, 0.525), yspan=(0.01, 0.02)
        )
        local_label.text(0, 0, "B", fontsize=15, weight="bold")
        local_label.axis("off")
        local_label.set_xlim(0, 0.01)
        local_label.set_ylim(0, 0.01)
        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.505, 0.795], yspan=[0.05, 0.23]
        )

        for index, indiv_id in enumerate(self.list_movies):
            [raw_sample, denoised_sample] = self.list_movies_data[indiv_id]
            list_pixel = np.random.randint(0, 512 * 512, size=10000)
            local_corr_raw = self.get_corr(raw_sample, c)
            local_corr_den = self.get_corr(denoised_sample, list_pixel)

            if index == 0:
                full_corr_raw = local_corr_raw
                full_corr_den = local_corr_den
            else:
                full_corr_raw = np.append(full_corr_raw, local_corr_raw)
                full_corr_den = np.append(full_corr_den, local_corr_den)

        result = scipy.stats.ks_2samp(
            full_corr_den[random.sample(range(0, 40000), 1000)],
            full_corr_raw[random.sample(range(0, 40000), 1000)],
        )
        print("Comparison of pixel correlation using KS two-sided test")
        print(result)
        print(
            "rawpixel corr, mean:{}, sem:{}, N={}".format(
                np.mean(full_corr_raw),
                scipy.stats.sem(full_corr_raw),
                len(full_corr_raw),
            )
        )
        print(
            "denoised pixel corr, mean:{}, sem:{}, N={}".format(
                np.mean(full_corr_den),
                scipy.stats.sem(full_corr_den),
                len(full_corr_den),
            )
        )

        self.plot_corr_distrib(ax, full_corr_den, full_corr_raw)
        plt.text(
            0.8,
            0.5,
            "N=40,000 pixels\n4 experiments",
            fontsize=8,
            transform=ax.transAxes,
        )

        local_label = placeAxesOnGrid(
            self.fig, xspan=(0.515, 0.525), yspan=(0.3, 0.31))
        local_label.text(0, 0, "C", fontsize=15, weight="bold")
        local_label.axis("off")
        local_label.set_xlim(0, 0.01)
        local_label.set_ylim(0, 0.01)

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.505, 0.795], yspan=[0.3, 0.45]
        )

        local_img = plt.imread(self.path_schematic)
        plt.imshow(local_img)
        plt.axis("off")

        local_label = placeAxesOnGrid(
            self.fig, xspan=(0.01, 0.02), yspan=(0.5, 0.51))
        local_label.text(0, 0, "D", fontsize=15, weight="bold")
        local_label.axis("off")
        local_label.set_xlim(0, 0.01)
        local_label.set_ylim(0, 0.01)

        ax = placeAxesOnGrid(
            self.fig, dim=[2, 2], xspan=[0.05, 0.795], yspan=[0.55, 0.95]
        )

        [raw_sample, denoised_sample] = self.list_movies_data[637998955]

        delta_space = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
        delta_time = [0, 1, 2, 3, 4, 5, 10, 25, 50]

        if os.path.isfile(self.path_to_mutual_info_anal):
            with open(self.path_to_mutual_info_anal, "rb") as f:
                mi_den_space = np.load(f)
                mi_raw_space = np.load(f)
                mi_den_time = np.load(f)
                mi_raw_time = np.load(f)
                corr_den_space = np.load(f)
                corr_raw_space = np.load(f)
                corr_den_time = np.load(f)
                corr_raw_time = np.load(f)
        else:
            [
                data_relation_raw_space,
                data_relation_den_space,
            ] = self.get_spatial_relationship_array(
                raw_sample, denoised_sample, list_deltas=delta_space
            )
            [
                data_relation_raw_time,
                data_relation_den_time,
            ] = self.get_temporal_relationship_array(
                raw_sample, denoised_sample, list_deltas=delta_time
            )

            print("Computing mutual information")
            print(data_relation_den_space.shape)
            mi_den_space = drv.information_mutual(data_relation_den_space)
            mi_raw_space = drv.information_mutual(data_relation_raw_space)
            mi_den_time = drv.information_mutual(data_relation_den_time)
            mi_raw_time = drv.information_mutual(data_relation_raw_time)

            print("Computing cross-correlation")
            corr_den_space = np.corrcoef(data_relation_den_space)
            corr_raw_space = np.corrcoef(data_relation_raw_space)
            corr_den_time = np.corrcoef(data_relation_den_time)
            corr_raw_time = np.corrcoef(data_relation_raw_time)

            with open(self.path_to_mutual_info_anal, "wb") as f:
                np.save(f, mi_den_space)
                np.save(f, mi_raw_space)
                np.save(f, mi_den_time)
                np.save(f, mi_raw_time)
                np.save(f, corr_den_space)
                np.save(f, corr_raw_space)
                np.save(f, corr_den_time)
                np.save(f, corr_raw_time)

        plt.sca(ax[0][0])
        plt.plot(
            np.array(delta_time) / 30,
            mi_den_time[0, :],
            label="DeepInterpolation",
            color="#8A181A",
        )
        plt.plot(
            np.array(delta_time) / 30, mi_raw_time[0, :], label="Raw", color="#4A484A"
        )

        plt.ylabel("Pixel Mutual information (bits)")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        new_tick_locations = np.arange(0, 1, 5 / 30)
        new_tick_str = np.arange(0, 30, 5)
        plt.xlim([0, 1])

        ax2 = ax[0][0].twiny()
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(new_tick_str)
        ax2.set_xlabel(r"Relative frame number from center frame")

        ax[0][0].add_patch(
            matplotlib.patches.Rectangle(
                xy=(0, 0),  # point of origin.
                width=0.2,
                height=1,
                linewidth=1,
                linestyle="dashed",
                color="black",
                fill=False,
            )
        )
        zoom_ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.15, 0.35], yspan=[0.575, 0.625]
        )
        plt.sca(zoom_ax)
        plt.plot(
            np.array(delta_time) / 30,
            mi_den_time[0, :],
            label="DeepInterpolation",
            color="#8A181A",
        )
        plt.plot(
            np.array(delta_time) / 30, mi_raw_time[0, :], label="Raw", color="#4A484A"
        )
        plt.xlim([0, 0.2])
        plt.ylim([0, 1])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.sca(ax[0][1])
        plt.plot(
            np.array(delta_space) * 400 / 512,
            mi_den_space[0, :],
            label="DeepInterpolation",
            color="#8A181A",
        )
        plt.plot(
            np.array(delta_space) * 400.0 / 512.0,
            mi_raw_space[0, :],
            label="Raw",
            color="#4A484A",
        )

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlim([0, 10])
        plt.legend(frameon=False, prop={"size": 8})
        plt.text(
            0.5,
            0.5,
            "N=1,000,000 pixels\n1 experiment",
            fontsize=8,
            transform=ax[0][1].transAxes,
        )

        def pix2um(x):
            return x * 400.0 / 512

        def um2pix(x):
            return x * 512.0 / 400

        ax2 = ax[0][1].secondary_xaxis("top", functions=(um2pix, pix2um))
        ax2.set_xlabel(r"Relative position from center pixel (pixels)")

        plt.sca(ax[1][0])
        plt.plot(
            np.array(delta_time) / 30,
            corr_den_time[0, :],
            label="DeepInterpolation",
            color="#8A181A",
        )
        plt.plot(
            np.array(delta_time) / 30, corr_raw_time[0, :], label="Raw", color="#4A484A"
        )

        plt.xlabel("Relative time from center frame (s)")
        plt.ylabel("Pearson correlation")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlim([0, 1])

        plt.sca(ax[1][1])
        plt.plot(
            np.array(delta_space) * 400 / 512,
            corr_den_space[0, :],
            label="DeepInterpolation",
            color="#8A181A",
        )
        plt.plot(
            np.array(delta_space) * 400 / 512,
            corr_raw_space[0, :],
            label="Raw",
            color="#4A484A",
        )

        plt.xlabel("Relative position from center pixel ($\mu$m)")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

        plt.xlim([0, 10])

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 3 - pixel_corr.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
