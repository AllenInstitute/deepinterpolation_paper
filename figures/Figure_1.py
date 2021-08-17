import pathlib
import re
import os
import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib
from scripts.plotting_helpers import placeAxesOnGrid
from matplotlib.patches import Rectangle
from read_roi import read_roi_zip
import cv2
import h5py as h5
import scipy.stats

matplotlib.use("TkAgg")
matplotlib.rcParams.update({"font.size": 8})

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['pdf.fonttype'] = 42

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file
        local_path = pathlib.Path(__file__).parent.absolute()

        self.path_training = os.path.join(local_path, "../data/*_running_terminal.txt")
        self.path_schematic = os.path.join(local_path, "Figure_1_top_panel.png")
        self.path_30_30_padding = os.path.join(
            local_path, "../data/2020-09-04-loss_30_30_padding.npy"
        )
        self.list_example = []
        self.list_example.append(
            os.path.join(
            local_path, "../data/validation_out_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0000-raw-example.png")
        )
        self.list_example.append(
            os.path.join(
            local_path, "../data/validation_out_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0025-example.png")
        )
        self.list_example.append(
            os.path.join(
            local_path, "../data/validation_out_2019_09_11_23_32_unet_single_1024_mean_absolute_error_Ai93-0450-example.png")
        )
        self.denoise_example = os.path.join(
            local_path, "../data/local_large_data/comp_505811062.h5"
        )
        self.roi_examples = os.path.join(local_path, "../data/505811062_roi.zip")
        self.snr_data = os.path.join(local_path, "../data/snr_voxels_distrib.npy")

    def load_data(self):
        # We first gather needed data
        self.get_external_files()

    def make_figure(self):
        self.fig = plt.figure(figsize=(10, 10))

        ax = placeAxesOnGrid(self.fig, xspan=[0, 0.95], yspan=[0.05, 0.66])

        plt.text(0, 1, "A", fontsize=15, weight="bold", transform=ax.transAxes)

        self.plot_schematic_2p(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.11, 0.33], yspan=[0.6, 0.98])

        self.plot_training_loss(ax)

        # panel letter
        plt.text(-0.5, 1.35, "B", fontsize=15, weight="bold", transform=ax.transAxes)

        ax = placeAxesOnGrid(
            self.fig, dim=[3, 1], xspan=[0.33, 0.55], yspan=[0.59, 0.99]
        )

        plt.sca(ax[0])
        plt.text(0.75, 1.05, "Raw frame", fontsize=8, transform=ax[0].transAxes)
        self.plot_single_example_img(ax[0], self.list_example[0], scale_bar=True)
        plt.sca(ax[1])

        plt.text(
            0.25,
            1.05,
            "After training (25k samples)",
            fontsize=8,
            transform=ax[1].transAxes,
        )
        self.plot_single_example_img(ax[1], self.list_example[1])
        plt.sca(ax[2])

        plt.text(
            0.25,
            1.05,
            "After training (450k samples)",
            fontsize=8,
            transform=ax[2].transAxes,
        )
        self.plot_single_example_img(ax[2], self.list_example[2])

        ax = placeAxesOnGrid(
            self.fig, dim=[3, 1], xspan=[0.45, 0.7], yspan=[0.59, 0.99]
        )

        plt.sca(ax[0])
        self.plot_single_example_img(ax[0], self.list_example[0], zoomed=True)
        plt.sca(ax[1])

        self.plot_single_example_img(ax[1], self.list_example[1], zoomed=True)
        plt.sca(ax[2])

        self.plot_single_example_img(ax[2], self.list_example[2], zoomed=True)

        ax = placeAxesOnGrid(
            self.fig, dim=[7, 1], xspan=[0.72, 0.95], yspan=[0.6, 0.78]
        )
        # panel letter
        plt.text(
            -0.25, 1.15, "C", fontsize=15, weight="bold", transform=ax[0].transAxes
        )

        self.plot_example_traces(ax)

        ax = placeAxesOnGrid(self.fig, xspan=[0.65, 0.95], yspan=[0.85, 0.98])

        # panel letter
        plt.text(0.05, 1.05, "D", fontsize=15, weight="bold", transform=ax.transAxes)

        self.plot_snr_distrib(ax)

        """
        ax = self.fig.add_subplot(global_grid[1, 1])

        self.plot_final_training_loss(ax)
        plt.text(-0.4, 1, "C", fontsize=20,
                 weight="bold", transform=ax.transAxes)
        """

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def get_external_files(self):
        self.list_files = glob.glob(self.path_training)

    def plot_snr_distrib(self, ax):
        out_data = np.load(self.snr_data)

        list_snr_raw = out_data[0]
        list_snr_den = out_data[1]

        bins_on = np.arange(0, 80, 0.5)
        [snr_den_bins, final_bins] = np.histogram(np.array(list_snr_den), bins=bins_on)
        [snr_raw_bins, final_bins] = np.histogram(np.array(list_snr_raw), bins=bins_on)

        print("Mean of raw snr: " + str(np.mean(list_snr_raw)))
        print("Mean of den snr: " + str(np.mean(list_snr_den)))
        print("SEM of raw snr: " + str(scipy.stats.sem(list_snr_raw)))
        print("SEM of den snr: " + str(scipy.stats.sem(list_snr_den)))
        print("mean increase of snr: " + str(np.mean(list_snr_den / list_snr_raw)))
        print(
            "SEM of increase snr: " + str(scipy.stats.sem(list_snr_den / list_snr_raw))
        )

        print("Number units :" + str(len(list_snr_den)))

        plt.plot(
            bins_on[:-1],
            snr_raw_bins / np.max(snr_raw_bins),
            "#4A484A",
            label="Raw",
            linewidth=1,
        )
        plt.fill_between(
            bins_on[:-1],
            snr_raw_bins / np.max(snr_raw_bins),
            color=(209 / 255.0, 209 / 255.0, 209 / 255.0),
        )
        plt.plot(
            bins_on[:-1],
            snr_den_bins / np.max(snr_den_bins),
            "#8A181A",
            label="DeepInterpolation",
            linewidth=1,
        )
        plt.fill_between(
            bins_on[:-1],
            snr_den_bins / np.max(snr_den_bins),
            color=(231 / 255.0, 209 / 255.0, 210 / 255.0),
        )

        plt.xlabel("single pixel SNR")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.legend(frameon=False, prop={"size": 6})

    def plot_schematic_2p(self, ax):
        local_img = plt.imread(self.path_schematic)
        plt.imshow(local_img)
        plt.axis("off")

    def plot_example_traces(self, ax):
        dict_roi = read_roi_zip(self.roi_examples)

        list_img = []
        for indiv_key in dict_roi.keys():
            local_roi = dict_roi[indiv_key]
            local_img = np.zeros([512, 512]).astype("int")

            if local_roi["type"] == "rectangle":
                local_img[local_roi["left"], local_roi["top"]] = 255
            else:
                x_coord = np.array(local_roi["x"]).astype("int")
                y_coord = np.array(local_roi["y"]).astype("int")
                local_img[x_coord, y_coord] = 255
                local_img = np.ascontiguousarray(local_img, dtype=np.uint8)
                local_img = cv2.fillConvexPoly(
                    local_img,
                    np.array(
                        [
                            (y_coord[index], x_coord[index])
                            for index in range(len(x_coord))
                        ]
                    ),
                    255,
                )

            list_img.append(local_img)

        with h5.File(self.denoise_example, r"r") as file_handle:
            raw_data = file_handle["data_raw"][:, :, :]
            proc_data = file_handle["data_proc"][:, :, :]

        for index, indiv_roi in enumerate(list_img):
            [x, y] = np.where(indiv_roi == 255)
            local_trace_raw = np.zeros(raw_data.shape[0])
            local_trace_den = np.zeros(raw_data.shape[0])

            for indiv_frame in range(raw_data.shape[0]):
                local_trace_raw[indiv_frame] = np.mean(raw_data[indiv_frame, x, y])
                local_trace_den[indiv_frame] = np.mean(proc_data[indiv_frame, x, y])

            if index >= 3:
                if index == 3:
                    plt.sca(ax[index])
                    plt.axis("off")

                plt.sca(ax[index + 1])
                if index == 3:
                    plt.title("Single pixels", fontsize=8)
            else:
                plt.sca(ax[index])
                if index == 0:
                    plt.title("Somatic ROIs", fontsize=8)

            local_trace_raw = (
                100
                * (local_trace_raw - np.median(local_trace_raw))
                / np.median(local_trace_raw)
            )
            local_trace_den = (
                100
                * (local_trace_den - np.median(local_trace_den))
                / np.median(local_trace_den)
            )
            local_time = 1 / 30 * np.arange(local_trace_den.shape[0])
            plt.plot(local_time, local_trace_raw, color="#4A484A", linewidth=1)
            plt.plot(local_time, local_trace_den, color="#8A181A", linewidth=1)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)

            if index < 5:
                plt.gca().spines["bottom"].set_visible(False)
                plt.gca().axes.xaxis.set_ticks([])
                plt.gca().axes.xaxis.set_ticklabels([])
            else:
                plt.xlabel("Time (s)")

            if index == 4 or index == 1:
                plt.ylabel("$\Delta$F/F (%)")

    def plot_final_training_loss(self, ax):
        orders_index = np.argsort(self.list_mean)

        self.list_label = [
            self.list_label[sorted_index] for sorted_index in orders_index[::-1]
        ]
        self.list_mean = [
            self.list_mean[sorted_index] for sorted_index in orders_index[::-1]
        ]

        plt.bar(self.list_label, self.list_mean)
        plt.ylim([0.5225, 0.525])

        plt.xlabel("Input frames")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("validation reconstrucion loss")
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    def plot_single_example_img(self, ax, path, scale_bar=False, zoomed=False):
        img_data = plt.imread(path)

        if zoomed:
            img_data = img_data[256:384, 256:384]

        plt.sca(ax)

        plt.imshow(img_data, cmap="gray")
        plt.axis("off")
        local_shape = img_data.shape

        if scale_bar:
            rectangle_length = 100 * local_shape[0] / 400
            rect = matplotlib.patches.Rectangle(
                [20, local_shape[0] - 40], rectangle_length, 15, angle=0.0, color="w"
            )
            ax.add_patch(rect)

        if zoomed:
            rect_zoom = [0, 1, 0, 1]
        else:
            rect_zoom = [0.5, 0.75, 0.5, 0.75]

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
                linewidth=1,
            )
            ax.add_patch(rect)
        plt.axis("off")

    def plot_training_loss(self, ax):
        self.list_label = []
        self.list_mean = []
        loss_to_look_list = ["val_loss: "]
        averaging_batches = 3

        for each_file in self.list_files:
            for loss_to_look in loss_to_look_list:
                with open(each_file, "r") as path_handle:
                    txt_terminal = path_handle.readlines()

                    list_training = []
                    previous_line = []
                    pre_previous_line = []
                    for line in txt_terminal:
                        if ("Epoch" in line) and ("val_loss" in previous_line):
                            index = previous_line.find(loss_to_look)
                            local_value = float(
                                previous_line[index + len(loss_to_look) :]
                            )
                            list_training.append(local_value)
                        elif ("Epoch" in line) and ("val_loss" in pre_previous_line):
                            index = pre_previous_line.find(loss_to_look)
                            pre_previous_line[index + len(loss_to_look) :]
                            list_training.append(local_value)

                        pre_previous_line = previous_line
                        previous_line = line
                list_training = np.convolve(
                    list_training,
                    np.ones((averaging_batches,)) / averaging_batches,
                    mode="valid",
                )
                local_label = os.path.basename(each_file)[-43:-24]

                if len(list_training) > 41:
                    list_training = list_training[0:50]
                index = local_label.find("pre")
                index_training = 2500 * np.arange(1, 1 + len(list_training))
                list_numbers = re.findall(r"\d+", local_label[index:])
                pre = int(list_numbers[0])
                post = int(list_numbers[1])
                local_label_constr = "Npre=" + str(pre) + " Npost=" + str(post)
                plt.plot(
                    index_training,
                    list_training,
                    label=local_label_constr,
                    linewidth=1.5,
                )

                try:
                    local_to_average = list_training[40:50]
                    if len(local_to_average) > 0:
                        local_mean = np.mean(local_to_average)
                        self.list_label.append((local_label_constr))
                        self.list_mean.append(local_mean)
                except:
                    local_mean = 0

        pre = 30
        post = 30
        local_label_constr = "Npre=" + str(pre) + " Npost=" + str(post)
        samples_per_epoch = 1000
        (epoch_3030, valid3030) = np.load(self.path_30_30_padding)

        valid3030 = np.convolve(
            valid3030, np.ones((averaging_batches,)) / averaging_batches, mode="valid",
        )
        epoch_3030 = np.convolve(
            epoch_3030, np.ones((averaging_batches,)) / averaging_batches, mode="valid",
        )
        plt.plot(
            samples_per_epoch * epoch_3030,
            valid3030,
            label=local_label_constr,
            linewidth=1.5,
        )
        index = np.argmin(valid3030)
        print(
            "Found minimum at "
            + str(samples_per_epoch * epoch_3030[index])
            + " samples"
        )
        plt.xlabel("number of unique samples")
        plt.ylabel("validation reconstrucion loss")
        plt.xticks([0, 200000, 400000])
        plt.gca().get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
        )
        plt.axvline(x=np.max(index_training), c="k", linestyle="dotted")
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.legend(frameon=False, prop={"size": 6})


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pdfs", "Figure 1 - training.pdf"
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
