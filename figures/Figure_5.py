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
from scripts.plotting_helpers import placeAxesOnGrid
from matplotlib.patches import Rectangle
from PIL import Image
import nibabel as nib
import time
import scipy

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['pdf.fonttype'] = 42


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file
        self.path_loss_val = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "fmri",
            "2020_08_28_00_25_fmri_unet_denoiser_mean_absolute_error_2020_08_28_00_25_val_loss.npy",
        )
        self.raw_fmri_volume_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "fMRI",
            "sub-02-ses-imageryTest02-func-sub-02_ses-imageryTest02_task-imagery_run-01_bold.nii",
        )
        self.den_fmri_volume_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "fMRI",
            "sub-02-ses-imageryTest02-func-sub-02_ses-imageryTest02_task-imagery_run-01_bold_out_unet_loss_0_0286_full.nii",
        )

        self.path_to_roi_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "fMRI",
            "list_roi_sub2",
        )
        local_path = pathlib.Path(__file__).parent.absolute()

        self.path_schematic = os.path.join(local_path, "Figure_5 - top_panel.png")

    def load_data(self):
        self.get_loss_curves()
        self.get_fmri_volumes()
        # self.get_roi_data()

    def get_roi_data(self):
        list_roi = os.listdir(self.path_to_roi_folder)

        self.dict_roi = {}
        for each_roi in list_roi:
            if ".nii" in each_roi:
                local_roi_file = os.path.join(self.path_to_roi_folder, each_roi)
                nib.load(local_roi_file)
                roi_volume = (
                    nib.load(local_roi_file).get_fdata()[:, :, :].astype("float")
                )

                self.dict_roi[each_roi] = roi_volume

        self.nb_roi = len(self.dict_roi.keys())

    def get_fmri_volumes(self):

        self.raw_brain_data = (
            nib.load(self.raw_fmri_volume_path)
            .get_fdata()[:, :, :, 3:-3]
            .astype("float")
        )

        self.den_brain_data = (
            nib.load(self.den_fmri_volume_path)
            .get_fdata()[:, :, :, 3:-3]
            .astype("float")
        )

    def get_loss_curves(self):
        self.val_loss = np.load(self.path_loss_val)

    def plot_schematic_fmri(self, ax):
        local_img = plt.imread(self.path_schematic)
        plt.imshow(local_img)
        plt.axis("off")

    def plot_val_loss(self):
        batch_size = 1000
        step_per_epoch = 6000
        exponent = 1000000
        x_axis = batch_size * step_per_epoch * np.arange(0, self.val_loss.shape[0])
        plt.plot(x_axis / exponent, self.val_loss)
        plt.xlabel("Number of unique samples (millions)")
        plt.ylabel("Validation loss")
        learning_rate_adj_list = np.arange(50, 250, 50)

        for indiv_learning_adj in learning_rate_adj_list:
            plt.vlines(
                batch_size * step_per_epoch * indiv_learning_adj / exponent,
                0,
                0.2,
                color="k",
                linewidth=0.7,
                linestyle=(0, (5, 5)),
            )

        plt.ylim([0, 0.2])
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

    def get_raw_den_error(self, movie_raw, movie_den):

        aver_img_raw = np.mean(movie_raw, axis=2)
        aver_img_den = np.mean(movie_den, axis=2)
        movie_raw_subs = movie_raw
        movie_den_subs = movie_den

        for index in np.arange(movie_raw.shape[2]):
            movie_raw_subs[:, :, index] = movie_raw[:, :, index] - aver_img_raw
            movie_den_subs[:, :, index] = movie_den[:, :, index] - aver_img_den

        error_raw_den = movie_raw_subs - movie_den_subs

        return [movie_raw_subs, movie_den_subs, error_raw_den]

    def apply_gaussian_filter_raw(self, gamma):
        from scipy.ndimage import gaussian_filter

    def make_figure(self):

        self.fig = plt.figure(figsize=(15, 20))

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.05, 0.65], yspan=[0.002, 0.35]
        )

        plt.text(-0.05, 0.9, "A", fontsize=20, weight="bold", transform=ax.transAxes)

        self.plot_schematic_fmri(ax)

        ax = placeAxesOnGrid(self.fig, dim=[1, 1], xspan=[0.7, 0.95], yspan=[0.1, 0.26])

        plt.sca(ax)
        # self.plot_val_loss()
        self.plot_snr_before_after()

        plt.text(-0.1, 0.95, "B", fontsize=20, weight="bold", transform=ax.transAxes)

        ax = placeAxesOnGrid(
            self.fig, dim=[2, 4], xspan=[0.005, 0.995], yspan=[0.33, 0.6]
        )
        plt.text(
            0.01, 1.2, "C", fontsize=20, weight="bold", transform=ax[0][0].transAxes
        )

        [movie_raw_subs, movie_den_subs, error_raw_den] = self.get_raw_den_error(
            self.raw_brain_data[:, 20, :, :].copy(),
            self.den_brain_data[:, 20, :, :].copy(),
        )

        plt.sca(ax[0][0])
        plt.imshow(self.raw_brain_data[:, 20, ::-1, 20].T, cmap="gray", clim=[0, 1831])
        plt.axis("off")
        plt.title("Raw")

        plt.sca(ax[0][1])
        plt.imshow(movie_raw_subs[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")
        plt.title("Mean subtracted\nRaw")

        plt.sca(ax[0][2])
        plt.imshow(movie_den_subs[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")
        plt.title("Mean subtracted\nDeepInterpolation")

        plt.sca(ax[0][3])
        plt.imshow(error_raw_den[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")
        plt.title("Residual")

        [movie_raw_subs, movie_den_subs, error_raw_den] = self.get_raw_den_error(
            self.raw_brain_data[35, :, :, :].copy(),
            self.den_brain_data[35, :, :, :].copy(),
        )

        plt.sca(ax[1][0])
        plt.imshow(self.raw_brain_data[35, :, ::-1, 20].T, cmap="gray", clim=[0, 1831])
        plt.axis("off")
        local_shape_raw = self.raw_brain_data.shape
        rectangle_length = 50 * 1.0 / 3
        rect = matplotlib.patches.Rectangle(
            [5, local_shape_raw[0] - 20], rectangle_length, 2, angle=0.0, color="w",
        )
        plt.gca().add_patch(rect)

        plt.sca(ax[1][1])
        plt.imshow(movie_raw_subs[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")

        plt.sca(ax[1][2])
        plt.imshow(movie_den_subs[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")

        plt.sca(ax[1][3])
        plt.imshow(error_raw_den[:, ::-1, 20].T, cmap="gray", clim=[-50, 50])
        plt.axis("off")

        """
        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.05, 0.4], yspan=[0.65, 0.80]
        )
        plt.text(-0.085, 1, "D", fontsize=20, weight="bold", transform=ax.transAxes)

        self.plot_snr_before_after()

        ax = placeAxesOnGrid(
            self.fig, dim=[3, 1], xspan=[0.47, 0.975], yspan=[0.65, 0.80]
        )
        plt.text(-0.1, 1, "E", fontsize=20, weight="bold", transform=ax[0].transAxes)

        self.plot_example_voxels(ax)
        """

        """
    ax = placeAxesOnGrid(self.fig, dim=[1, 1], xspan=[
                         0.1, 0.9], yspan=[0.65, 0.995])
    plt.text(-0.085, 1, "D", fontsize=20,
             weight="bold", transform=ax.transAxes)

    self.plot_roi_raw_den()
    """

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def get_rois_averages(self, brain_volume):

        average_volume = np.mean(brain_volume, axis=3)
        volume_shape = brain_volume.shape
        nb_frames = volume_shape[3]

        for local_frame in np.arange(0, volume_shape[3]):
            brain_volume[:, :, :, local_frame] = (
                brain_volume[:, :, :, local_frame] / average_volume
            )

        rois_average = {}
        average_activity = np.zeros(nb_frames)
        for local_roi_key in self.dict_roi:
            print(local_roi_key)
            rois_average[local_roi_key] = np.zeros(nb_frames)
            for local_frame in np.arange(0, volume_shape[3]):
                local_volume = brain_volume[:, :, :, local_frame]
                local_roi = self.dict_roi[local_roi_key]
                rois_average[local_roi_key][local_frame] = np.mean(
                    local_volume[local_roi == 1]
                )

            average_activity = (
                average_activity + rois_average[local_roi_key] / self.nb_roi
            )

        return rois_average, average_activity

    def plot_roi_raw_den(self):
        [rois_average, average_activity] = self.get_rois_averages(self.raw_brain_data)
        index_x = np.arange(average_activity.shape[0]) * 1 / 3

        for index, local_roi_key in enumerate(self.dict_roi):
            plt.plot(
                index_x,
                index * 0.015 + rois_average[local_roi_key] - average_activity,
                label=local_roi_key,
                color="k",
            )

        [rois_average, average_activity] = self.get_rois_averages(self.den_brain_data)

        for index, local_roi_key in enumerate(self.dict_roi):
            plt.plot(
                index_x,
                index * 0.015 + rois_average[local_roi_key] - average_activity,
                label=local_roi_key,
                color="r",
            )
            plt.gca().text(
                np.max(index_x) + 4, index * 0.015, local_roi_key[-7:-4], ha="right"
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Average background subtracted\nROI intensity (a.u.)")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)

    def get_variance_in_and_out_head(self):
        nb_voxels = 10000
        list_std_back_raw = []
        list_std_head_raw = []

        list_std_back_den = []
        list_std_head_den = []

        list_std_back_res = []
        list_std_head_res = []

        kept = 0

        box_background_corner1 = np.array([3, 0, 0])
        box_background_size = np.array([6, 189, 147]) / 3
        box_head_corner1 = np.array([48, 60, 57]) / 3
        box_head_size = np.array([135, 114, 60]) / 3 - box_head_corner1
        edge_time = 5

        for index in range(nb_voxels):

            x_back = np.random.randint(
                box_background_corner1[0],
                box_background_corner1[0] + box_background_size[0],
            )
            y_back = np.random.randint(
                box_background_corner1[1],
                box_background_corner1[1] + box_background_size[1],
            )
            z_back = np.random.randint(
                box_background_corner1[2],
                box_background_corner1[2] + box_background_size[2],
            )

            x_head = np.random.randint(
                box_head_corner1[0], box_head_corner1[0] + box_head_size[0]
            )
            y_head = np.random.randint(
                box_head_corner1[1], box_head_corner1[1] + box_head_size[1]
            )
            z_head = np.random.randint(
                box_head_corner1[2], box_head_corner1[2] + box_head_size[2]
            )

            local_trace_back = local_figure.raw_brain_data[
                x_back, y_back, z_back, edge_time:-edge_time
            ]
            local_trace_head = local_figure.raw_brain_data[
                x_head, y_head, z_head, edge_time:-edge_time
            ]

            local_trace_back_den = local_figure.den_brain_data[
                x_back, y_back, z_back, edge_time:-edge_time
            ]
            local_trace_head_den = local_figure.den_brain_data[
                x_head, y_head, z_head, edge_time:-edge_time
            ]

            list_std_back_raw.append(np.std(local_trace_back))
            list_std_head_raw.append(np.std(local_trace_head))

            list_std_back_den.append(np.std(local_trace_back_den))
            list_std_head_den.append(np.std(local_trace_head_den))

            list_std_back_res.append(np.std(local_trace_back - local_trace_back_den))
            list_std_head_res.append(np.std(local_trace_head - local_trace_head_den))

        print(
            f"background std raw = {np.mean(list_std_back_raw)} +/- {scipy.stats.sem(list_std_back_raw)}, N = {len(list_std_back_raw)}"
        )
        print(
            f"head std raw = {np.mean(list_std_head_raw)} +/- {scipy.stats.sem(list_std_head_raw)}, N = {len(list_std_head_raw)}"
        )

        print(
            f"background std den = {np.mean(list_std_back_den)} +/- {scipy.stats.sem(list_std_back_den)}, N = {len(list_std_back_den)}"
        )
        print(
            f"head std den = {np.mean(list_std_head_den)} +/- {scipy.stats.sem(list_std_head_den)}, N = {len(list_std_head_den)}"
        )

        print(
            f"background std res = {np.mean(list_std_back_res)} +/- {scipy.stats.sem(list_std_back_res)}, N = {len(list_std_back_res)}"
        )
        print(
            f"head std res = {np.mean(list_std_head_res)} +/- {scipy.stats.sem(list_std_head_res)}, N = {len(list_std_head_res)}"
        )

    def get_test_snr_before_after(self):
        nb_voxels = 2000000
        list_snr_raw = []
        list_snr_den = []
        kept = 0
        for index in range(nb_voxels):
            edge = 3
            edge_off = 0

            x_in = np.random.randint(50)
            y_in = np.random.randint(50)
            z_in = np.random.randint(50)

            local_trace_raw = local_figure.raw_brain_data[
                x_in, y_in, z_in, edge + edge_off : -edge + edge_off
            ]
            local_trace_den = local_figure.den_brain_data[
                x_in, y_in, z_in, edge + edge_off : -edge + edge_off
            ]

            local_mean_raw = np.mean(local_trace_raw)
            local_mean_den = np.mean(local_trace_den)

            local_std_raw = np.std(local_trace_raw)
            local_std_den = np.std(local_trace_den)
            try:
                tsnr_raw = local_mean_raw / local_std_raw
                tsnr_den = local_mean_den / local_std_den
                if local_mean_raw > 500:
                    list_snr_raw.append(tsnr_raw)
                    list_snr_den.append(tsnr_den)
                    kept = kept + 1
                    if kept > 100000:
                        break
            except:
                print("Invalid voxel value")

        bins_on = np.arange(-150, 150)
        [snr_diff, final_bins] = np.histogram(
            np.array(list_snr_den) - np.array(list_snr_raw), bins=bins_on
        )

        import scipy

        out = scipy.stats.ttest_rel(list_snr_raw, list_snr_den)
        print(out)
        statistic = out[0]
        pvalue = out[1]
        print(statistic)
        print(pvalue)
        print(f"p value for paired t-test is {pvalue:.20f}.")

        print(
            f"raw tsnr = {np.mean(list_snr_raw)} +/- {np.std(list_snr_raw)}, N = {len(list_snr_raw)}"
        )
        print(
            f"den tsnr = {np.mean(list_snr_den)} +/- {np.std(list_snr_den)}, N = {len(list_snr_den)}"
        )

        plt.figure()
        plt.plot(bins_on[:-1], snr_diff)
        plt.show()

    def plot_snr_before_after(self):
        nb_voxels = 2000000
        list_snr_raw = []
        list_snr_den = []
        kept = 0
        for index in range(nb_voxels):
            edge = 3
            edge_off = 0

            x_in = np.random.randint(50)
            y_in = np.random.randint(50)
            z_in = np.random.randint(50)

            local_trace_raw = local_figure.raw_brain_data[
                x_in, y_in, z_in, edge + edge_off : -edge + edge_off
            ]
            local_trace_den = local_figure.den_brain_data[
                x_in, y_in, z_in, edge + edge_off : -edge + edge_off
            ]

            local_mean_raw = np.mean(local_trace_raw)
            local_mean_den = np.mean(local_trace_den)

            local_std_raw = np.std(local_trace_raw)
            local_std_den = np.std(local_trace_den)
            try:
                tsnr_raw = local_mean_raw / local_std_raw
                tsnr_den = local_mean_den / local_std_den
                if local_mean_raw > 500:
                    list_snr_raw.append(tsnr_raw)
                    list_snr_den.append(tsnr_den)
                    kept = kept + 1
                    if kept > 100000:
                        break
            except:
                print("Invalid voxel value")
        bins_on = np.arange(0, 250)
        [snr_den_bins, final_bins] = np.histogram(np.array(list_snr_den), bins=bins_on)
        [snr_raw_bins, final_bins] = np.histogram(np.array(list_snr_raw), bins=bins_on)

        plt.plot(snr_raw_bins, "#4A484A", label="Raw")
        plt.fill_between(
            bins_on[:-1], snr_raw_bins, color=(209 / 255.0, 209 / 255.0, 209 / 255.0)
        )
        plt.plot(snr_den_bins, "#8A181A", label="DeepInterpolation")
        plt.fill_between(
            bins_on[:-1], snr_den_bins, color=(231 / 255.0, 209 / 255.0, 210 / 255.0)
        )

        plt.xlabel("tSNR")
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.legend(frameon=False)

    def get_voxel_trace(self, xyx_coord):
        x_in = xyx_coord[0]
        y_in = xyx_coord[1]
        z_in = xyx_coord[2]

        local_trace_raw = local_figure.raw_brain_data[x_in, y_in, z_in, :]
        local_trace_den = local_figure.den_brain_data[x_in, y_in, z_in, :]
        return [local_trace_raw, local_trace_den]

    def plot_example_voxels(self, ax):

        all_voxel = []
        local_voxel = {}
        local_voxel["x"] = 33
        local_voxel["y"] = 54
        local_voxel["z"] = 20
        local_voxel["label"] = "Voxel in V1"
        all_voxel.append(local_voxel.copy())
        local_voxel["x"] = 37
        local_voxel["y"] = 49
        local_voxel["z"] = 20
        local_voxel["label"] = "Voxel in V2"
        all_voxel.append(local_voxel.copy())
        local_voxel["x"] = 43
        local_voxel["y"] = 32
        local_voxel["z"] = 29
        local_voxel["label"] = "Voxel in white matter"
        all_voxel.append(local_voxel.copy())
        """
        local_voxel['x'] = 56
        local_voxel['y'] = 8
        local_voxel['z'] = 23
        local_voxel['label'] = 'Background'
        all_voxel.append(local_voxel.copy())
        """

        for index, indiv_voxels in enumerate(all_voxel):
            plt.sca(ax[index])
            [local_trace_raw, local_trace_den] = self.get_voxel_trace(
                [indiv_voxels["x"], indiv_voxels["y"], indiv_voxels["z"]]
            )

            index_x = np.arange(local_trace_raw.shape[0]) * 1 / 3
            local_ave = np.mean(local_trace_raw)
            plt.plot(
                index_x, 100 * (local_trace_raw - local_ave) / local_ave, "#4A484A"
            )
            plt.plot(
                index_x, 100 * (local_trace_den - local_ave) / local_ave, "#8A181A"
            )
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["top"].set_visible(False)
            plt.title(indiv_voxels["label"])
            # plt.gca().text(np.max(index_x)+4, 5,
            #               indiv_voxels['label'], ha='right')

            if index < 2:
                plt.gca().spines["bottom"].set_visible(False)
                plt.gca().axes.xaxis.set_ticks([])
                plt.gca().axes.xaxis.set_ticklabels([])
            else:
                plt.xlabel("Time (s)")

            if index == 1:
                plt.ylabel("Percentage change")


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Figure 5 - application to fMRI.pdf",
    )

    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.get_variance_in_and_out_head()
    local_figure.make_figure()
    local_figure.save_figure()
    local_figure.get_test_snr_before_after()

