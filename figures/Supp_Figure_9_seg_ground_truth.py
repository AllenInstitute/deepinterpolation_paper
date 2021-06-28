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

        # , 'first_volume', 'typicalVolume80mW']
        self.list_naomi_volumes = [
            'first_volume', 'sparseLabelingVolume', 'typicalVolume80mW', "typicalVolume160mW"]
        self.list_methods = ['noisy', 'deepi', 'gaussian', 'pmd']

        self.roi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..",
                                     "data",
                                     "local_large_data",
                                     "ground_truth_2"
                                     )

        self.path_roc_data = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth_2",
            "ROC_data2.npy"
        )

    def load_data(self):
        self.filters = {}
        for local_volume in self.list_naomi_volumes:
            for local_method in self.list_methods:

                path_to_filters = os.path.join(
                    self.roi_path, local_volume, 'ROC', local_method, 'sweep')

                local_threshold_filters = self.load_threshold_data(
                    path_to_filters)

                if not(local_volume in self.filters.keys()):
                    self.filters[local_volume] = {}

                self.filters[local_volume][local_method] = local_threshold_filters

            path_to_truth = os.path.join(
                self.roi_path, local_volume, local_volume+"_ideal.mat")

            self.filters[local_volume]["Truth"] = self.get_ground_truth_rois(
                path_to_truth)

        if os.path.isfile(self.path_roc_data):
            self.ROC = np.load(self.path_roc_data, allow_pickle=True).item()
        else:
            self.ROC = {}

        self.align_filters_to_truth()

    def load_threshold_data(self, path_to_filters):

        all_filters = {}
        for local_dir in os.listdir(path_to_filters):
            local_filters = self.get_segmented_filters(
                os.path.join(path_to_filters, local_dir), whitout_suitep=True)
            all_filters[local_dir.split('_scal')[1]] = local_filters

        return all_filters

    def get_segmented_filters(self, path_to_segm, whitout_suitep=False):
        cell_stat = "stat.npy"

        if whitout_suitep:
            cell_path = os.path.join(path_to_segm, 'plane0', cell_stat)
        else:
            cell_path = os.path.join(
                path_to_segm, 'suite2p', 'plane0', cell_stat)

        cell_filters = np.load(cell_path, allow_pickle=True)
        all_cells = self.get_cell_image_filters(cell_filters)
        return all_cells

    def align_filters_to_truth(self):
        # This is because of introduced motion artifacts in some of the simulation

        for index_volume, key_volume in enumerate(self.filters.keys()):
            local_filters = self.filters[key_volume]
            for index_technique, key_technique in enumerate(local_filters.keys()):
                local_truths = self.filters[key_volume]['Truth']
                if "Truth" not in key_technique:
                    first_threshold = list(
                        local_filters[key_technique].keys())[0]
                    bestx, besty = self.filters_to_filters_delta(
                        local_filters[key_technique][first_threshold], local_truths)
                    # print(bestx, besty)
                    for index_threshold, key_threshold in enumerate(local_filters[key_technique].keys()):

                        filters_to_align = self.filters[key_volume][key_technique][key_threshold]
                        for index, indiv_filter in enumerate(filters_to_align):
                            aligned_filter = self.transl_image(
                                indiv_filter, bestx, besty)
                            self.filters[key_volume][key_technique][key_threshold][index] = aligned_filter
            print("Alignement done for "+key_volume)

    def transl_image(self, image, translx, transly):
        # Store height and width of the image
        height, width = image.shape[:2]

        T = np.float32([[1, 0, translx], [0, 1, transly]])

        # We use warpAffine to transform
        # the image using the matrix, T
        img_translation = cv2.warpAffine(image, T, (width, height))

        return img_translation

    def find_large_filters(self, filters, nb_pixel_threshold=5, nb_pixel_max=300):
        kept_index = []
        for index in np.arange(0, len(filters)):
            local_filter = filters[index]
            if len(local_filter[local_filter > 0]) > nb_pixel_threshold and len(local_filter[local_filter > 0]) < nb_pixel_max:
                kept_index.append(index)

        return kept_index

    def filters_to_filters_delta(self, filters_to_align, filters_truth):
        shared_one = self.find_large_filters(filters_to_align, 5, 300)
        shared_two = self.find_large_filters(filters_truth, 5, 300)

        # shared_one, shared_two = self.find_shared_filters(filters_to_align, filters_truth, corr_threshold = 0.65)

        filters_to_align_3d = self.re_org_array_non_flat(filters_to_align)
        filters_truth_3d = self.re_org_array_non_flat(filters_truth)

        filters_to_align_3d = np.squeeze(
            np.max(filters_to_align_3d[shared_one, :, :], axis=0))
        filters_truth_3d = np.squeeze(
            np.max(filters_truth_3d[shared_two, :, :], axis=0))

        """
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(filters_to_align_3d)
        plt.subplot(2,1,2)
        plt.imshow(filters_truth_3d)
        plt.show()
        """

        startx = -10
        starty = -10
        maxcorr = 0
        for translx in np.arange(-10, 10):
            for transly in np.arange(-10, 10):
                img_translation = self.transl_image(
                    filters_to_align_3d, translx, transly)

                corr = np.corrcoef(img_translation.flatten(),
                                   filters_truth_3d.flatten())[0][1]

                # print(translx, transly, corr)
                if corr > maxcorr:
                    maxcorr = corr
                    bestx = translx
                    besty = transly
        # print(bestx, besty)
        return bestx, besty

    def get_cell_image_filters(self, cell_filters, nb_pixel_threshold=5, nb_pixel_max=300):
        # We recreate all cells filters
        all_cells = []
        for neuron_nb in range(len(cell_filters)):
            local_cell = np.zeros((256, 256))

            list_x = cell_filters[neuron_nb]["xpix"]
            list_y = cell_filters[neuron_nb]["ypix"]
            weight = cell_filters[neuron_nb]["lam"]
            local_cell[list_y, list_x] = weight
            if len(weight) > nb_pixel_threshold and len(weight) < nb_pixel_max:
                all_cells.append(local_cell)

        return all_cells

    def get_ground_truth_rois(self, path_to_truth, nb_pixel_threshold=5, nb_pixel_max=300):
        data_mat = h5py.File(path_to_truth, 'r')
        ground_images = data_mat['ideal']

        final_roi = []
        for index, roi in enumerate(np.arange(ground_images.shape[0])):
            local_img = ground_images[roi, 0:256, 0:256]
            if len(local_img[local_img > 0]) > nb_pixel_threshold and len(local_img[local_img > 0]) < nb_pixel_max:
                final_roi.append(np.transpose(
                    ground_images[roi, 0:256, 0:256]))

        return final_roi

    def re_org_array(self, filter):
        nb_filter = len(filter)
        img = filter[0]
        size_img = img.shape

        new_filter = np.zeros((nb_filter, size_img[0]*size_img[1]))

        for ind_filter in range(nb_filter):
            new_filter[ind_filter, :] = filter[ind_filter].flatten()

        return new_filter

    def re_org_array_non_flat(self, filter):
        nb_filter = len(filter)
        img = filter[0]
        size_img = img.shape

        new_filter = np.zeros((nb_filter, size_img[0], size_img[1]))

        for ind_filter in range(nb_filter):
            new_filter[ind_filter, :, :] = filter[ind_filter] / \
                np.max(filter[ind_filter].flatten())

        return new_filter

    def find_shared_filters(self, filter_set_one, filter_set_two, corr_threshold=0.7):
        flat_one = self.re_org_array(filter_set_one)
        flat_two = self.re_org_array(filter_set_two)

        unit_one = len(filter_set_one)
        unit_two = len(filter_set_two)

        corr_array = np.corrcoef(flat_one, flat_two)
        shared_corr = corr_array[0:unit_one, unit_one:-1]

        shared_one, shared_two = np.where(shared_corr > corr_threshold)
        shared_corr_selected = shared_corr[shared_one, shared_two]
        return shared_one, shared_two, shared_corr_selected

    def plot_all_filters(self, filters, keep_filters, scale_bar=False):

        hue = 255 * np.ones((256, 256))
        sat = 255 * np.ones((256, 256))
        val = 0 * np.ones((256, 256))
        # print(keep_filters)
        for index_mask in keep_filters:
            # print(index_mask)
            img = filters[index_mask][:, :]
            img = img/(100*np.max(img.flatten()))

            # All ROI should be the same color to avoid confusion with colors in following plots
            hue[img > 0] = 0
            sat[img > 0] = 0.75
            val[img > 0] = val[img > 0] + 255 * img[img > 0]

        final_rgb = np.zeros((256, 256, 3))
        for x in range(256):
            for y in range(256):
                final_rgb[x, y, :] = matplotlib.colors.hsv_to_rgb(
                    [hue[x, y], sat[x, y], val[x, y]]
                )

        plt.imshow(final_rgb)
        plt.axis("off")

        if scale_bar:
            rectangle_length = 100 * 256 / 200
            rect = matplotlib.patches.Rectangle(
                [15, 256 - 15],
                rectangle_length,
                7,
                angle=0.0,
                color="w",
            )
            plt.gca().add_patch(rect)

    def make_figure(self):

        self.fig = plt.figure(figsize=(6, 12))

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.0, 0.05), yspan=(0.0, 0.05))
        plt.axis('off')
        plt.text(-0.2, 1.05, "A", fontsize=10, weight="bold",
                 transform=local_axis.transAxes)

        plt.text(3.5, 1.2, "threshold_scaling = 1", fontsize=5,
                 weight="bold", transform=local_axis.transAxes)

        local_axis_left = placeAxesOnGrid(
            self.fig, dim=[4, 3], xspan=(0.05, 0.5), yspan=(0.01, 0.3), wspace=0)

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.5, 0.55), yspan=(0.0, 0.05))
        plt.axis('off')

        plt.text(2.5, 1.2, "threshold_scaling = 3", fontsize=5,
                 weight="bold", transform=local_axis.transAxes)

        local_axis_right = placeAxesOnGrid(
            self.fig, dim=[4, 3], xspan=(0.5, 0.95), yspan=(0.01, 0.3), wspace=0)
        panels = [local_axis_left, local_axis_right]

        for index_volume, key_volume in enumerate(self.filters.keys()):
            if key_volume not in self.ROC.keys():
                self.ROC[key_volume] = {}

            local_filters = self.filters[key_volume]
            for index_technique, key_technique in enumerate(local_filters.keys()):

                if "Truth" not in key_technique:
                    if key_technique not in self.ROC[key_volume].keys():
                        self.ROC[key_volume][key_technique] = {'true_positive': [
                        ], 'false_positive': [], 'threshold': [], 'index': {}, 'shared_corr': []}

                    for index_threshold, key_threshold in enumerate(local_filters[key_technique].keys()):
                        if key_threshold not in self.ROC[key_volume][key_technique]['index'].keys():
                            self.ROC[key_volume][key_technique]['index'][key_threshold] = {
                            }

                            shared_one, shared_two, shared_corr_selected = self.find_shared_filters(
                                local_filters["Truth"], local_filters[key_technique][key_threshold])

                            all_units = np.arange(
                                len(local_filters[key_technique][key_threshold]))
                            shared_two_false_positive = []
                            for indiv_unit in all_units:
                                if indiv_unit not in shared_two:
                                    shared_two_false_positive.append(
                                        indiv_unit)

                            false_positive_number = len(
                                shared_two_false_positive)
                            true_positive_number = len(shared_two)

                            false_positive_rate = 100*false_positive_number / \
                                len(local_filters[key_technique]
                                    [key_threshold])
                            true_positive_rate = 100*true_positive_number / \
                                len(local_filters["Truth"])

                            self.ROC[key_volume][key_technique]['true_positive'].append(
                                true_positive_rate)
                            self.ROC[key_volume][key_technique]['false_positive'].append(
                                false_positive_rate)
                            self.ROC[key_volume][key_technique]['threshold'].append(
                                key_threshold)
                            self.ROC[key_volume][key_technique]['index'][key_threshold]['shared_one'] = shared_one
                            self.ROC[key_volume][key_technique]['index'][key_threshold]['shared_two'] = shared_two
                            self.ROC[key_volume][key_technique]['index'][key_threshold]['shared_two_false_positive'] = shared_two_false_positive
                            self.ROC[key_volume][key_technique]['shared_corr'].append(
                                np.mean(shared_corr_selected))

                        if index_volume == 0:

                            shared_one = self.ROC[key_volume][key_technique]['index'][key_threshold]['shared_one']
                            shared_two = self.ROC[key_volume][key_technique]['index'][key_threshold]['shared_two']
                            shared_two_false_positive = self.ROC[key_volume][key_technique][
                                'index'][key_threshold]['shared_two_false_positive']
                            for index_panel, check_index_threshold in enumerate([2, 4]):
                                local_axis = panels[index_panel]
                                if (index_threshold == check_index_threshold):
                                    plt.sca(local_axis[index_technique][0])

                                    if index_technique == 3 and index_panel > 0:
                                        self.plot_all_filters(
                                            local_filters[key_technique][key_threshold], shared_two, scale_bar=True)
                                    else:
                                        self.plot_all_filters(
                                            local_filters[key_technique][key_threshold], shared_two)

                                    if index_panel == 0:
                                        plt.text(-0.75, 0.5, key_technique, weight="bold", fontsize=5,
                                                 transform=local_axis[index_technique][0].transAxes)

                                    if index_technique == 0:
                                        plt.text(0.1, 1.05, "True Positive\nin Suite2p\nsegmentation", fontsize=3,
                                                 weight="normal", transform=local_axis[index_technique][0].transAxes)

                                    all_units = np.arange(
                                        len(local_filters[key_technique][key_threshold]))

                                    plt.sca(local_axis[index_technique][1])
                                    self.plot_all_filters(
                                        local_filters[key_technique][key_threshold], shared_two_false_positive)

                                    if index_technique == 0:
                                        plt.text(0.1, 1.05, "False Positive\nin Suite2p\nsegmentation", fontsize=3,
                                                 weight="normal", transform=local_axis[index_technique][1].transAxes)

                                    plt.sca(local_axis[index_technique][2])
                                    self.plot_all_filters(
                                        local_filters["Truth"], shared_one)
                                    if index_technique == 0:
                                        plt.text(0.1, 1.05, "Matched True positive ROIs\nfrom\nground truth", fontsize=3,
                                                 weight="normal", transform=local_axis[index_technique][2].transAxes)

                                    plt.axis('off')

        np.save(self.path_roc_data, self.ROC)

        ROC_averages = {}
        nb_volumes = len(self.ROC.keys())
        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.05, 0.1), yspan=(0.35, 0.4))
        plt.axis('off')
        plt.text(-0.8, 1.1, "B", fontsize=10, weight="bold",
                 transform=local_axis.transAxes)
        color_dict = {'deepi': "#8A181A", 'noisy': 'black',
                      'gaussian': 'green', 'pmd': 'blue'}
        matching_technique_name = {'deepi': 'DeepInterpolation',
                                   'noisy': 'Noisy', 'gaussian': 'Gaussian kernel', 'pmd': 'PMD'}

        local_axis = placeAxesOnGrid(self.fig, xspan=(
            0.05, 0.3), yspan=(0.35, 0.45), wspace=0)

        # Get average
        for index_volume, key_volume in enumerate(self.ROC.keys()):
            for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
                local_false_positive = np.array(
                    self.ROC[key_volume][key_technique]['false_positive'])
                local_true_positive = np.array(
                    self.ROC[key_volume][key_technique]['true_positive'])
                local_corr = np.array(
                    self.ROC[key_volume][key_technique]['shared_corr'])

                if index_volume == 0:
                    ROC_averages[key_technique] = {}
                    ROC_averages[key_technique]['false_positive_av'] = local_false_positive/nb_volumes
                    ROC_averages[key_technique]['true_positive_av'] = local_true_positive/nb_volumes
                    ROC_averages[key_technique]['threshold'] = self.ROC[key_volume][key_technique]['threshold']
                    ROC_averages[key_technique]['shared_corr_av'] = local_corr/nb_volumes

                else:
                    ROC_averages[key_technique]['false_positive_av'] = np.array(
                        ROC_averages[key_technique]['false_positive_av']) + local_false_positive/nb_volumes
                    ROC_averages[key_technique]['true_positive_av'] = np.array(
                        ROC_averages[key_technique]['true_positive_av']) + local_true_positive/nb_volumes
                    ROC_averages[key_technique]['shared_corr_av'] = np.array(
                        ROC_averages[key_technique]['shared_corr_av']) + local_corr/nb_volumes

        # Get Std
        for index_volume, key_volume in enumerate(self.ROC.keys()):
            for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
                local_false_positive = self.ROC[key_volume][key_technique]['false_positive']
                local_true_positive = self.ROC[key_volume][key_technique]['true_positive']
                local_corr = self.ROC[key_volume][key_technique]['shared_corr']

                if index_volume == 0:
                    ROC_averages[key_technique]['false_positive_std'] = (
                        local_false_positive-ROC_averages[key_technique]['false_positive_av'])**2/nb_volumes
                    ROC_averages[key_technique]['true_positive_std'] = (
                        local_true_positive-ROC_averages[key_technique]['true_positive_av'])**2/nb_volumes
                    ROC_averages[key_technique]['shared_corr_std'] = (
                        local_corr-ROC_averages[key_technique]['shared_corr_av'])**2/nb_volumes
                else:
                    ROC_averages[key_technique]['false_positive_std'] += (
                        local_false_positive-ROC_averages[key_technique]['false_positive_av'])**2/nb_volumes
                    ROC_averages[key_technique]['true_positive_std'] += (
                        local_true_positive-ROC_averages[key_technique]['true_positive_av'])**2/nb_volumes
                    ROC_averages[key_technique]['shared_corr_std'] += (
                        local_corr-ROC_averages[key_technique]['shared_corr_av'])**2/nb_volumes

        for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
            ROC_averages[key_technique]['false_positive_std'] = np.sqrt(
                ROC_averages[key_technique]['false_positive_std'])
            ROC_averages[key_technique]['true_positive_std'] = np.sqrt(
                ROC_averages[key_technique]['true_positive_std'])
            ROC_averages[key_technique]['shared_corr_std'] = np.sqrt(
                ROC_averages[key_technique]['shared_corr_std'])

        for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
            local_true_positive = ROC_averages[key_technique]['true_positive_av']
            local_true_positive_std = ROC_averages[key_technique]['true_positive_std']
            local_thresholds = ROC_averages[key_technique]['threshold']
            local_corr_std = ROC_averages[key_technique]['shared_corr_std']

            plt.plot(local_thresholds, local_true_positive,
                     color=color_dict[key_technique], label=matching_technique_name[key_technique], linewidth=1)
            plt.fill_between(local_thresholds, local_true_positive-local_true_positive_std,
                             local_true_positive+local_true_positive_std, alpha=0.2, color=color_dict[key_technique])

        plt.legend(prop={"size": 4}, frameon=False, loc='upper right')

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel('threshold_scaling', fontsize=6)
        plt.ylabel('True positive\n(% of ground truth ROI)', fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.4, 0.45), yspan=(0.35, 0.4))
        plt.axis('off')
        plt.text(-0.5, 1.1, "C", fontsize=10, weight="bold",
                 transform=local_axis.transAxes)

        local_axis = placeAxesOnGrid(self.fig, xspan=(
            0.4, 0.6), yspan=(0.35, 0.45), wspace=0)
        """    
        for index_volume, key_volume in enumerate(self.ROC.keys()): 
            for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()): 
                plt.plot(self.ROC[key_volume][key_technique]['threshold'], self.ROC[key_volume][key_technique]['false_positive'], alpha=index_volume/(len(self.ROC.keys())), color=color_dict[key_technique], label=key_technique)
        """

        for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
            local_false_positive = ROC_averages[key_technique]['false_positive_av']
            local_false_positive_std = ROC_averages[key_technique]['false_positive_std']
            local_thresholds = ROC_averages[key_technique]['threshold']

            plt.plot(local_thresholds, local_false_positive,
                     color=color_dict[key_technique], label=key_technique, linewidth=1)
            plt.fill_between(local_thresholds, local_false_positive-local_false_positive_std,
                             local_false_positive+local_false_positive_std, alpha=0.2, color=color_dict[key_technique])

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel('threshold_scaling', fontsize=6)
        plt.ylabel('False positive\n(% of detected ROI)', fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.7, 0.75), yspan=(0.35, 0.4))
        plt.axis('off')
        plt.text(-0.7, 1.1, "D", fontsize=10, weight="bold",
                 transform=local_axis.transAxes)

        local_axis = placeAxesOnGrid(self.fig, xspan=(
            0.7, 0.95), yspan=(0.35, 0.45), wspace=0)

        for index_technique, key_technique in enumerate(self.ROC[key_volume].keys()):
            local_false_positive = ROC_averages[key_technique]['false_positive_av']
            local_true_positive = ROC_averages[key_technique]['true_positive_av']
            local_thresholds = ROC_averages[key_technique]['threshold']

            plt.plot(local_false_positive, local_true_positive,
                     color=color_dict[key_technique], label=key_technique, linewidth=1)

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.xlabel('False positive\n(% of total ROI)', fontsize=6)
        plt.ylabel('True positive\n(% of ground truth ROI)', fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 9 - ground truth segmentation.pdf",
    )

    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
