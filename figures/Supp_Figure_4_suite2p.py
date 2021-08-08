import matplotlib.pylab as plt
import numpy as np
import os
from scripts.plotting_helpers import placeAxesOnGrid
import pathlib
import h5py
import cv2
import matplotlib


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file
        self.path_suite2p_raw = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "suite2p",
            "505811062",
            "raw"
        )
        self.path_suite2p_deep = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "suite2p",
            "505811062",
            "deep"
        )

        self.path_comp_movie = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_505811062.h5"
        )

        self.color_raw = "#4A484A"
        self.color_den = "#8A181A"

    def load_data(self):
        self.get_suite2p_segmentation()
        self.get_rois_position()
        self.get_movies()

    def get_movies(self):
        self.movie_pointer = h5py.File(self.path_comp_movie, 'r')
        self.raw_movie = self.movie_pointer['data_raw']
        self.deep_movie = self.movie_pointer['data_proc']

    def get_path_segmentation(self, root_path):
        F_path = os.path.join(root_path, "F.npy")
        is_cell_path = os.path.join(root_path, "iscell.npy")
        png_path = os.path.join(root_path, "projection.png")

        return (F_path, is_cell_path, png_path)

    def get_segmented(self, root_path):
        raw_traces = "F.npy"
        sorting = "iscell.npy"
        cell_stat = "stat.npy"

        traces_path = os.path.join(root_path, raw_traces)
        sorting_path = os.path.join(root_path, sorting)
        cell_path = os.path.join(root_path, cell_stat)
        traces_f = np.load(traces_path)
        is_cell = np.load(sorting_path)[:, 0] == 1
        index_filters = np.load(cell_path, allow_pickle=True)
        cell_filters = self.get_cell_image_filters(index_filters)
        return traces_f, cell_filters

    def get_cell_image_filters(self, index_filters):
        # We recreate all cells filters
        all_cells = np.zeros((512, 512, len(index_filters)))

        for neuron_nb in range(len(index_filters)):
            list_x = index_filters[neuron_nb]["xpix"]
            list_y = index_filters[neuron_nb]["ypix"]
            weight = index_filters[neuron_nb]["lam"]

            all_cells[list_y, list_x, neuron_nb] = weight

        return all_cells

    def get_suite2p_segmentation(self):

        self.raw_traces, self.raw_filters = self.get_segmented(
            self.path_suite2p_raw)
        self.deep_traces, self.deep_filters = self.get_segmented(
            self.path_suite2p_deep)

        self.deep_traces_dff = self.deep_traces
        self.raw_traces_dff = self.raw_traces

        for index_cell in range(self.raw_traces.shape[0]):
            self.deep_traces_dff[index_cell, :] = self.deep_traces[index_cell,
                                                                   :]/np.mean(self.deep_traces[index_cell, :])-1
            self.raw_traces_dff[index_cell, :] = self.raw_traces_dff[index_cell,
                                                                     :]/np.mean(self.raw_traces_dff[index_cell, :])-1

    def get_rois_position(self):

        self.raw_ind_x = []
        self.raw_ind_y = []
        self.deep_ind_x = []
        self.deep_ind_y = []

        for local_roi_index in range(self.raw_filters.shape[2]):
            pos_raw = np.unravel_index(np.argmax(
                self.raw_filters[:, :, local_roi_index], axis=None), self.raw_filters.shape)
            pos_raw = np.argwhere(self.raw_filters[:, :, local_roi_index] == np.max(
                self.raw_filters[:, :, local_roi_index]))[0]
            self.raw_ind_x.append(pos_raw[0])
            self.raw_ind_y.append(pos_raw[1])

        for local_roi_index in range(self.deep_filters.shape[2]):
            pos_deep = np.unravel_index(np.argmax(
                self.deep_filters[:, :, local_roi_index], axis=None), self.deep_filters.shape)
            pos_deep = np.argwhere(self.deep_filters[:, :, local_roi_index] == np.max(
                self.deep_filters[:, :, local_roi_index]))[0]

            self.deep_ind_x.append(pos_deep[0])
            self.deep_ind_y.append(pos_deep[1])

    def convert_gray_rgb(self, local_img, bottom_perc=2, top_perc=98):
        bot_perc_val = np.percentile(local_img.flatten(), bottom_perc)
        top_perc_val = np.percentile(local_img.flatten(), top_perc)

        local_img[local_img > top_perc_val] = top_perc_val
        local_img[local_img < bot_perc_val] = bot_perc_val

        local_img = (
            1
            * (local_img - np.min(local_img))
            / (np.max(local_img) - np.min(local_img))
        )
        local_img = np.expand_dims(local_img, axis=2)
        local_img = np.repeat(local_img, 3, axis=2)

        return local_img

    def colorized(self, base_image, mask_img, color):
        hsv_img = cv2.cvtColor(base_image.astype("float32"), cv2.COLOR_RGB2HSV)

        hsv_img_color = hsv_img[:, :, 0]
        hsv_img_color[mask_img == 1] = color
        hsv_img[:, :, 1] = hsv_img_color

        final_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        return final_img

    def match_deep_to_raw(self, index_deep):
        distance = np.sqrt((self.raw_ind_x-self.deep_ind_x[index_deep])**2+(
            self.raw_ind_y-self.deep_ind_y[index_deep])**2)
        ind_raw = np.argmin(distance)

        return ind_raw

    def zoom_filter(self, imarray, square_size=50):
        orig_shape = imarray.shape
        # We locate the ROI center
        value = np.max(imarray.flatten())
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

        final_img = imarray[int(xmin): int(xmax), int(ymin): int(ymax)]

        return final_img

    def plot_example_movie_roi(self, current_mov, roi_filters, scale_bar=True):

        frame_index = 200

        # We first create the ROI + image of the figure
        local_img = current_mov[frame_index, :, :]
        #local_img = np.swapaxes(local_img, 0, 1)

        local_img = self.convert_gray_rgb(local_img)
        index_trace = 0

        for index_filter in range(roi_filters.shape[2]):
            index_trace = index_trace + 1

            local_roi = roi_filters[:, :, index_filter] > 0
            color = np.random.randint(0, 255, size=1)

            local_img = self.colorized(local_img, local_roi, color)

        plt.imshow(local_img)
        plt.gca().axis("off")

        if scale_bar:
            rectangle_length = 100 * current_mov.shape[1] / 400
            rect = matplotlib.patches.Rectangle(
                [20, 470], rectangle_length, 10, angle=0.0, color="w"
            )
            plt.gca().add_patch(rect)

    def make_figure(self):

        self.fig = plt.figure(figsize=(10, 10))

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 2], xspan=[0.15, 0.85], yspan=[0.05, 0.35]
        )

        plt.text(-0.2, 1, "A", fontsize=20,
                 weight="bold", transform=ax[0].transAxes)
        plt.axis('off')

        plt.sca(ax[0])
        self.plot_example_movie_roi(self.raw_movie, self.raw_filters)
        plt.axis('off')
        plt.title('Suite2p with Raw', fontdict={
                  'fontsize': 12}, color=self.color_raw)

        plt.sca(ax[1])
        self.plot_example_movie_roi(self.deep_movie, self.deep_filters)
        plt.axis('off')
        plt.title('Suite2p with DeepInterpolation', fontdict={
                  'fontsize': 12}, color=self.color_den)

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.05, 0.1], yspan=[0.40, 0.50]
        )

        plt.text(0.1, 1, "B", fontsize=20,
                 weight="bold", transform=ax.transAxes)
        plt.axis('off')

        soma_list = [43, 85, 94, 58, 34, 144, 221, 302, 320]

        total_rows = 3
        total_columns = 3
        global_ax = placeAxesOnGrid(
            self.fig, dim=[total_rows, 2*total_columns], xspan=[0.15, 0.85], yspan=[0.40, 0.70]
        )
        path_cell = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..",
                                 "data",
                                 "local_large_data", 'out')

        # Use to output all filters
        """
        for deep_index in range(self.deep_filters.shape[0]):
                raw_index = self.match_deep_to_raw(deep_index)
                plt.subplot(1,2,1)
                plt.imshow(self.zoom_filter(self.raw_filters[:,:,raw_index]), cmap='gray')
                                
                plt.subplot(1,2,2)
                plt.imshow(self.zoom_filter(self.deep_filters[:,:,deep_index]), cmap='gray')
                plt.savefig(os.path.join(path_cell, 'Matching_'+str(deep_index)+'.png'))
        """

        for row in range(total_rows):
            for column in range(total_columns):
                deep_index = soma_list[row*total_rows + column]
                raw_index = self.match_deep_to_raw(deep_index)

                plt.sca(global_ax[row][2*column])
                cmin = np.percentile(self.zoom_filter(
                    self.deep_filters[:, :, deep_index]), 5)
                cmax = np.percentile(self.zoom_filter(
                    self.deep_filters[:, :, deep_index]), 100)

                plt.imshow(self.zoom_filter(self.raw_filters[:, :, raw_index]), clim=[
                           cmin, cmax], cmap='gray')
                if row == 0:
                    plt.title('Raw', fontdict={
                              'fontsize': 8}, color=self.color_raw)

                if row == 2 and column == 0:
                    rectangle_length = 10 * self.raw_movie.shape[1] / 400
                    rect = matplotlib.patches.Rectangle(
                        [5, 40], rectangle_length, 3, angle=0.0, color="w"
                    )
                    plt.gca().add_patch(rect)

                # plt.axis('off')
                y_axis = plt.gca().get_yaxis()
                y_axis.set_visible(True)
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])
                plt.gca().spines['left'].set_visible(False)
                plt.gca().spines['bottom'].set_visible(False)
                plt.gca()
                plt.ylabel('ROI '+str(row*total_rows + column),
                           fontdict={'fontsize': 8})
                plt.sca(global_ax[row][2*column+1])
                plt.imshow(self.zoom_filter(self.deep_filters[:, :, deep_index]), clim=[
                           cmin, cmax], cmap='gray')
                if row == 0:
                    plt.title('DeepInterpolation', fontdict={
                              'fontsize': 8}, color=self.color_den)
                plt.axis('off')

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.05, 0.1], yspan=[0.75, 0.80]
        )
        plt.text(0.1, 1, "C", fontsize=20,
                 weight="bold", transform=ax.transAxes)

        plt.axis('off')

        soma_list = [43, 85]
        top_index = [0, 1]
        ax = placeAxesOnGrid(
            self.fig, dim=[1, 2], xspan=[0.15, 0.85], yspan=[0.75, 1]
        )

        for index, index_soma_deep in enumerate(soma_list):
            plt.sca(ax[index])
            index_soma_raw = self.match_deep_to_raw(index_soma_deep)

            local_deep_trace = self.deep_traces_dff[index_soma_deep][0:1000]
            local_raw_trace = self.raw_traces_dff[index_soma_raw][30:1030]
            plt.title('ROI '+str(top_index[index]))
            plt.plot(1/30*np.arange(len(local_raw_trace)), 100 *
                     local_raw_trace, label='raw', linewidth=1, color=self.color_raw)
            plt.plot(1/30*np.arange(len(local_deep_trace)), 100 *
                     local_deep_trace, label='deep', linewidth=1, color=self.color_den)

            plt.xlabel('Time (s)')
            plt.ylabel(u'ΔF/F (%)')
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 4 - Suite2p.pdf",
    )

    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
