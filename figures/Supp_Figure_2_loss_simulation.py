import h5py
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import pathlib
import os
from scripts.plotting_helpers import placeAxesOnGrid
from sklearn.linear_model import LinearRegression


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.local_comp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_657391625.h5",
        )

    def load_data(self):

        with h5py.File(self.local_comp, "r") as file_handle:
            self.raw_dat = file_handle["data_raw"][:, :, :]

        self.get_loss_simulation()

    def get_loss_simulation(self):

        self.mean_img = np.mean(self.raw_dat, axis=0)

        # We z-score to mimick our preprocessing in the neuronal net.
        z_mean_img = self.mean_img.flatten()
        z_mean_img = z_mean_img - np.mean(z_mean_img)
        z_mean_img = z_mean_img / np.std(z_mean_img)

        # We generate a poisson noise version of it
        self.list_peak_photons = np.arange(3, 100, 1)
        self.loss_list = []
        self.img_with_poissons = []
        for PEAK in self.list_peak_photons:
            poissonNoise = (
                np.random.poisson(
                    self.mean_img / np.max(self.mean_img.flatten()) * PEAK
                )
                / PEAK
                * np.max(self.mean_img.flatten())
            )

            self.img_with_poissons.append(poissonNoise)
            # We z-score to mimick our preprocessing in the neuronal net.
            poissonNoise = poissonNoise.flatten()
            poissonNoise = poissonNoise - np.mean(poissonNoise)
            poissonNoise = poissonNoise / np.std(poissonNoise)
            local_loss = np.mean(np.abs(poissonNoise.flatten() - z_mean_img))
            self.loss_list.append(np.mean(np.abs(poissonNoise.flatten() - z_mean_img)))

    def plot_loss_photon_peak(self, overlay_theo=False):
        plt.plot(
            self.list_peak_photons,
            self.loss_list,
            label="Averaged loss\ngiven real pixel distribution",
        )
        plt.xlabel("peak photons detected/pixel")
        plt.ylabel("Normalized validation loss")

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        ax = plt.gca()
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(14)

        if overlay_theo:
            # This is to overlay a theoretical fit
            reg = LinearRegression().fit(
                np.array([1 / np.sqrt(self.list_peak_photons)]).T,
                np.array([self.loss_list]).T,
            )
            y_fit = reg.predict(np.array([1 / np.sqrt(self.list_peak_photons)]).T)

            plt.plot(
                self.list_peak_photons,
                y_fit.flatten(),
                "r",
                label="Theoretical loss\ngiven homogenous pixel distribution",
            )
            plt.legend(frameon=False)

    def plot_corresponding_loss_img(self, local_loss, clim=[0, 400]):

        # We plot images to illustrate noise levels associated with losses
        min_index = np.argmin(np.abs(np.array(self.loss_list) - local_loss))
        local_img = self.img_with_poissons[min_index]

        plt.imshow(local_img, cmap="gray", clim=clim)
        plt.axis("off")
        plt.title("normalized validation loss : " + str(local_loss))

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def make_figure(self):

        self.fig = plt.figure(figsize=(18, 16))
        ax = placeAxesOnGrid(
            self.fig, dim=[1, 1], xspan=[0.05, 0.95], yspan=[0.05, 0.45]
        )
        plt.sca(ax)

        plt.text(-0.05, 1.1, "A", fontsize=15, weight="bold", transform=ax.transAxes)

        self.plot_loss_photon_peak(overlay_theo=True)

        ax = placeAxesOnGrid(
            self.fig, dim=[1, 4], xspan=[0.05, 0.95], yspan=[0.40, 0.95], wspace=0
        )

        list_loss_plot = [0.8, 0.6, 0.5]

        plt.text(-0.2, 1.1, "B", fontsize=15, weight="bold", transform=ax[0].transAxes)

        clim = [0, 600]
        for index, indiv_loss in enumerate(list_loss_plot):
            plt.sca(ax[index])

            self.plot_corresponding_loss_img(indiv_loss, clim=clim)
            if index == 0:
                local_shape_raw = [512, 512]
                rectangle_length = 100 * local_shape_raw[0] / 400
                rect = matplotlib.patches.Rectangle(
                    [30, local_shape_raw[0] - 60],
                    rectangle_length,
                    30,
                    angle=0.0,
                    color="w",
                )
                ax[3].add_patch(rect)

        plt.sca(ax[3])
        plt.imshow(self.mean_img, cmap="gray", clim=clim)
        plt.axis("off")
        plt.title("Ground truth")


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 2 - loss_simulation.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
