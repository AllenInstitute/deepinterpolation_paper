# Amazing code developed by Nathan Gouwens
import numpy as np
import matplotlib.pylab as plt
import os
import matplotlib


class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

    def load_data(self):
        # We first gather needed data
        self.meta_data_1 = "dummy"

        self.get_external_data()

    def make_figure(self):
        self.fig = plt.figure(figsize=(10, 10))

        global_grid = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3])

        spec1 = global_grid[0].subgridspec(nrows=1, ncols=2)

        ax = self.fig.add_subplot(spec1[0])
        # panel letter
        plt.text(-0.4, 1, "A", fontsize=20, weight="bold", transform=ax.transAxes)

        self.plot_dummy_example(ax)
        ax = self.fig.add_subplot(spec1[1])
        self.plot_dummy_example(ax)

        spec2 = global_grid[1].subgridspec(nrows=1, ncols=3)

        ax = self.fig.add_subplot(spec2[0])
        # panel letter
        plt.text(-0.15, 1.1, "B", fontsize=20, weight="bold", transform=ax.transAxes)
        self.plot_dummy_example(ax)

        ax = self.fig.add_subplot(spec2[1])
        self.plot_dummy_example(ax)

        ax = self.fig.add_subplot(spec2[2])
        self.plot_dummy_example(ax)

        plt.tight_layout()

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def get_external_data(self):
        self.dummy = np.random.rand(100)

    def plot_dummy_example(self, ax):
        plt.plot(self.dummy)
        plt.xlabel("dummy")
        plt.ylabel("dummy")


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pdfs", "Dummy_figure.pdf"
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
