import h5py
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import pathlib
import os
from scripts.plotting_helpers import placeAxesOnGrid
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Rectangle

class Qc2p():
    """QC class for 2p movies
    Objects contain methods to generate QC metrics from physio movies 
    data should be contained in an hdf5 as a TxXxY array where T is number of time frames. 
    Args:  
        filepath:  The full path to the two-photon physio movie 
        h5_path_to_data: internal path used to access the movie data into the hdf5 file
        dyn_range:  An array corresponding to the min and max pixel value
    Attributes: 
        data_pointer:  A reference to the HDF5 file of interest
        _cache:  A cache that stores previously subsampled and cropped videos
    """

    def __init__(self,  filepath, h5_path_to_data = 'data', dyn_range = [0, 65533]):
        h5_pointer = h5py.File(filepath,'r')
        self.data_pointer = h5_pointer[h5_path_to_data]
        self._dyn_range = dyn_range
        self._cache = {}

    def plot_poisson_curve(self, start_frame=1, end_frame=2001, crop=(150,150), perc_min=1, perc_max=99):
        """Obtain a plot showing Poisson characteristics of the signal.
        Args: 
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile
        Returns: 
            A figure.
        """
        end_frame = min(end_frame, self.data_pointer.shape[0])
        photon_gain = self.get_photon_gain_parameters(start_frame=start_frame, 
                                                      end_frame=end_frame, 
                                                      crop=crop,
                                                      perc_min=perc_min,
                                                      perc_max=perc_max)

        h, xedges, yedges = np.histogram2d(photon_gain['var'], photon_gain['mean'], bins=(200,200))
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
     
        plt.imshow(h, origin='lower', extent=extent, aspect='auto', cmap='Blues')
        plt.colorbar(label='Pixel density')
        plt.xlabel('Mean of pixel value, AU)')
        plt.ylabel('Variance of pixel value, AU)')
        
        plt.xlim(photon_gain['mean'].min(), photon_gain['mean'].max())
        plt.ylim(photon_gain['var'].min(), photon_gain['var'].max())
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        mean_range = np.linspace(0, photon_gain['mean'].max(), num=200)
        background_noise_mean = -photon_gain['offset'] / photon_gain['slope']
        plt.tight_layout()
        plt.plot(mean_range, photon_gain['slope'] * (mean_range - background_noise_mean), 'r')

    def get_dynamic_range(self):
        """Get the dynamic range set for the data
        Returns: 
            An array corresponding to the min and max pixel value"""

        return self._dyn_range

    def subsample_and_crop_video(self, subsample, crop, start_frame=0, end_frame=-1):
        """Subsample and crop a video, cache results. Also functions as a data_pointer load.
        Args: 
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
            start_frame:  The index of the first desired frame
            end_frame:  The index of the last desired frame
        Returns: 
            The resultant array.
        """
        if (subsample, crop[0], crop[1], start_frame, end_frame) in self._cache:
            return self._cache[(subsample, crop[0], crop[1], start_frame, end_frame)]

        _shape = self.data_pointer.shape
        px_y_start, px_x_start = crop
        px_y_end = _shape[1] - px_y_start
        px_x_end = _shape[2] - px_x_start
        
        if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
            cropped_video = self.data_pointer[start_frame::subsample, 
                                                px_y_start:px_y_end,
                                                px_x_start:px_x_end]
        else:
            cropped_video = self.data_pointer[start_frame:end_frame:subsample, 
                                                px_y_start:px_y_end,
                                                px_x_start:px_x_end]

        self._cache[(subsample, crop[0], crop[1], start_frame, end_frame)] = cropped_video

        return cropped_video

    def get_axis_mean(self, axis=2, subsample=1, crop=(0,0), start_frame=0, end_frame=-1):
        """Get the mean of the video across a given axis.
        Args: 
            axis:  The axis (0, 1, 2, or None) over which to average
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
            start_frame:  The index of the first desired frame
            end_frame:  The index of the last desired frame
        Returns: 
            The averaged array or value, depending on the chosen axis.
        """
        cropped_video = self.subsample_and_crop_video(subsample=subsample, 
                                                      crop=crop, 
                                                      start_frame=start_frame, 
                                                      end_frame=end_frame)
        return np.mean(cropped_video, axis=axis)

    def validate_crop(self, crop):
        """Helper function to ensure cropping is not too extreme.
        Args:  
            crop:  (px_y, px_x), the number of pixels to remove from the borders
        
        Returns:  
            A possibly adjusted crop tuple.
        """
        px_y, px_x = crop
        if 2*crop[0] > self.data_pointer.shape[1]:
            px_y = 0
        if 2*crop[1] > self.data_pointer.shape[2]:
            px_x = 0
        return (px_y, px_x)

    def get_photon_metrics(self, start_frame=1, end_frame=2001, crop=(30,30)):
        """Photon Metrics.
        
        Compute metrics related to the physio signal. Helper function for _calculate_qc_metrics
        
        Args:  
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
        Returns:  
            A dictionary of metrics
        """
        cropped_video = self.subsample_and_crop_video(subsample=1,
                                                      start_frame=start_frame,
                                                      end_frame=end_frame,
                                                      crop=self.validate_crop(crop))
        
        photon_gain_dict = self.get_photon_gain_parameters(start_frame=start_frame, end_frame=end_frame)
        background_noise_mean = -photon_gain_dict['offset'] / photon_gain_dict['slope']
        photon_flux = (cropped_video.flatten() - background_noise_mean) / photon_gain_dict['slope']

        return {'photon_flux_median': np.median(photon_flux),
                'photon_gain': photon_gain_dict['slope'],
                'background_noise': background_noise_mean,
                'photon_offset': photon_gain_dict['offset']}

    def get_photon_gain_parameters(self, start_frame=1, end_frame=2001, crop=(150,150), perc_min=3, perc_max=90):
        """Photon Gain.
        
        Compute a variety of parameters related to the physio signal's gain.
        
        Args: 
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile
        Returns: 
            A dictionary of parameters related to the physio signal.  Useful in making plots and metrics.
        """
        cropped_video = self.subsample_and_crop_video(subsample=1,
                                                      start_frame=start_frame,
                                                      end_frame=end_frame,
                                                      crop=self.validate_crop(crop))

        # Remove saturated pixels
        dynamic_range = self.get_dynamic_range()
        idxs_not_saturated = np.where(cropped_video.max(axis=0).flatten() < dynamic_range[1])

        _var = cropped_video.var(axis=0).flatten()[idxs_not_saturated]
        _mean = cropped_video.mean(axis=0).flatten()[idxs_not_saturated]

        # Remove pixels that deviate from Poisson stats
        _var_scale = np.percentile(_var, [perc_min, perc_max])
        _mean_scale = np.percentile(_mean, [perc_min, perc_max])
        
        # Remove outliers
        _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
        _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
        _no_outliers = np.logical_and(_var_bool, _mean_bool)

        _var_filt = _var[_no_outliers]
        _mean_filt = _mean[_no_outliers]
        _mat = np.vstack([_mean_filt, np.ones(len(_mean_filt))]).T
        try:
            slope, offset = np.linalg.lstsq(_mat, _var_filt, rcond=None)[0]
        except LinAlgError:
            raise DataCorruptionError('Unable to get photon metrics - check video for anomalies.')

        return {'var': _var_filt,
                'mean': _mean_filt,
                'slope': slope,
                'offset': offset}

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        local_comp1 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_505811062.h5",
        )
        
        local_comp2 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_509904120.h5",
        )
        
        local_comp3 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_637998955.h5",
        )
        
        local_comp4 = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "comp_657391625.h5",
        )
        
        self.local_comp = [local_comp1, local_comp2, local_comp3, local_comp4]

    def plot_one_frame(self):
        image = self.data_pointer[0].data_pointer[100,:,:]
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        local_shape_raw = [512, 512]
        rectangle_length = 100 * local_shape_raw[0] / 400
        rect = matplotlib.patches.Rectangle(
            [20, local_shape_raw[0] - 30], rectangle_length, 15, angle=0.0, color="w"
        )
        plt.gca().add_patch(rect)

    def plot_x_pixels(self, ax, x_pixels=10):
        for ind_i in range(x_pixels):
            x = np.random.randint(512)
            y = np.random.randint(512)

            local_trace = self.data_pointer[0].data_pointer[:,x,y]
            plt.sca(ax[ind_i])
            plt.plot(1/30*np.arange(len(local_trace)), local_trace)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            
    def load_data(self):
        self.data_pointer = []
        for local_comp_file in self.local_comp:
            local_pointer = Qc2p(filepath=local_comp_file, h5_path_to_data="data_raw")
            self.data_pointer.append(local_pointer)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)

    def make_figure(self):

        self.fig = plt.figure(figsize=(18, 16))
        
        ax = placeAxesOnGrid(
            self.fig,  xspan=[0.05, 0.45], yspan=[0.05, 0.45]
        )
        self.plot_one_frame()
        plt.text(-0.15, 1, "A", fontsize=15, weight="bold", transform=ax.transAxes)

        x_pixels = 10
        ax = placeAxesOnGrid(
            self.fig, dim=[10, 1], xspan=[0.55, 0.95], yspan=[0.05, 0.45]
        )
        
        self.plot_x_pixels(ax = ax, x_pixels=x_pixels)
        plt.text(-0.25, 1.1, "B", fontsize=15, weight="bold", transform=ax[0].transAxes)
        plt.sca(ax[9])
        plt.xlabel('Time (s)')
        
        plt.sca(ax[5])
        plt.ylabel('Fluorescence (AU)')
        
        plt.sca(ax[0])
        plt.title('Individual pixel traces')     
        
        ax = placeAxesOnGrid(
            self.fig, dim=[2, 2], xspan=[0.05, 0.95], yspan=[0.55, 0.95]
        )
        for x_ind in range(2):
            for y_ind in range(2):  
                plt.sca(ax[x_ind][y_ind])
                self.data_pointer[2*x_ind+y_ind].plot_poisson_curve()
                plt.title('Movie '+str(2*x_ind+y_ind))

                if x_ind==0:
                    plt.gca().set_xlabel('')
                    
        plt.sca(ax[0][0])

        plt.text(-0.10, 1.1, "C", fontsize=15, weight="bold", transform=ax[0][0].transAxes)

if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Supp Figure 1 - shot_noise.pdf",
    )
    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
