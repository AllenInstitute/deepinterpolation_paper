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
from PIL import Image

class Figure:
    def __init__(self, output_file):
        self.output_file = output_file

        self.path_to_raw_clean = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_clean.npy",
        )
        self.path_to_raw_noisy = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_noisy.npy",
        )
        self.path_to_di = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_di_3.npy",
        )
        self.path_to_pmd = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_pmd.npy",
        )
        
        self.path_to_n2v = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_n2v.tif",
        )

        self.path_to_roi = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "Ground_truth_RoiSet.zip",
        )

        self.path_to_gaussian = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "data",
            "local_large_data",
            "ground_truth",
            "20191215_raw_gaussian.npy",
        )

    def fill_rois(self):
        img = np.zeros([256, 256])

        self.final_roi = dict()
        for index, roi in enumerate(self.rois.keys()):
            # if index==4 or index==5 or index==6:
            img = np.zeros([256, 256])
            local_x = self.rois[roi]["x"]
            local_y = self.rois[roi]["y"]
            list_contours = [
                [local_x[index], local_y[index]] for index in range(len(local_x))
            ]
            cv2.fillPoly(img, pts=[np.array(list_contours)], color=(1, 1, 1))
            self.final_roi[roi] = img.T

    def load_data(self):

        self.raw_movie_clean = np.load(self.path_to_raw_clean)[:,:,30:-30]
        self.raw_movie_noisy = np.load(self.path_to_raw_noisy)[:,:,30:-30]
        self.raw_movie_gaussian = np.load(self.path_to_gaussian)[:,:,30:-30]

        self.di_movie = np.load(self.path_to_di)[:,:,:]
        self.pmd_movie = np.load(self.path_to_pmd)[:,:,30:-30]
        self.rois = read_roi_zip(self.path_to_roi)
        self.fill_rois()

        dataset = Image.open(self.path_to_n2v)
        h,w = np.shape(dataset)
        tiffarray = np.zeros((h,w,dataset.n_frames))
        for i in range(dataset.n_frames):
            dataset.seek(i)
            tiffarray[:,:,i] = np.transpose(np.array(dataset))
        self.n2v_movie  = tiffarray[:,:,30:-30].astype(np.double)

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

    def get_av_trace_plot(self, input_mov, input_roi):
        local_trace = []
        for index_frame in range(input_mov.shape[2]):
            local_img_mov = input_mov[:, :, index_frame]
            local_trace.append(np.mean(local_img_mov[input_roi == 1.0].flatten()))
        return (local_trace - np.mean(local_trace)) / np.mean(local_trace)

    def get_error_comparison(self):
        list_mov = {
            "ground_truth": self.raw_movie_clean,
            "noisy": self.raw_movie_noisy,
            "Noise2Void": self.n2v_movie,
            "Gaussian": self.raw_movie_gaussian,
            "PMD": self.pmd_movie,
            "deep interpolation": self.di_movie,
        }

        l2_dict = {}
        for index, local_key in enumerate(self.final_roi.keys()):
            local_roi = self.final_roi[local_key]
            local_mov = list_mov["ground_truth"]
            local_trace_ground = self.get_av_trace_plot(local_mov, local_roi)

            for mov_key in list_mov.keys():
                if mov_key != "ground_truth":
                    local_mov = list_mov[mov_key]
                    local_trace = self.get_av_trace_plot(local_mov, local_roi)
                    local_l2 = np.sqrt(np.sum((local_trace - local_trace_ground) ** 2))
                    try:
                        l2_dict[mov_key].append(local_l2)
                    except:
                        l2_dict[mov_key] = [local_l2]
        self.l2_dict = l2_dict

    def plot_error_comparison(self, norm=False):
        if norm == False:
            w = 0.8
            plt.bar(
                x=["noisy", "Noise2Void", "PMD", "Gaussian\nkernel", "Deep\nInterpolation"],
                height=[
                    np.mean(self.l2_dict["noisy"]),
                    np.mean(self.l2_dict["Noise2Void"]),
                    np.mean(self.l2_dict["PMD"]),                    
                    np.mean(self.l2_dict["Gaussian"]),
                    np.mean(self.l2_dict["deep interpolation"]),
                ],
                width=w,
                color=(0, 0, 0, 0),
                edgecolor="black",
            )

            plt.xticks(fontsize=8)

            for index, key in enumerate(self.l2_dict.keys()):
                # distribute scatter randomly across whole width of bar
                plt.gca().scatter(
                    index + np.random.random(len(self.l2_dict[key])) * w - w / 2,
                    self.l2_dict[key],
                    color="black",
                )

            plt.ylabel("L2 norm of error with ground truth")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
        else:
            for indiv_key in self.l2_dict.keys():
                if "noisy" not in indiv_key:
                    if "interpolation" not in indiv_key:
                        plt.plot(
                            self.l2_dict["noisy"],
                            np.array(self.l2_dict[indiv_key])
                            / np.array(self.l2_dict["deep interpolation"]),
                            '.',
                            label=indiv_key,
                            markersize=2
                        )

            plt.legend(prop={"size": 6}, frameon=False, loc='upper right')
            plt.xlabel("Raw data: L2 norm\nerror with ground truth")
            plt.ylabel("Improvement of\nDeepInterpolation\nover another method")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)

    def plot_example_movie_roi(self, current_mov, number_traces=False, scale_bar=False):

        frame_index = 200
        list_roi = [0, 1, 3, 4, 5]

        # We first create the ROI + image of the figure
        local_img = current_mov[:, :, frame_index]

        local_img = self.convert_gray_rgb(local_img)

        list_color = [50, 50, 50, 50, 50, 50, 50, 50, 50]
        index_trace = 0

        for index_key, local_key in enumerate(self.final_roi.keys()):
            if index_key in list_roi:
                index_trace = index_trace + 1

                local_roi = self.final_roi[local_key]
                color = list_color[index_key]
                if number_traces:
                    [location_roi_x, location_roi_y] = np.where(local_roi > 0)
                    x_mean = np.mean(location_roi_x)
                    y_mean = np.mean(location_roi_y)

                    plt.text(y_mean, x_mean + 15, str(index_trace), color="red")

                local_img = self.colorized(local_img, local_roi, color)

        plt.imshow(local_img)
        plt.gca().axis("off")

        if scale_bar:
            rectangle_length = 100 * current_mov.shape[0] / 400
            rect = matplotlib.patches.Rectangle(
                [20, 230], rectangle_length, 10, angle=0.0, color="w"
            )
            plt.gca().add_patch(rect)

    def plot_example_roi_traces(
        self, ax, current_mov, ground_mov, number_traces=False, time_axis=True
    ):

        list_mov = {"extracted trace": current_mov, "ground truth trace": ground_mov}
        start_trace = 200
        list_roi = [0, 1, 3, 4, 5]
        index_trace = 0
        for index, local_key in enumerate(self.final_roi.keys()):
            if index in list_roi:
                plt.sca(ax[index_trace])

                index_trace = index_trace + 1
                local_roi = self.final_roi[local_key]
                for index_key, mov_key in enumerate(list_mov.keys()):
                    local_mov = list_mov[mov_key]
                    local_trace = self.get_av_trace_plot(local_mov, local_roi)

                    plt.plot(100 * local_trace, label=mov_key, linewidth=0.5)

                plt.gca().spines["bottom"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)

                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                plt.ylim(-15, 25)
                plt.xlim(start_trace, 2000)

                if number_traces:
                    plt.text(start_trace - 60, 0, str(index_trace), color="red")

                #if number_traces and index_trace == 4:
                #    plt.legend(loc="lower right", prop={"size": 6})

        if time_axis:
            rect = matplotlib.patches.Rectangle(
                [start_trace + 1, -10], 300, 1, angle=0.0, color="k",
            )
            plt.gca().add_patch(rect)
            rect = matplotlib.patches.Rectangle(
                [start_trace + 1, -10], 1, 25, angle=0.0, color="k",
            )

            plt.gca().add_patch(rect)
            plt.text(
                0.04, -0.175, "10s ", fontsize=6, transform=plt.gca().transAxes,
            )
            plt.text(
                -0.20,
                0.05,
                "25 %\n$\Delta$F/F (%)",
                fontsize=6,
                transform=plt.gca().transAxes,
            )


    def plot_fourier_transform_trace(self, current_mov, ref_fft = [], legend_text='None', window=False, color = 'k'):
    
        list_mov = {"extracted trace": current_mov}
        start_trace = 200
        list_roi = np.arange(len(self.final_roi.keys())) #[0, 1, 3, 4, 5]
        index_trace = 0
        for index, local_key in enumerate(self.final_roi.keys()):
            if index in list_roi:
                local_roi = self.final_roi[local_key]
                for index_key, mov_key in enumerate(list_mov.keys()):
                    local_mov = list_mov[mov_key]
                    local_trace = self.get_av_trace_plot(local_mov, local_roi)


                    # local_trace = (local_trace-np.mean(local_trace))/np.mean(local_trace)

                    local_fft = fft(local_trace)
                    T = 1/30.0
                    N = len(local_fft)
                    local_xfft = fftfreq(N, T)[:N//2]
                    local_yfft = 2.0/N * np.abs(local_fft[0:N//2])

                    if index == 0:
                        all_yfft = np.zeros((len(list_roi), len(local_yfft)))
                    
                    all_yfft[index_trace, :] = local_yfft
                    index_trace = index_trace + 1

        if ref_fft == []:
            all_mean = np.mean(all_yfft, axis = 0)
            all_sem = sem(all_yfft, axis = 0)
        else:
            all_mean = np.mean(all_yfft, axis = 0)-ref_fft

        plt.semilogy(local_xfft[1:], all_mean[1:], label=legend_text, color = color)
        plt.xlim([0, 10])
        
        return all_mean


    def plot_fourier_transform(self, current_mov, ref_fft = [], legend_text='None', window=False):
        list_roi = np.arange(0, 1000)
        local_ref = []
        local_error = []
        rise_time = 0.100
        decay_time = 1.5
        period = 1/30
        time_array = np.arange(30)*period
        tau_decay = decay_time/np.log(2)
        tau_rise = rise_time/np.log(2)

        trace_decay = np.exp(-time_array/tau_decay)
        tau_rise = rise_time/np.log(2)
        trace_rise = np.exp(-time_array/tau_rise)
        win = np.concatenate([trace_rise[::-1], trace_decay])
        win = win/np.sum(win)

        # win = signal.windows.triang(110)

        for index, local_key in enumerate(list_roi):
                x = np.random.randint(256)
                y = np.random.randint(256)
                local_trace = current_mov[x, y, :]

                local_trace = local_trace/np.mean(local_trace)-1
                
                if window:
                    local_trace = signal.convolve(local_trace, win, mode='same') / sum(win)

                local_fft = fft(local_trace)
                T = 1/30.0
                N = len(local_fft)
                local_xfft = fftfreq(N, T)[:N//2][1:]
                local_yfft = 2.0/N * np.abs(local_fft[0:N//2])[1:]

                if index == 0:
                    all_yfft = np.zeros((len(list_roi), len(local_yfft)))
                
                all_yfft[index, :] = local_yfft

        if ref_fft == []:
            all_mean = np.mean(all_yfft, axis = 0)
            all_sem = sem(all_yfft, axis = 0)
        else:
            all_mean = np.mean(all_yfft, axis = 0)-ref_fft

        plt.semilogy(local_xfft, all_mean, label=legend_text)
        plt.xlim([0, 10])

        return all_mean

    def get_fft(self, local_trace, T, N):
        local_fft = fft(local_trace)
    
        local_xfft = fftfreq(N, T)[:N//2]
        local_yfft = 2.0/N * np.abs(local_fft[0:N//2])
        return local_xfft, local_yfft

    def compare_dff(self, current_mov, ref_movie, legend_text='None'):
        list_roi = np.arange(0, 100)
        local_ref = []
        local_error = []
        for index, local_key in enumerate(list_roi):
            x = np.random.randint(256)
            y = np.random.randint(256)
            local_trace = current_mov[x, y, :]
            local_trace_ref = ref_movie[x, y, :]
            local_trace_dff = local_trace/np.mean(local_trace)-1
            local_trace_ref_dff = local_trace_ref/np.mean(local_trace_ref)-1
            error_dff = (local_trace_dff-local_trace_ref_dff)**2
            local_error.extend(error_dff)
            local_ref.extend(local_trace_ref_dff)

        plt.hist2d(local_ref, local_error, bins = [20, 20])


    def compare_fourier_transform(self, current_mov, ref_movie, legend_text='None'):
        # list_roi = [0, 1, 3, 4, 5]
        list_roi = np.arange(0, 1000)
        iterator = 0
        for index, local_key in enumerate(list_roi):
            #local_roi = self.final_roi[local_key]
            x = np.random.randint(256)
            y = np.random.randint(256)
            local_trace = current_mov[x, y, :]
            local_trace_ref = ref_movie[x, y, :]

            T = 1/30.0
  
            local_trace = local_trace/np.mean(local_trace)-1
            local_trace_ref = local_trace_ref/np.mean(local_trace_ref)-1

            local_xfft, local_yfft = self.get_fft(local_trace, T, len(local_trace))
            local_xfft_ref, local_yfft_ref = self.get_fft(local_trace_ref, T, len(local_trace))


            if index == 0:
                all_xfft = np.zeros((len(list_roi), len(local_xfft)))
            
            all_xfft[iterator, :] = np.abs(local_yfft/local_yfft_ref)
            iterator += 1
        all_mean = np.mean(all_xfft, axis = 0)
        all_sem = sem(all_xfft, axis = 0)

        plt.semilogy(local_xfft, all_mean, label=legend_text)
        plt.xlim([0, 10])
        plt.xlabel('Frequency (Hz)')

        return all_mean


    def make_figure(self):

        self.fig = plt.figure(figsize=(9, 16))

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.05, 0.35), yspan=(0.1, 0.25))

        plt.sca(local_axis)
        plt.text(
            -0.2, 1.05, "A", fontsize=15, weight="bold", transform=local_axis.transAxes
        )

        self.plot_example_movie_roi(self.raw_movie_noisy, number_traces=True)
        plt.text(
            0.75,
            1.05,
            "Simulated in vivo two-photon data",
            fontsize=10,
            transform=local_axis.transAxes,
        )
    
        local_axis = placeAxesOnGrid(
            self.fig, dim=[5, 1], xspan=(0.40, 0.7), yspan=(0.1, 0.25), wspace=0
        )
        self.plot_example_roi_traces(
            local_axis,
            self.raw_movie_noisy,
            self.raw_movie_clean,
            number_traces=True,
            time_axis=False,
        )

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.80, 1), yspan=(0.1, 0.25), wspace=0
        )

        #ref_ftt = self.plot_fourier_transform(self.raw_movie_noisy, legend_text='Ground truth + Noise')
        #ref_ftt = self.plot_fourier_transform(self.raw_movie_noisy, legend_text='Ground truth + Noise + Calcium kernel', window=True)
        #ref_ftt = self.plot_fourier_transform(self.raw_movie_clean, legend_text='Ground truth')
        self.plot_fourier_transform_trace(self.raw_movie_clean, legend_text='Ground truth', color='C1')
        self.plot_fourier_transform_trace(self.raw_movie_noisy, legend_text='Ground truth + Noise', color='C0')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.ylabel('Power density')

        plt.legend(prop={'size': 6}, frameon=False)
        ax = plt.gca()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.05, 0.35), yspan=(0.3, 0.45))
        plt.sca(local_axis)
        plt.text(
            -0.2, 1.05, "B", fontsize=15, weight="bold", transform=local_axis.transAxes
        )

        self.plot_example_movie_roi(self.pmd_movie)
        plt.text(
            0.6,
            1.05,
            "After Penalized-Matrix Decomposition (PMD)",
            fontsize=10,
            transform=local_axis.transAxes,
        )

        local_axis = placeAxesOnGrid(
            self.fig, dim=[5, 1], xspan=(0.4, 0.7), yspan=(0.3, 0.45)
        )
        self.plot_example_roi_traces(
            local_axis, self.pmd_movie, self.raw_movie_clean, time_axis=False
        )

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.80, 1), yspan=(0.3, 0.45), wspace=0
        )

        #self.compare_dff(self.di_movie, self.raw_movie_clean)
        #ref_ftt = self.plot_fourier_transform(self.di_movie, legend_text='Ground truth')
        self.plot_fourier_transform_trace(self.raw_movie_clean, legend_text='Ground truth', color = 'C1')
        self.plot_fourier_transform_trace(self.pmd_movie, legend_text='PMD', color = 'C0')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.ylabel('Power density')
        plt.legend(prop={'size': 6}, frameon=False)

        ax = plt.gca()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.05, 0.35), yspan=(0.5, 0.65))
        plt.sca(local_axis)
        plt.text(
            -0.2, 1.05, "C", fontsize=15, weight="bold", transform=local_axis.transAxes
        )

        self.plot_example_movie_roi(self.di_movie, scale_bar=True)
        plt.text(
            1,
            1.05,
            "After DeepInterpolation",
            fontsize=10,
            transform=local_axis.transAxes,
        )

        local_axis = placeAxesOnGrid(
            self.fig, dim=[5, 1], xspan=(0.4, 0.7), yspan=(0.5, 0.65)
        )
        self.plot_example_roi_traces(
            local_axis, self.di_movie, self.raw_movie_clean, time_axis=True
        )

        local_axis = placeAxesOnGrid(
            self.fig, xspan=(0.80, 1), yspan=(0.5, 0.65), wspace=0
        )

        # self.plot_fourier_transform(self.raw_movie_clean, ref_fft = ref_ftt, legend_text='Ground truth')
        #self.compare_fourier_transform(self.pmd_movie, self.raw_movie_clean, legend_text='PMD')
        #self.compare_fourier_transform(self.di_movie, self.raw_movie_clean, legend_text='Deep Interpolation')
        #self.compare_fourier_transform(self.di_movie, self.raw_movie_noisy, legend_text='Deep Interpolation')
        #self.compare_fourier_transform(self.pmd_movie, self.raw_movie_noisy, legend_text='Deep Interpolation')
        self.plot_fourier_transform_trace(self.raw_movie_clean, legend_text='Ground truth', color='C1')
        self.plot_fourier_transform_trace(self.di_movie, legend_text='DeepInterpolation', color = 'C0')
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power density')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(prop={'size': 6}, frameon=False)

        ax = plt.gca()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.05, 0.5), yspan=(0.7, 0.85))
        plt.sca(local_axis)
        plt.text(
            -0.2, 1.05, "D", fontsize=15, weight="bold", transform=local_axis.transAxes
        )

        self.get_error_comparison()
        self.plot_error_comparison()

        local_axis = placeAxesOnGrid(self.fig, xspan=(0.65, 0.95), yspan=(0.7, 0.85))
        plt.sca(local_axis)
        plt.text(
            -0.2, 1.05, "E", fontsize=15, weight="bold", transform=local_axis.transAxes
        )
        self.plot_error_comparison(norm=True)

    def save_figure(self):
        self.fig.savefig(self.output_file, bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    output_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pdfs",
        "Ext Figure 1 - simulation.pdf",
    )

    if os.path.isfile(output_file):
        os.remove(output_file)

    local_figure = Figure(output_file)
    local_figure.load_data()
    local_figure.make_figure()
    local_figure.save_figure()
