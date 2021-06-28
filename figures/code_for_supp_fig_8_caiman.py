import caiman
from caiman.source_extraction.cnmf import cnmf as cnmf
import matplotlib.pylab as plt
import numpy as np
import os

path_analysis_raw = r"C:\Users\jeromel\Documents\Projects\Deep2p\Publication\deep_interpolation_paper_dev\data\local_large_data\caiman\637998955_raw_analysis_results.hdf5"
path_analysis_deepinterp = r"C:\Users\jeromel\Documents\Projects\Deep2p\Publication\deep_interpolation_paper_dev\data\local_large_data\caiman\637998955_deepInterpolation_analysis_results.hdf5"

c, dview, n_processes = caiman.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

background=False

if background:
    cnm_raw = cnmf.load_CNMF(path_analysis_raw, n_processes=n_processes, dview=dview)
    raw_filters = cnm_raw.estimates.b.copy()
    raw_filters = np.reshape(raw_filters, [512, 512, -1])

    cnm_deep = cnmf.load_CNMF(path_analysis_deepinterp, n_processes=n_processes, dview=dview)
    deep_filters = cnm_deep.estimates.b.copy()
    deep_filters = np.reshape(deep_filters, [512, 512, -1])

    Cdeep = cnm_deep.estimates.f.copy()
    Craw = cnm_raw.estimates.f.copy()
else:
    cnm_raw = cnmf.load_CNMF(path_analysis_raw, n_processes=n_processes, dview=dview)
    raw_filters = cnm_raw.estimates.A.toarray().copy()
    raw_filters = np.reshape(raw_filters, [512, 512, -1])

    cnm_deep = cnmf.load_CNMF(path_analysis_deepinterp, n_processes=n_processes, dview=dview)
    deep_filters = cnm_deep.estimates.A.toarray().copy()
    deep_filters = np.reshape(deep_filters, [512, 512, -1])

    Cdeep = cnm_deep.estimates.C.copy()
    Craw = cnm_raw.estimates.C.copy()

if 'dview' in locals():
    caiman.stop_server(dview=dview)

raw_ind_x = []
raw_ind_y = []
deep_ind_x = []
deep_ind_y = []

def zoom_filter(imarray, square_size=512):
    orig_shape = imarray.shape
    # We locate the ROI center
    value = np.max(imarray.flatten())
    square_size = 100
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

    final_img = imarray[int(xmin) : int(xmax), int(ymin) : int(ymax)]

    return final_img

for local_roi_index in range(raw_filters.shape[2]):
    pos_raw = np.unravel_index(np.argmax(raw_filters[:,:, local_roi_index], axis=None), raw_filters.shape)
    pos_raw = np.argwhere(raw_filters[:,:, local_roi_index] == np.max(raw_filters[:,:, local_roi_index]))[0]
    raw_ind_x.append(pos_raw[0])
    raw_ind_y.append(pos_raw[1])

for local_roi_index in range(deep_filters.shape[2]):
    pos_deep = np.unravel_index(np.argmax(deep_filters[:,:, local_roi_index], axis=None), deep_filters.shape)
    pos_deep = np.argwhere(deep_filters[:,:, local_roi_index] == np.max(deep_filters[:,:, local_roi_index]))[0]

    deep_ind_x.append(pos_deep[0])
    deep_ind_y.append(pos_deep[1])

path_out = r'C:\Users\jeromel\Documents\Projects\Deep2p\Publication\deep_interpolation_paper_dev\data\local_large_data\caiman\comparison_filters'
plt.figure()    
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,3,4)
ax5 = plt.subplot(2,3,5)
ax6 = plt.subplot(2,3,6)

soma_list = np.arange(deep_filters.shape[2]) #[155, 156, 171, 159, 161, 176, 177, 183, 252]
zoomed = False
for local_roi_index in soma_list:
    distance = np.sqrt((raw_ind_x-deep_ind_x[local_roi_index])**2+(raw_ind_y-deep_ind_y[local_roi_index])**2)
    ind_min = np.argmin(distance)

    plt.sca(ax1)
    plt.cla()
    
    plt.imshow(raw_filters[:,:,ind_min])

    plt.sca(ax2)
    plt.cla()
        
    plt.imshow(zoom_filter(raw_filters[:,:,ind_min]))
    vmin, vmax = plt.gci().get_clim()

    plt.title("x="+str(deep_ind_x[local_roi_index])+" y="+str(deep_ind_y[local_roi_index]))
    plt.sca(ax3)
    plt.cla()
    plt.plot(Craw[ind_min][30:1000])

    plt.sca(ax4)
    plt.cla()
        
    plt.imshow(deep_filters[:,:,local_roi_index], clim=[vmin, vmax])

    plt.sca(ax5)
    plt.cla()

    plt.imshow(zoom_filter(deep_filters[:,:,local_roi_index]), clim=[vmin, vmax])

    plt.sca(ax6)
    plt.cla()
    plt.plot(Cdeep[local_roi_index][0:971])

    if background:
        path_file = 'back_'+str(local_roi_index)
    else:
        path_file = 'filter_'+str(local_roi_index)
    
    if not(zoomed):
        path_file = 'wide_'+path_file


    plt.savefig(os.path.join(path_out, path_file))