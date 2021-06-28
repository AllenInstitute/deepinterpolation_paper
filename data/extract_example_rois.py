import numpy as np
import matplotlib.pylab as plt
import os

path_filter_raw = r"X:\510021399\raw\suite2p\plane0\stat.npy"
path_filter_den = r"X:\510021399\suite2p\plane0\stat.npy"
path_save = r"X:\510021399\matched_cells"

def load_cell_filters(path_stat):
    data_filter = np.load(path_stat, allow_pickle=True)

    all_cells = np.zeros((512, 512, len(data_filter)))
    meanx = np.zeros((len(data_filter)))
    meany = np.zeros((len(data_filter)))

    for neuron_nb in range(len(data_filter)):
        list_x = data_filter[neuron_nb]["xpix"]
        list_y = data_filter[neuron_nb]["ypix"]
        weight = data_filter[neuron_nb]["lam"]
        all_cells[list_y, list_x, neuron_nb] = weight
        meanx[neuron_nb] = np.mean(list_x)
        meany[neuron_nb] = np.mean(list_y)

    return all_cells, meanx, meany

def zoom_filter(imarray):
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
    cmin = np.percentile(final_img.flatten(), 0)
    cmax = np.percentile(final_img.flatten(), 99.97)

    plt.imshow(final_img, cmap="gray", clim=[cmin, cmax], aspect='auto')

cell_filter_raw, meanx_raw, meany_raw = load_cell_filters(path_filter_raw)
cell_filter_den, meanx_den, meany_den = load_cell_filters(path_filter_den)

for index, neuron_nb in enumerate(range(cell_filter_raw.shape[2])):
    mean_positionx = meanx_raw[index]
    mean_positiony = meany_raw[index]
    distance = np.sqrt((meanx_den-mean_positionx)**2 + (meany_den-mean_positiony)**2)

    index_den = np.argmin(distance)

    plt.figure(figsize=[20, 10])
    plt.subplot(1,2,1)
    zoom_filter(cell_filter_raw[:,:,index])
    plt.axis('off')
    plt.subplot(1,2,2)
    zoom_filter(cell_filter_den[:,:,index_den])
    plt.axis('off')

    local_cell_path = os.path.join(path_save, 'cell_'+str(index)+'.png')
    plt.savefig(local_cell_path)
    plt.close()