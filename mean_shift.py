from config import *

"""
Mean-shift algorithm
- Get data points (with many features, i.e. RGB values)
- Choose feature (column of the data points vector)
- For each data-point/pixel/row:
    a) Init window of size r
    a.1) Apply Gaussian in window (to place more importance to the data points closest to density center)
    b) Implement findpeak() until convergence (peak is found):
        - Calculate mean within data points in window
        - Shift window center to the mean
        - repeat until peak found (data points equally distributed, mean = window center)
    c) Assign label to each point according to its peak (cluster it belongs to) 
"""


def find_peak(data, idx, r, c, PLOT_ALL=False):
    """
    Assign a label to data[idx] corresponding to its associated peak
    :param data: n-dimensional dataset consisting of p points
    :param idx: index of the data point for which we wish to compute its associated density peak
    :param r: search window radius
    :return: peak - n-dim array with "coordinates" of the found peak
             cpts - vector storing a 1 for each point that is a distance<=r/c from the path, 0 otherwise
             close - vector storing a 1 for each point that is at a distance≤ r from the peak, 0 otherwise
    """
    peak = np.zeros(data.shape[1])
    cpts = np.zeros(data.shape[0])
    close = np.zeros(data.shape[0])
    window = data[idx]
    num_points_inside = 0

    if PLOT_ALL:
        plt.subplot(1, 3, 1)
        plt.scatter(data[:, 0], data[:, 1], c='c')
        plt.scatter(window[0], window[1], c='r')

    # PEAK
    path = [window]
    found = False
    while not found:
        distances = np.array(cdist(window.reshape(1, -1), data, metric='euclidean').reshape(-1, 1))
        indices = np.where(distances < r)[0]
        data_inside = data[indices]
        peak = np.mean(data_inside, axis=0)
        path.append(peak)
        if abs(num_points_inside - data_inside.shape[0]) < 1:
            found = True
            path = np.array(path)
        else:
            window = peak
            if PLOT_ALL:
                plt.scatter(window[0], window[1], c='m')
            num_points_inside = data_inside.shape[0]

    # CPTS
    for pos in path:
        distances = np.array(cdist(pos.reshape(1, -1), data, metric='euclidean').reshape(-1, 1))
        indices = np.where(distances < (r / c))[0]
        cpts[indices] = 1

    # SPEEDUP: Upon finding a peak, associate each data point that is at a distance≤ r from the peak with the cluster defined by that peak.
    distances2 = np.array(cdist(peak.reshape(1, -1), data, metric='euclidean').reshape(-1, 1))
    indices = np.where(distances2 <= r)[0]
    close[indices] = 1

    if PLOT_ALL:
        plt.subplot(1, 3, 2)
        plt.scatter(data[:, 0], data[:, 1], c='c')
        plt.scatter(data[:, 0], data[:, 1], c=cpts, cmap="viridis")

        plt.subplot(1, 3, 3)
        plt.scatter(data[:, 0], data[:, 1], c='c')
        plt.scatter(data[:, 0], data[:, 1], c=close, cmap="viridis")
        plt.show()

    return peak, cpts, close


def meanshift(data, r, c):
    """
    :returns: labels: vector containing the label (cluster label) for each data point
             peaks: each column of the matrix is a peak. For each peak we have n-dim values (3D in case RGB)
    """
    LABEL_INIT = 200000
    LABEL_XTRA = 100000
    labels = np.full(data.shape[0], LABEL_INIT)
    peaks = []
    end = False
    while not end:
        idx = np.nonzero(labels == LABEL_INIT)[0][0]
        print(idx, "out of", data.shape[0])

        peak, cpts, close = find_peak(data, idx, r, c)
        if np.count_nonzero(close) > 2:
            # After each call findpeak(), similar peaks (distance between them is smaller than r/2) are merged
            # and the data point is given the associated peak label in PEAKS.

            similar = [x for x in peaks if cdist(peak.reshape(1, -1), x.reshape(1, -1), metric='euclidean') < r / 2]
            if len(similar) == 1:
                peak = similar[0]
            else:
                peaks.append(peak)

            array = np.array(peaks)
            label = np.where(array == peak)[0][0]

            # SPEEDUP: points that are within a distance of r/c of the search path are associated with the converged peak
            indices = np.nonzero(cpts)[0]
            labels[indices] = label
            indices = np.nonzero(close)[0]
            labels[indices] = label
            labels[idx] = label
            # print(np.count_nonzero(labels != 200), "labeled points out of", len(data))
        else:
            labels[idx] = LABEL_XTRA

        if np.count_nonzero(labels == LABEL_INIT) == 0:
            end = True

    print("Found", len(peaks), "peaks")

    # for pixels without label, assign closest peak/label
    indices = np.where(labels == LABEL_XTRA)[0]
    for i in indices:
        listt = []
        for peak in peaks:
            listt.append([peak, cdist(peak.reshape(1, -1), data[i].reshape(1, -1), metric='euclidean')[0][0]])
        listt = np.asarray(listt, dtype="object")
        label = np.where(listt[:, 1] == min(listt[:, 1]))[0][0]
        labels[i] = label

    peaks = np.array(peaks)
    return labels, peaks
