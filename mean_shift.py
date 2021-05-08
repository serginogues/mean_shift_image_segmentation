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


def find_peak(data, idx, r, threshold, c=4):
    """
    Assign a label to data[idx] corresponding to its associated peak
    :param threshold:
    :param data: n-dimensional dataset consisting of p points
    :param idx: index of the data point for which we wish to compute its associated density peak
    :param r: search window radius
    :return: peak - n-dim array with "coordinates" of the found peak
             cpts - vector storing a 1 for each point that is a distance<=r/c from the path and 0 otherwise
    """
    peak = np.zeros(data.shape[1])
    cpts = np.zeros(data.shape[0])
    window = data[idx]
    num_points_inside = 0

    # PEAK
    path = []
    found = False
    while not found:
        distances = np.array(cdist(window.reshape(1, -1), data, metric='euclidean').reshape(-1, 1))
        indices = np.where(distances < r)[0]
        data_inside = data[indices]
        peak = np.mean(data_inside, axis=0)
        path.append(peak)
        if abs(num_points_inside-data_inside.shape[0]) < 3:
            found = True
            path = np.array(path)
            print("Found peak at", peak, "after", path.shape[0], "window shifts")
        else:
            window = peak
            num_points_inside = data_inside.shape[0]
    # CPTS
    for pos in path:
        distances = np.array(cdist(pos.reshape(1, -1), data, metric='euclidean').reshape(-1, 1))
        indices = np.where(distances < r/c)[0]
        cpts[indices] += 1

    return peak, cpts


def meanshift(data, r, c=4):
    """
    :returns: labels: vector containing the label (cluster label) for each data point
             peaks: each column of the matrix is a peak. For each peak we have n-dim values (3D in case RGB)
    """
    labels = np.zeros(data.shape[0])
    peaks = []

    #TODO:
    # 2 - After each call findpeak(), similar peaks (distance between them is smaller than r/2) are merged
    # 3 - If the found peak already exists in PEAKS, it is discarded and the data point is given the associated peak label in PEAKS.

    # 1
    for i in range(len(data)):
        peak, cpts = find_peak(data, i, r, 5, c)

        # cpts is a vector storing a 1 for each point that is a distance<=r/c from the path and 0 otherwise
        indices = np.nonzero(cpts)[0]


        # Upon finding a peak, associate each data point that is at a distance≤ r from the peak with the cluster defined by that peak.

        if peak not in peaks:
            peaks.append(peak)
        # 2 + 3

    peaks = np.array(peaks)

    labels, peaks = 0, 0
    return labels, peaks
