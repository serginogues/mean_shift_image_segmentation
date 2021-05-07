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


def cdist(data_point, data, metric='euclidian'):
    """
    find distances between a data point and all other data
    """


def find_peak_opt(data, idx, r, threshold, c=4):
    """
    - First SPEEDUP basin of attraction: Upon finding a peak, associate each data point that is at a distance≤ r from the peak with the cluster defined by that peak.\n
    - Second SPEEDUP: points that are within a distance of r/c of the search path are associated with the converged peak \n
    :param threshold:
    :param c: = 4 but you are also asked to check other values
    :param data: n-dimensional dataset consisting of p points
    :param idx: column index of the data point for which we wish to compute its associated density peak
    :param r: search window radius
    :return: peak, cpts (vector storing a 1 for each point that is a distance of r/4 from the path and 0 otherwise)
    """
    print("Peak search starts")

    return peak, cpts


def meanshift_opt(data, r, c):
    """
    Calls findpeak() for each point and then assigns a label to each point according to its peak.
    Peaks are compared after each call to the findpeak function and similar peaks (distance between
    them is smaller than r/2) are merged. Also, if the found peak already exists in PEAKS, it is discarded and
    the data point is given the associated peak label in PEAKS.
    :param c:
    :param data:
    :param r:
    :return: LABELS: vector containing the label for each data point (labels) - PEAKS: matrix storing the density peaks found using meanshift() as its columns
    """
    print("Mean-Shift algorithm starts")
    peak, cpts = find_peak_opt(data, 0, r, 5, c)
    found = np.argwhere(labels)  # labels[labels>0]
    distances = cdist(data_point, data)

