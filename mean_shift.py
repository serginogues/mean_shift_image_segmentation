"""
Mean-shift algorithm
"""


def findpeak (data, idx, r):
    """
    :param data: n-dimensional dataset consisting of p points
    :param idx: column index of the data point for which we wish to compute its associated density peak
    :param r: search window radius
    :return:
    """
    print("Peak search starts")


def meanshift(data, r):
    """
    Calls findpeak() for each point and then assigns a label to each point according to its peak.
    Peaks are compared after each call to the findpeak function and similar peaks (distance between
    them is smaller than r/2) are merged. Also, if the found peak already exists in PEAKS, it is discarded and
    the data point is given the associated peak label in PEAKS.
    :param data:
    :param r:
    :param labels:
    :return: LABELS: vector containing the label for each data point (labels) - PEAKS: matrix storing the density peaks found using meanshift() as its columns
    """
    print("Mean-Shift algorithm starts")
