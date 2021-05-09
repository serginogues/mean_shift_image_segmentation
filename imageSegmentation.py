"""
Image Segmentation with mean-shift algorithm
"""
from mean_shift import meanshift
from visualization import plot_all


def segmIm(im, r):
    """
    :param im: ndarray with shape (num data points,3) - input color RGB image
    :param r: radius
    """
    labels, peaks = meanshift(im, r, c=4)
    plot_all(im, labels, peaks)
    # plotclusters3D(im, labels.tolist(), np.array(peaks).tolist())
