"""
Image Segmentation with mean-shift algorithm
"""
from config import cv2
from mean_shift import meanshift_opt
from visualization import plotclusters3D


def segmIm(im, r):
    """
    Work using the CIELAB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    :param im: input color RGB image
    :param r: radius
    :return:
    """
    labels, peaks = meanshift_opt(im, r, c=4)
    # convert resulting cluster centers back to RGB
    cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    plotclusters3D(im, labels, peaks)
