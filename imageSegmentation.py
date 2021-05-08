"""
Image Segmentation with mean-shift algorithm
"""
from config import cv2
from mean_shift import meanshift
from visualization import plotclusters3D


def segmIm(im, r):
    """
    :param im: ndarray with shape (num data points,3) - input color RGB image
    :param r: radius
    """
    labels, peaks = meanshift(im, r, c=4)

    #TODO: convert resulting cluster centers back to RGB
    cv2.cvtColor(im, cv2.COLOR_LAB2RGB)
    plotclusters3D(im, labels, peaks)
