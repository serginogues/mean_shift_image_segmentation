"""
Image Segmentation with mean-shift algorithm
"""
from config import *


def segmIm(im, r):
    """
    Work using the CIELAB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    :param im: input color RGB image
    :param r: radius
    :return:
    """
    cv2.cvtColor(im, cv2.COLOR_RGB2LAB)  # cluster the image data in CIELAB color space by first converting the RGB color vectors to CIELAB

    cv2.cvtColor(im, cv2.COLOR_LAB2RGB)  # convert resulting cluster centers back to RGB
