"""
Image Segmentation with mean-shift algorithm
"""
from utils import post_process
from mean_shift import *
from visualization import plotclusters3D, cv2


def segmIm(im, r, c=4, rgb=True):
    """
    Image Segmentation with mean-shift algorithm
    """

    if rgb:
        # 3d feature space
        points = im.reshape(-1, 3)
    else:
        #ToDO: 5d feature space
        print("not implemented")

    labels, peaks = meanshift(data=points, r=r, c=c)
    segmented_img = post_process(labels, peaks, im)

    plotclusters3D(points, labels, peaks)

    return segmented_img, len(peaks)
