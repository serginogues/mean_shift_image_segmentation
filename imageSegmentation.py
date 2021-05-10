"""
Image Segmentation with mean-shift algorithm
"""
from utils import post_process
from mean_shift import *
from visualization import *


def segmIm(im, r, c=4, dim=True):
    """
    Image Segmentation with mean-shift algorithm
    """

    if dim:
        # 3d feature space
        points = im.reshape(-1, 3)
    else:
        points = []
        for row in range(int(im.shape[0])):
            for col in range(int(im.shape[1])):
                points.append([im[row, col][0], im[row, col][1], im[row, col][2], row, col])
        points = np.array(points).reshape(-1, 5)

    labels, peaks = meanshift(data=points, r=r, c=c)
    segmented_img = post_process(labels, peaks, im, dim)

    # plotclusters3D(points, labels, peaks)

    return segmented_img, len(peaks)
