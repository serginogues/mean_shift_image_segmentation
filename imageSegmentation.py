"""
Image Segmentation with mean-shift algorithm
"""
from config import np
from utils import post_process
from mean_shift import meanshift
from visualization import plotclusters3D


def segmIm(im, r, c=4, dim=True):
    """
    Image Segmentation with mean-shift algorithm
    """
    points = reshape_im(im, dim)
    labels, peaks = meanshift(data=points, r=r, c=c)
    segmented_img = post_process(labels, peaks, im, dim)

    plotclusters3D(points, labels, peaks)

    return segmented_img, len(peaks)


def reshape_im(im, dim=True):
    """
    Prepare data for image segmentation. Handles 3D and 5D reshaping
    :param im: Image to be reshaped
    :param dim: if True 3D reshaping, 5D otherwise
    """
    if dim:
        # 3d feature space (R,G,B)
        points = im.reshape(-1, 3)
    else:
        # 5d feature space (R,G,B,x,y)
        points = []
        for row in range(int(im.shape[0])):
            for col in range(int(im.shape[1])):
                points.append([im[row, col][0], im[row, col][1], im[row, col][2], row, col])
        points = np.array(points).reshape(-1, 5)
    return points
