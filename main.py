"""
Own image segmentation application, using the Mean-shift algorithm.
"""
import io
from scipy import io
import numpy as np


if __name__ == '__main__':
    print("Not implemeneted")

    """
    Debug your algorithm using the data set (pts.mat which stores a set of 3D points belonging
    to two 3D Gaussians) provided with the assignment zip folder with  r = 2 (this should give
    two clusters)
    """
    r = 2
    mat = io.loadmat(r'data/pts.mat')
    data = np.array(mat['data'])
