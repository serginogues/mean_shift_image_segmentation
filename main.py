"""
Own image segmentation application, using the Mean-shift algorithm.

Steps:
1. Read the image and apply the preprocessing steps suggested above.
2. Implement the Mean-shift algorithm, considering the suggested optimizations.
3. Apply the algorithm on the image features, transform the result back to an image and visualize the obtained segmentation.
4. Test different parameters, such as r, c, feature types
"""
from preprocess import load_image, load_mat
from imageSegmentation import segmIm

if __name__ == '__main__':

    """
    Debug your algorithm using the data set (pts.mat which stores a set of 3D points belonging
    to two 3D Gaussians) provided with the assignment zip folder with  r = 2 (this should give
    two clusters)
    """
    r = 2
    img, flatten = load_image()
    points = load_mat()
    segmIm(points, r)
