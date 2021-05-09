"""
Image Segmentation with mean-shift algorithm
"""
from mean_shift import *
from visualization import plotclusters3D, cv2


def segmIm(im, r, im_origin):
    """
    :param im_origin: original image
    :param im: ndarray with shape (num data points,3) - input color RGB image
    :param r: radius
    """
    cv2.imshow('Original Image', im_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    labels, peaks = meanshift(im, r, c=4)
    # plotclusters3D(im, labels, peaks)

    """Then convert the resulting cluster centers back to RGB"""
    segmented_img = peaks[np.reshape(labels, im_origin.shape[:2])]
    segmented_img = cv2.cvtColor(segmented_img.astype(np.uint8), cv2.COLOR_LAB2RGB)
    cv2.imshow('Segmented Image', segmented_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
