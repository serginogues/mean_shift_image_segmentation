from config import *


def plotclusters3D(data, labels, peaks):
    """
    Plots the modes of the given image data in 3D by coloring each pixel
    according to its corresponding peak.

    Args:
        data: image data in the format [number of pixels]x[feature vector].
        labels: a list of labels, one for each pixel.
        peaks: a list of vectors, whose first three components can
        be interpreted as BGR values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    bgr_peaks = np.array(peaks[:, 0:3], dtype=float)
    rgb_peaks = bgr_peaks[...,::-1]
    rgb_peaks /= 255.0
    for idx, peak in enumerate(rgb_peaks):
        color = np.random.uniform(0, 1, 3)
        #TODO: instead of random color, you can use peaks when you work on actual images
        # color = peak
        cluster = data[np.where(labels == idx)[0]].T
        ax.scatter(cluster[0], cluster[1], cluster[2], c=[color], s=.5)
    fig.show()
    plt.show()


def show_image(img, title='Segmented Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Display two images
def show_two_images(a, b, title1="Original", title2="Preprocessed"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
