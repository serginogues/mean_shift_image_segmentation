from config import *

#TODO: convert resulting cluster centers back to RGB
# cv2.cvtColor(im, cv2.COLOR_LAB2RGB)


def plot_all(data, labels, peaks):
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c='c')
    plt.scatter(peaks[:, 0], peaks[:, 1], c='r')

    plt.subplot(1, 2, 2)
    plt.scatter(data[:, 0], data[:, 1], c='c')
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")

    plt.show()


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
