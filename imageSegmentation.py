"""
Image Segmentation with mean-shift algorithm
"""
from config import np, plt
from utils import post_process
from mean_shift import meanshift


def segmIm(im, r, c=4, dim=True):
    """
    Image Segmentation with mean-shift algorithm
    """
    print("---Image segmentation starts---")
    points = reshape_im(im, dim)
    labels, peaks = meanshift(data=points, r=r, c=c)
    segmented_img = post_process(labels, peaks, im, dim)
    # plotclusters3D(points, labels, peaks)
    print("---Image segmentation finished---")

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


def mean_shift_test():
    """
    Test fucntion to compare with scratch implementation
    https://www.machinecurve.com/index.php/2020/04/23/how-to-perform-mean-shift-clustering-with-python-in-scikit/
    """
    from sklearn.datasets import make_blobs
    from sklearn.cluster import MeanShift, estimate_bandwidth
    # Configuration options
    num_samples_total = 10000
    cluster_centers = [(5, 5), (3, 3), (1, 1)]
    num_classes = len(cluster_centers)
    # Generate data
    X, targets = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes,
                            center_box=(0, 1), cluster_std=0.30)
    # Estimate bandwith
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    # Fit Mean Shift with Scikit
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    # labels = ms.labels_
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    # Predict the cluster for all the samples
    P = ms.predict(X)
    # Generate scatter plot for training data
    colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426' if x == 2 else '#67c614', P))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    plt.title(f'Sklearn')

    labels, peaks = meanshift(data=X, r=bandwidth, c=2)
    colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426' if x == 2 else '#67c614', labels))
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
    plt.title(f'Scratch code')

    plt.show()
