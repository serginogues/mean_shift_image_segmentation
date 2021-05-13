"""
Preprocess image before mean shift algorithm

Examples:
- If features are RGB values, turn image into a matrix where each row (pixel) has three columns (R,G,B)
- For spatial position, turn each pixel into 5D vector, then, for each row (pixel) we have 5 columns (R,G,B,x,y)
"""
from config import *
from visualization import *


def load_mat():
    """
    :return: flattened image with RGB pixel values and shape (2000, 3)
    """
    points = io.loadmat(r'data/pts.mat')["data"].reshape(-1, 3)
    return points


def pre_process():
    """
    Work using the CIELAB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    """
    print("---Preprocessing starts---")
    img_origin = cv2.imread(PATH)
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    if RESIZE:
        img = resize(img, int(img_origin.shape[0] / 2), int(img_origin.shape[1] / 2))
    if BLUR:
        img_blur = blur(img)
        img_post = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    else:
        img_post = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    print("---Preprocessing finished---")
    return img_post, img


def save_image(img, SAVE_NAME):
    import os
    path_ = "renders/" + IMAGE_NAME + "/"
    try:
        os.makedirs(path_, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path_)
    else:
        print("Successfully created the directory %s " % path_)
        cv2.imwrite(path_ + "/" + SAVE_NAME, img)
        print('Image saved with name:', SAVE_NAME)


def post_process(labels, peaks, im, dim):
    """
    Get segmented image from labels and peaks
    """

    if dim:
        segmented_img = peaks[np.reshape(labels, im.shape[:2])]
    else:
        peaks2 = peaks[:, :3]
        segmented_img = peaks2[np.reshape(labels, im.shape[:2])]

    segmented_img = cv2.cvtColor(segmented_img.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return segmented_img.astype(np.uint8)


def resize(im, height=220, width=220):
    dim = (width, height)
    res = cv2.resize(im, dim, interpolation=cv2.INTER_LINEAR)
    return res


def blur(im):
    """
    remove unwanted noise with Gaussian blur
    :param im:
    :return:
    """
    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    return blurred


def concatenate_images(list):
    if len(list) == 2:
        return cv2.hconcat([list[0], list[1]])
    elif len(list) == 3:
        return cv2.hconcat([list[0], list[1], list[2]])
    elif len(list) == 4:
        return cv2.hconcat([list[0], list[1], list[2], list[3]])


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def save_all_original_img(SAVE_NAME):
    list2 = []
    a = 'data/animal.jpg'
    img_origin = cv2.imread(a)
    list2 = [img_origin]
    b = 'data/bigben.jpg'
    c = 'data/girl.jpg'
    d = 'data/mask.jpg'
    list = [b, c, d]

    for im in list:
        img_origin = cv2.imread(im)
        img = resize(img_origin, int(img_origin.shape[0] / 2), int(img_origin.shape[1] / 2))
        list2.append(img)

    im = concatenate_images(list2)
    save_image(im, SAVE_NAME)


def mean_shift_test():
    """
    Test fucntion to compare with scratch implementation
    https://www.machinecurve.com/index.php/2020/04/23/how-to-perform-mean-shift-clustering-with-python-in-scikit/
    """
    from sklearn.datasets import make_blobs
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from mean_shift import meanshift
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
