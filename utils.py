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


def load_image():
    """
    Work using the CIELAB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    """
    img_origin = cv2.imread(PATH)
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)
    if RESIZE:
        img = resize(img, int(img_origin.shape[0]/2), int(img_origin.shape[1]/2))
    if BLUR:
        img_blur = blur(img)
        img_post = cv2.cvtColor(img_blur, cv2.COLOR_RGB2LAB)
    else:
        img_post = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return img_post, img


def save_image(img):
    import os
    path_ = "renders/"+IMAGE_NAME+"/"
    try:
        os.makedirs(path_, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path_)
    else:
        print("Successfully created the directory %s " % path_)
        cv2.imwrite(path_+"/"+SAVE_NAME, img)
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


def convolution(img, kernel, padding=True):
    """
    Performs convolution operation given an image and a kernel
    """
    result = np.zeros_like(img)
    p_size_i = kernel.shape[0] // 2
    p_size_j = kernel.shape[1] // 2

    # init new image and contour limits (if padding, 2 extra contour pixels)
    # number rows = i_last - i_first
    # number columns = j_last - j_first
    if padding:
        # add extra
        padded_img = np.zeros((img.shape[0] + 2 * p_size_i, img.shape[1] + 2 * p_size_j))
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1
        padded_img[i_first: i_last + 1, j_first: j_last + 1] = img
    else:
        padded_img = img.copy()
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1

    # apply kernel for each window (window.shape = kernel.shape = 3x3)
    for i in range(i_first, i_last):
        for j in range(j_first, j_last):
            window = padded_img[i - p_size_i: i + p_size_i + 1, j - p_size_j: j + p_size_j + 1]
            res_pix = np.sum(window * kernel)
            result[i - p_size_i, j - p_size_j] = res_pix
    return result


def apply_filter(img, log_kernel):
    kernel_size = log_kernel.shape[0]
    res = convolution(img, log_kernel, kernel_size)
    return res


def edge_detection(img):
    """
    Zero-crossing edge detection
    :param img:
    :return:
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255
    log17 = io.loadmat('../../Assignments/mean_shift_image_segmentation/data/Log17.mat')['Log17']
    log_res = apply_filter(img_gray, log17)
    log_res_bin = (log_res >= 0).astype(int)
    zero_cross_mask = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])
    zero_cross_res = np.zeros((log_res_bin.shape[0] - 2, log_res_bin.shape[1] - 2))
    for i in range(1, log_res_bin.shape[0] - 1):
        for j in range(1, log_res_bin.shape[1] - 1):
            if log_res_bin[i, j] == 1:
                tmp_window = log_res_bin[i - 1: i + 2, j - 1: j + 2]
                value = np.sum(tmp_window * zero_cross_mask)
                # print(value)
                if value < 9:
                    zero_cross_res[i - 1, j - 1] = 1

    """fig = plt.imshow(zero_cross_res, cmap='gray')
    plt.title('Edge image computed using the zero-crossing approach')
    plt.show()"""
    return zero_cross_res


def img_to_binary(img, threshold):
    """
    Converts image to binary.
    :param: img: array-like
    :param: threshold: int - threshold for binary image
    """
    # first convert to grayscale if image is 3-channel RGB
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    bin_img = (gray >= threshold).astype(int)
    return bin_img


def histogram(img, title='Image histogram'):
    """
    :param img: Image in Gray scale
    """
    plt.hist(img.ravel(), bins=256)
    plt.xlabel('pixel value')
    plt.ylabel('absolute frequency')
    plt.title(title)
    plt.show()


def concatenate_images(list):
    return cv2.hconcat([list[0], list[1], list[2]])


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
