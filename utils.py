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


def concatenate_images(list):
    if len(list) == 2:
        return cv2.hconcat([list[0], list[1]])
    else:
        return cv2.hconcat([list[0], list[1], list[2]])


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)
