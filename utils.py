"""
Preprocess image before mean shift algorithm

Examples:
- If features are RGB values, turn image into a matrix where each row (pixel) has three columns (R,G,B)
- For spatial position, turn each pixel into 5D vector, then, for each row (pixel) we have 5 columns (R,G,B,x,y)
"""
from config import *


def load_mat():
    """
    :return: flattened image with RGB pixel values and shape (2000, 3)
    """
    points = io.loadmat(r'data/pts.mat')["data"].reshape(-1, 3)
    return points


def load_image(path=r'data/wild.jpg'):
    """
    Work using the CIELAB color space in your code, where Euclidean distances in this space correlate
    better with color changes perceived by the human eye.
    """
    img_origin = cv2.imread(path, cv2.COLOR_BGR2RGB)
    # cluster the image data in CIELAB color space by first converting the RGB color vectors to CIELAB
    img = cv2.cvtColor(img_origin, cv2.COLOR_RGB2LAB)
    return img


def save_image(img, filename='test.jpg'):
    cv2.imwrite("renders/"+filename, img)
    print('Image saved with name:', filename)


def post_process(labels, peaks, im):
    """
    :return: segmented image
    """
    segmented_img = peaks[np.reshape(labels, im.shape[:2])]
    segmented_img = cv2.cvtColor(segmented_img.astype(np.uint8), cv2.COLOR_LAB2RGB)

    return segmented_img.astype(np.uint8)
