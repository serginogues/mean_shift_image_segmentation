"""
Preprocess image before mean shift algorithm

Examples:
- If features are RGB values, turn image into a matrix where each row (pixel) has three columns (R,G,B)
- For spatial position, turn each pixel into 5D vector, then, for each row (pixel) we have 5 columns (R,G,B,x,y)
"""
from config import *


def img_to_gray_hsv(image_path='lab1a.png'):
    img = cv2.imread(image_path)

    # image from BGR, which is the default in cv2, to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('Size of the color image: {}'.format(img.shape))

    # d) image to gray values
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print('Size of the grayscale image: {}'.format(img_gray.shape))

    # image to HSV space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img, img_gray, img_hsv


def load_data(path=r'data/pts.mat'):
    mat = io.loadmat(path)
    data = np.array(mat['data'])
    return data
