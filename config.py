from scipy import io
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
from tqdm import tqdm
from numba import njit
import time

# mean shift parameters
R = 40    # window radius
C = 5
FEATURE_3D = True   # FEATURE_3D=True -> 3D, FEATURE_3D=False -> 5D

# image parameters
IMAGE_NAME = 'girl'  # image to be used from data/
BLUR = False  # blur the image during preprocessing
RESIZE = False  # resize by half the image during preprocessing

SAVE = False  # save the image at the end
PATH = 'data/'+IMAGE_NAME+'.jpg'


def blur_name():
    if BLUR:
        return '_Blur'
    else:
        return '_noBlur'


def dim_name():
    if FEATURE_3D:
        return '_3D'
    else:
        return '_5D'

