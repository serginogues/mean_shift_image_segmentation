from scipy import io
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import cv2
import itertools
from tqdm import tqdm

# mean shift parameters
R = 50  # window radius
C = 2
FEATURE_3D = False  # FEATURE_3D=True -> 3D, FEATURE_3D=False -> 5D

# image parameters
IMAGE_NAME = 'bigben'  # image to be used from data/
BLUR = True  # blur the image during preprocessing
RESIZE = True  # resize by half the image during preprocessing
SAVE = True  # save the image at the end

PATH = 'data/'+IMAGE_NAME+'.jpg'


def blur_name():
    if BLUR:
        return '_blur'
    else:
        return '_no_blur'


def dim_name():
    if FEATURE_3D:
        return '_3D'
    else:
        return '_5D'


SAVE_NAME = IMAGE_NAME+'_r'+str(R)+'_c'+str(C)+dim_name()+blur_name()+'.jpg'

