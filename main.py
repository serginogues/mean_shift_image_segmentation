"""
Mean-shift Image segmentation

HOW TO USE ():
0- Open config.py
1- Set R, C and FEATURE_3D
2- Set IMAGE_NAME and booleans RESIZE and BLUR (for preprocessing)
3- Set boolean SAVE (otherwise shows image at the end)
4- Run
"""
from config import *
from utils import load_image, save_image, concatenate_images
from imageSegmentation import segmIm
from visualization import cv2, show_image

if __name__ == '__main__':

    img, img_origin = load_image()
    segmented_img, num_peaks = segmIm(img, R, C, FEATURE_3D)
    final_im = concatenate_images([cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR), img, segmented_img])

    if SAVE:
        save_image(final_im)
    else:
        show_image(final_im)


"""
EVALUATE:
- For each value of r, run your algorithm using (1) just the CIELAB color values
(i.e., a 3D feature vector), and (2) CIELAB+position values (i.e., a 5D feature vector)
- various values of r and c and show the results, while also report results in time gain.
- provide details about the effect of speeding approaches
- Can you suggest other processing steps in order to improve the segmentation
results? Explain your steps and the reasons behind them.

IMAGES:
- use small pictures, for example images containing frontal face images
- For 3 different images:
    girl - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/181091.html
    mask - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/55075.html
    bigben - https://www2.eecs.b erkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/368078.html
    
REPORT:
- 800-1500 words with plots and figures
- explain and display also intermediary steps
- describe your observations for the different parameters involved.
"""
