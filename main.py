"""
Mean-shift Image segmentation

********HOW TO USE***********
pip install -r requirements.txt
0 - Open config.py
1 - Set R, C and FEATURE_3D
2 - Set IMAGE_NAME and booleans RESIZE and BLUR (for preprocessing)
3 - Run
"""
from config import *
from utils import pre_process, save_image, concatenate_images
from imageSegmentation import segmIm
from visualization import cv2, show_image


if __name__ == '__main__':

    start_time = time.time()

    img, img_origin = pre_process()
    segmented_img, num_peaks = segmIm(img, R, C, FEATURE_3D)
    final_im = concatenate_images([cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR), img, segmented_img])

    seconds = time.time() - start_time
    print("--- %s seconds ---" % seconds)

    if SAVE:
        SAVE_NAME = IMAGE_NAME+'_r'+str(R)+'_c'+str(C)+dim_name()+blur_name()+'_'+str(round(seconds))+'sec.jpg'
        save_image(final_im, SAVE_NAME)
    else:
        show_image(final_im)
