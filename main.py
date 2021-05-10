"""
Mean-shift Image segmentation

Experiments:
- For different values of r and c
- For 3 different images:
    girl - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/181091.html
    mask - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/55075.html
    bigben - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300/html/dataset/images/color/368078.html
- For each value of r, run your algorithm using (1) just the CIELAB color values
(i.e., a 3D feature vector), and (2) CIELAB+position values (i.e., a 5D feature vector)

Report:
Can you suggest other processing steps in order to improve the segmentation
results? Explain your steps and the reasons behind them.
"""
from utils import load_image, save_image, edge_detection
from imageSegmentation import segmIm
from visualization import *

r = 5
c = 2

if __name__ == '__main__':

    image_name = 'girl'
    img, img_origin = load_image('data/'+image_name+'.jpg')
    segmented_img, num_peaks = segmIm(img, r, c, False)
    final_im = plot_all([cv2.cvtColor(img_origin, cv2.COLOR_RGB2BGR), img, segmented_img])
    save_image(final_im, image_name+'_r'+str(r)+'_c'+str(c)+'_'+str(num_peaks)+'peaks.jpg')
