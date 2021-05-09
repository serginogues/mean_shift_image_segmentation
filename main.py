"""
Mean-shift Image segmentation

Steps:
1. Read the image and apply the preprocessing steps suggested above.
2. Implement the Mean-shift algorithm, considering the suggested optimizations.
3. Apply the algorithm on the image features, transform the result back to an image and visualize the obtained segmentation.
4. Test different parameters, such as r, c, feature types
"""
from preprocess import load_image, load_mat
from imageSegmentation import segmIm

if __name__ == '__main__':

    r = 7
    # points = load_mat()
    img, points = load_image('data/wild.jpg')
    segmIm(points, r, img)
