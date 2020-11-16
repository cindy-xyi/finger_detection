#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d, gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
from scipy import ndimage
import util
import os

def clean_image(depth_img):

    # Binarize depth image
    bin_img = util.binarize(depth_img, display=False)
    bin_img[bin_img != 0] = 255

    # Erode to get rid of noise
    kernel = np.ones((10, 5))
    experiment = cv2.erode(bin_img, kernel)

    # Ignore bottom few rows
    bin_img[220:, :] = 0

    # Median filter to get rid of some more noise.
    bin_img = ndimage.median_filter(bin_img, size=10)

    return bin_img


def palm_subtraction(bin_img):

    # Make erosion kernel and erode the binary image to get rid of the fingers
    kernel = np.ones((15, 20), dtype='uint8')
    image_erode = cv2.erode(bin_img, kernel)

    # Make smoothing kernel that's circular to smooth out the stump
    kernel = np.zeros((40, 40, 3), dtype='uint8')
    kernel = cv2.circle(kernel, (20, 20), 20, (255, 255, 255), -1)
    image_dilate = cv2.dilate(image_erode, kernel[:, :, 1])

    # Subtract palm from hand to get fingers
    fingies = bin_img - image_dilate
    fingies[fingies != 255] = 0

    return fingies


def main(argv):
    data_dir = argv[0]
    img_num = argv[1]

    # Read in images
    print(os.getcwd())
    img = cv2.imread(f'./{data_dir}/{img_num}-color.png', cv2.IMREAD_COLOR)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    depth_img = util.read_depth_map(f'./{data_dir}/{img_num}-depth.bin')

    # Clean depth image of noise
    bin_img = clean_image(depth_img)

    # Do palm subtraction
    fingies = palm_subtraction(bin_img)

    # Count number of fingers
    _, lab_img = cv2.connectedComponents(fingies)
    numFingies = np.max(lab_img)

    # Scale so output image looks good
    lab_img = lab_img.astype(np.float32) * 255 / numFingies

    # Write and output number of fingers counted
    cv2.imwrite(f"output/{img_num}-has-{numFingies}-fingers.png", lab_img)
    print(f'number of fingers is {numFingies}')


if __name__ == '__main__':
    main(sys.argv[1:])
# example usage:  python palm_subtraction.py data/acquisitions/S1/G2 1

# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
