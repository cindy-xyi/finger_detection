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
import p1n2 as hw

def clean_image(depth_img):

    # Binarize depth image
    bin_img = util.binarize(depth_img, display=False)
    bin_img[bin_img != 0] = 255

    # Erode to get rid of noise
    kernel = np.ones((10, 5))
    bin_img = cv2.erode(bin_img, kernel)

    # Ignore bottom few rows
    bin_img[220:, :] = 0

    # Median filter to get rid of some more noise.
    bin_img = ndimage.median_filter(bin_img, size=10)

    _, lab_im = cv2.connectedComponents(bin_img)


    # find areas and subtract it
    max_area = -1
    max_label = -1
    for label in range(1, np.max(np.unique(lab_im))+1):
        area = np.sum((lab_im == label).astype('uint8'))
        if area > max_area:
            max_area = area
            max_label = label
    bin_img = (lab_im == max_label).astype('uint8') * 255
    
    return bin_img


def palm_subtraction(bin_img):

    # Make erosion kernel and erode the binary image to get rid of the fingers
    kernel = np.ones((15, 20), dtype='uint8')
    image_erode = cv2.erode(bin_img, kernel)

    # Make smoothing kernel that's circular to smooth out the stump
    kernel = np.zeros((40, 40, 3), dtype='uint8')
    kernel = cv2.circle(kernel, (20, 20), 20, (255, 255, 255), -1)
    image_dilate = cv2.dilate(image_erode, kernel[:, :, 1])

    attribute_list = hw.get_attribute(image_dilate/255)
    h, w = image_dilate.shape
    radius = 25

    y = np.arange(int(attribute_list[0]['position']['x']), h)
    b = attribute_list[0]['position']['y'] - np.tan(attribute_list[0]['orientation'])*attribute_list[0]['position']['x']
    m = np.tan(attribute_list[0]['orientation'])

    for (yi) in zip(y):
        x_calc = int(np.round((yi - b)/ min(-m, m)))
        # image_dilate[yi, x_calc - radius : x_calc + radius] = 255
        image_dilate[yi, 0 : x_calc + radius] = 255
    
    # Subtract palm from hand to get fingers
    fingies = bin_img - image_dilate
    fingies[fingies != 255] = 0

    return fingies


def main(argv):
    data_dir = argv[0]
    img_num = argv[1]

    # Read in images
    print(os.getcwd())
    print(f'{data_dir}/{img_num}-color.png')
    img = cv2.imread(f'{data_dir}/{img_num}-color.png', cv2.IMREAD_COLOR)
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
