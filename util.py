import numpy as np
import matplotlib.pyplot as plt
import cv2

def binarize(im, threshold=50, kernel=np.ones((4,4)), display=False):
    '''
    args:
        im: input image (any scale, any type) as grayscale numpy array.
        threshold: threshold value for simple thresholding
        kernel: kernel for eroding and dilating operations to remove noise
        display: boolean variable to designate whether or not result is displayed or not. 
    output: 
        im: binarize image
    '''
    im = im/im.max() * 255
    im = im.astype(np.uint8)
    im[im > 50] = 0
    im = cv2.erode(im, kernel)
    im = cv2.dilate(im, kernel)
    
    if display is True:
        plt.imshow(im)
        plt.colorbar()
        plt.show()
    return im

def read_depth_map(file_path, h=240, w=320):
    with open(file_path, mode='rb') as f:
        im = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)
    return im