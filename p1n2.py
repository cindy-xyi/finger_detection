#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d, gaussian_filter, maximum_filter
import matplotlib.pyplot as plt


def binarize(gray_image, thresh_val):
    """ Function to threshold grayscale image to binary
        Sets all pixels lower than threshold to 0, else 255

        Args:
        - gray_image: grayscale image as an array
        - thresh_val: threshold value to compare brightness with

        Return:
        - binary_image: the thresholded image
    """
    # TODO: 255 if intensity >= thresh_val else 0
    binary_image = np.zeros(gray_image.shape, dtype=np.uint8)
    binary_image[gray_image >= thresh_val] = 255

    return binary_image


def label(binary_image):
    """ Function to labeled components in a binary image
        Uses a sequential labeling algorithm

        Args:
        - binary_image: binary image with multiple components to label

        Return:
        - lab_im: binary image with grayscale level as label of component
    """

    _, lab_im = cv2.connectedComponents(binary_image)
    return lab_im


def get_position(image, label_):
    n_rows, n_cols = image.shape
    # Isolate object from the rest of the image
    sep_img = np.zeros_like(image)
    sep_img[image == label_] = 1

    # Calculate area of image given the label
    area = (image == label_).sum()
    # print("Area", area)

    # Calculate the Center of the Area (First Moment)
    j_vector = np.reshape(np.arange(n_cols), (-1, 1)).T
    i_vector = np.reshape(np.arange(n_rows), (-1, 1))
    # print("Label", label_)

    x_bar = (j_vector * sep_img).sum() / area
    y_bar = n_rows - (i_vector * sep_img).sum() / area
    return {"x": x_bar, "y": y_bar}


def get_orientation(image, label_, x_bar, y_bar):
    n_rows, n_cols = image.shape
    # Isolate object from the rest of the image
    sep_img = np.zeros_like(image)
    sep_img[image == label_] = 1

    # Initialize a, b, c
    a = 0
    b = 0
    c = 0
    for col in range(n_cols):
        for row in range(n_rows):
            y = (n_rows - row - 1)
            x = col
            # if pixel value is 0, don't calculate cum sum for this pixel (is zero)
            if sep_img[y, x] == 0:
                continue
            # calculate x_prime and y_prime
            x_prime = x - x_bar

            # note the y transformation from upper left hand corner (y pointing down) to bottom left corner (y pointing up)
            y_prime = -(y + y_bar)

            # Calculate a, b, c
            a += (x_prime ** 2) * sep_img[y, x]
            b += 2 * (x_prime * y_prime) * sep_img[y, x]
            # Redefine y_prime so there isn't a huge negative offset (that compounds when squared).
            # This is essentially applying the "flip"
            y_prime = row - y_bar + 1
            c += (y_prime ** 2) * sep_img[y, x]

    # Calculate theta
    theta = np.arctan2(b, a - c)
    theta /= 2
    # Return positive angle
    if theta < 0:
        theta += np.pi
    return theta, a, b, c


def get_roundedness(a, b, c, theta1):
    # Calculate theta2
    theta2 = theta1 + (np.pi / 2)
    # Calculate energies for theta1/2
    E1 = a * (np.sin(theta1) ** 2) - b * np.sin(theta1) * np.cos(theta1) + c * (np.cos(theta1) ** 2)
    E2 = a * (np.sin(theta2) ** 2) - b * np.sin(theta2) * np.cos(theta2) + c * (np.cos(theta2) ** 2)
    return E1 / E2


def get_attribute(labeled_image):
    """ Function to get the attributes of each component of the image
        Calculates the position, orientation, and roundedness

        Args:
        - labeled_image: image file with labeled components

        Return:
        - attribute_list: a list of the aforementioned attributes
    """

    # Initialize variables
    n_labels = np.max(labeled_image).astype(int) + 1
    attribute_list = []

    for i in range(1, n_labels):
        # Initialize attribute for label i:
        attribute = dict.fromkeys(("position", "orientation", "roundedness"))

        # Calculate Position of Objects
        pos_dict = get_position(labeled_image, i)
        attribute["position"] = pos_dict

        # Calculate Orientation
        theta1, a, b, c = get_orientation(labeled_image, i, pos_dict['x'], pos_dict['y'])
        attribute["orientation"] = theta1

        # Calculate Roundedness
        roundness_ = get_roundedness(a, b, c, theta1)
        attribute["roundedness"] = roundness_

        # Append to list
        attribute_list.append(attribute)
    return attribute_list


def draw_attributes(image, attribute_list):
    num_row = image.shape[0]
    attributed_image = image.copy()
    for attribute in attribute_list:
        center_x = (int)(attribute["position"]["x"])
        center_y = (int)(attribute["position"]["y"])
        slope = np.tan(attribute["orientation"])

        cv2.circle(attributed_image, (center_x, num_row - center_y), 2, (0, 255, 0), 2)
        cv2.line(
            attributed_image,
            (center_x, num_row - center_y),
            (center_x + 20, int(20 * (-slope) + num_row - center_y)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            attributed_image,
            (center_x, num_row - center_y),
            (center_x - 20, int(-20 * (-slope) + num_row - center_y)),
            (0, 255, 0),
            2,
        )

    return attributed_image


def derivative_sobel(image):
    # Sobel filters as numpy array
    s_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    s_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8

    # Apply sobel filters
    filt_x = convolve2d(image, s_x, mode="same")
    filt_y = convolve2d(image, s_y, mode="same")

    return filt_x, filt_y


def non_max_suppression(image, phase, threshold):
    M, N = image.shape
    sup_img = np.zeros((M, N), dtype=np.uint32)
    # If there is a negative phase, convert it to positive
    phase[phase < 0] += np.pi

    # Now all phases are positive. Iterate over each pixel
    for i in range(1, M - 1):  # over x
        for j in range(1, N - 1):  # over y
            q = 255
            r = 255

            # if correct angle is 0
            if (0 <= phase[i, j] < np.pi / 4) or (7 * np.pi / 8 <= phase[i, j] < np.pi):
                q = image[i, j + 1]
                r = image[i, j - 1]
            # if correct angle is 45
            elif np.pi / 4 <= phase[i, j] < 3 * np.pi / 8:
                q = image[i + 1, j - 1]
                r = image[i - 1, j + 1]
            # if correct angle is 90
            elif 3 * np.pi / 8 <= phase[i, j] < 5 * np.pi / 8:
                q = image[i - 1, j]
                r = image[i + 1, j]
            # if correct angle is 135:
            elif 5 * np.pi / 8 <= phase[i, j] < 7 * np.pi / 8:
                q = image[i - 1, j - 1]
                r = image[i + 1, j + 1]
            # Save if maximum:
            if (image[i, j] >= q) and (image[i, j] >= r):
                sup_img[i, j] = image[i, j]
            # if not maximum, supress
            else:
                sup_img[i, j] = 0
    sup_img[sup_img < threshold] = 0
    sup_img[sup_img != 0] = 255
    return sup_img


def double_threshold_hysteresis(img, high_thresh_frac, low_thresh_frac=0.05):
    '''
    :param img: image to process. Grayscale.
    :param high_thresh_frac: fraction of max brightness of grayscale image
    :param low_thresh_frac: fraction of fraction of high threshold
    :return: double_edges: returns edge image. pixel with value 255 is strong, 100 is weak, 0 is no edge.
    '''
    high_thresh = high_thresh_frac * img.max()
    low_thresh = high_thresh * low_thresh_frac

    strong_r, strong_c = np.where(img >= high_thresh)
    weak_r, weak_c = np.where((img >= low_thresh) & (img <= high_thresh))

    double_edges = np.zeros_like(img)
    double_edges[strong_r, strong_c] = 255
    # make strong edge image copy to return
    return_image = double_edges.copy()
    # Set weak edges to some value that is not 0 or 255 for identification
    double_edges[weak_r, weak_c] = 100

    # Make padded image just to make things easier to calculate
    n_rows, n_cols = double_edges.shape
    padded_double_edge = np.zeros((n_rows + 2, n_cols + 2))
    padded_double_edge[1:-1, 1:-1] = double_edges.copy()

    # Offset for the padded image Origin moves from (0,0) -> (1,1)
    strong_r_offset = strong_r + 1
    strong_c_offset = strong_c + 1

    for r, c in zip(strong_r_offset, strong_c_offset):
        # Check each strong pixel to see if neighboring values are weak.

        # indices of weak pixels connected to strong pixels
        r_pad, c_pad = np.where(padded_double_edge[r - 1:r + 2, c - 1:c + 2] == 100)

        # Convert back to unpadded indices
        r_unpad = r_pad - 1
        c_unpad = r_pad - 1

        # Convert weak edges to strong edges
        return_image[r_unpad, c_unpad] = 255
    return return_image


def detect_edges(image, sigma, threshold, low_thresh_frac=0.05):
    """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
    - edge_image (2D binary image): each location indicates whether it belongs to an edge or not
  """

    # Blur image
    blurred = gaussian_filter(image, sigma=sigma)

    # Calculate derivative of blurred image (DoG implementation)
    ix, iy = derivative_sobel(blurred)

    # Calculate Magnitude
    gradient_magnitude = np.sqrt(ix ** 2 + iy ** 2)
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    # cv2.imwrite("output/" + "test" + "_magnitude.png", gradient_magnitude)

    # Calculate Phase
    gradient_phase = np.arctan2(iy, ix)
    # cv2.imwrite("output/" + "test" + "_phase.png", gradient_phase / np.pi * 255)

    # Non-maximal suppression
    edge_image = non_max_suppression(gradient_magnitude, gradient_phase, threshold)

    # Double threshold + Hysteresis:
    edge_image = double_threshold_hysteresis(edge_image, threshold / 255)

    # Return
    edge_image = np.array(edge_image, dtype=np.uint8)
    return edge_image


def calc_d(line_point1, line_point2, point):
    # Find distance from a line to a point
    d = np.linalg.norm(np.cross(line_point2 - line_point1, line_point1 - point)) \
        / np.linalg.norm(line_point2 - line_point1)
    return d


def calculate_length(edge_image_i, rho, theta, threshold, original_image, label):
    '''
    Calculate length of one edge (parameters from rho, theta) for edge image (only has one object).
    Args:
        edge_image_i: edge image with one object in it. Line given must be from this object or length will be 0.
        rho: distance output from hough space for one line
        theta: angle output from hough space for one line
        threshold: threshold of distance away from line allowed
        original_image: original image to plot
    Returns:
        length: length of edge in pixels. Sums up the number of pixels close enough to the line defined by threshold
    '''

    # Define line parameters from HoughSpace()
    angle = theta
    distance = rho
    a = np.cos(angle)
    b = np.sin(angle)
    x0 = a * distance
    y0 = b * distance
    x1, y1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
    x2, y2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

    # Initialize Parameters
    length = 0
    y, x = np.where(edge_image_i == 255)

    # Now iterate through all edge pixels, calculating distance to them and only adding them to the length if
    # they are lower than the threshold
    for x0, y0 in zip(x, y):
        proj = calc_d(np.array([x1, y1]), np.array([x2, y2]), np.array([x0, y0]))
        if proj <= threshold:
            length += 1

    return length


def get_edge_attribute(labeled_image, edge_image):
    '''
  Function to get the attributes of each edge of the image
        Calculates the angle, distance from the origin and length in pixels
  Args:
    labeled_image: binary image with grayscale level as label of component
    edge_image (2D binary image): each location indicates whether it belongs to an edge or not

  Returns:
     attribute_list: a list of list [[dict()]]. For example, [lines1, lines2,...],
     where lines1 is a list and it contains lines for the first object of attribute_list in part 1.
     Each item of lines 1 is a line, i.e., a dictionary containing keys with angle, distance, length.
     You should associate objects in part 1 and lines in part 2 by putting the attribute lists in same order.
     Note that votes in HoughLines opencv-python is not longer available since 2015. You will need to compute the length yourself.
  '''
    # TODO
    n_labels = np.max(labeled_image) + 1
    attribute_list = []
    line_idx = 0
    for i in range(1, n_labels):
        mask = (labeled_image == i)
        edge_image_i = mask * edge_image
        object_i_lines = []
        # cv2.imwrite("output/" + f"edge_{i}" + "_edges.png", edge_image_i)

        lines = cv2.HoughLines(edge_image_i, 0.1, np.pi / 360, 15)
        # lines = cv2.HoughLines(edge_image_i, 0.45, np.pi / 180, 27)
        # lines = cv2.HoughLines(edge_image_i, 0.45, np.pi / 75, 27)

        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    edge_length = calculate_length(edge_image_i, rho, theta, threshold=0.1,
                                                   original_image=labeled_image * 100, label=line_idx)
                    line_idx += 1
                    line = {"angle": theta, "distance": rho, "length": edge_length}
                    object_i_lines.append(line)
            attribute_list.append(object_i_lines)
    return attribute_list


def draw_edge_attributes(image, attribute_list):
    attributed_image = image.copy()
    for lines in attribute_list:
        for line in lines:
            angle = (float)(line["angle"])
            distance = (float)(line["distance"])

            a = np.cos(angle)
            b = np.sin(angle)
            x0 = a * distance
            y0 = b * distance
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(
                attributed_image,
                pt1,
                pt2,
                (0, 255, 0),
                2,
            )

    return attributed_image


def get_circle_attribute(labeled_image, edge_image):
    # extra credits
    n_labels = np.max(labeled_image) + 1
    attribute_list = []

    for i in range(1, n_labels):
        mask = (labeled_image == i)
        edge_image_i = mask * edge_image
        # edge_image_i = (mask*labeled_image).astype(np.uint8)
        object_i_circles = []

        circles = cv2.HoughCircles(edge_image_i, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0,:]:
                center = {'x': circle[0], 'y': circle[1]}
                radius = circle[2]
                circle_info = {"center": center, "radius": radius}
                object_i_circles.append(circle_info)
            attribute_list.append(object_i_circles)
    return attribute_list


def draw_circle_attributes(image, attribute_list):
    attributed_image = image.copy()
    for circles in attribute_list:
        for circle_info in circles:
            center = circle_info['center']
            radius = circle_info['radius']

            cv2.circle(attributed_image, (center['x'], center['y']), radius, (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(attributed_image, (center['x'], center['y']), 2, (0, 0, 255), 3)
    return attributed_image

def main(argv):
    img_name = argv[0]
    thresh_val = int(argv[1])

    img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('output/' + img_name + "_gray.png", gray_image)

    # part 1
    binary_image = binarize(gray_image, thresh_val=thresh_val)
    cv2.imwrite('output/' + img_name + "_binary.png", binary_image)

    labeled_image = label(binary_image)
    cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)

    attribute_list = get_attribute(labeled_image)
    print('attribute list:')
    print(attribute_list)

    attributed_image = draw_attributes(img, attribute_list)
    cv2.imwrite("output/" + img_name + "_attributes.png", attributed_image)

    # part 2
    # feel free to tune hyper parameters or use double-threshold
    edge_image = detect_edges(gray_image, sigma=1, threshold=70, low_thresh_frac=0.05)
    cv2.imwrite("output/" + img_name + "_edges.png", edge_image)

    edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
    print('edge attribute list:')
    print(edge_attribute_list)

    attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
    cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)

    # # extra credits for part 2: show your circle attributes and plot circles
    circle_attribute_list = get_circle_attribute(labeled_image, edge_image)
    attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
    cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)


if __name__ == '__main__':
    main(sys.argv[1:])
# example usage:  python p1n2.py two_objects 128
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
