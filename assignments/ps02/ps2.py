"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def process_base_image(img, kernel_size, show_image=False):
    """
    Will take a given input image and convert the image to
    gray scale as well apply a Gaussian Blur.

    Args:
        img (numpy.array): image to convert to gray scale
        kernel_size (tuple): the size of the Gaussian Kernel
        show_image (boolean): If the image should be displayed after being processed. Used to debug.

    Returns:
        processed_image (numpy.array): image that has a Gaussian Blur applied
                                       and has been converted to gray scale.
    """
    processed_image = img.copy()
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.GaussianBlur(processed_image, kernel_size, 1)
    if show_image:
        cv2.imshow('Gray Scale Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return processed_image


def hough_circles(img, dp, min_dist, p1, p2, min_rad, max_rad):
    return cv2.HoughCircles(
        image=img,
        method=cv2.cv.CV_HOUGH_GRADIENT,
        # method=cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist*2,
        param1=p1,
        param2=p2,
        minRadius=min_rad,
        maxRadius=max_rad
    )


def pixel_color(img, x, y):
    return img[y, x]


def traffic_light_detection(img_in, radii_range):
    """

    Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    img = process_base_image(img_in, (7, 7))

    # find all the circles in an image using Hough Circles
    min_radii = min(radii_range)
    max_radii = max(radii_range)

    circles = hough_circles(img, 1.15, min_radii, 30, 15, min_radii, max_radii)
    # cleanup circles so its easier to use.
    circles = circles[0, :]
    # round the numbers of the array to uint16 values.
    circles = np.uint16(np.around(circles))

    # If there are more than 3 circles found, eliminate the outliers.
    if len(circles) > 3:
        median_x = np.median(circles[:, 0])
        min_x = median_x - 5
        max_x = median_x + 5
        circles = circles[circles[:, 0] > min_x, :]
        circles = circles[circles[:, 0] < max_x, :]

    # sort the circles from top down to allow color compare.
    circles = circles[np.argsort(circles[:, 1])]  # sort by Y direction.
    # creating some names for clarity due to x, y being col, row.

    red_row, red_col, yellow_row, yellow_col, green_row, green_col = [
        circles[0][1],
        circles[0][0],
        circles[1][1],
        circles[1][0],
        circles[2][1],
        circles[2][0],
    ]

    # determine colors.
    state = 'yellow'  # default state.
    cords = (yellow_col, yellow_row)

    red_color = np.array([0, 0, 255])
    green_color = np.array([0, 255, 0])

    if (img_in[red_row, red_col] == red_color).all():
        state = 'red'
    elif (img_in[green_row, green_col] == green_color).all():
        state = 'green'

    return cords, state


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    raise NotImplementedError




def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = process_base_image(img_in, (7, 7))
    # Assumption made that a DNE sign will always have at least a
    # radius of 5.
    min_radius = 5
    max_radius = img.shape[1]

    circles = hough_circles(img, 1, min_radius, 30, 30, min_radius, max_radius)
    circles = np.uint16(np.around(circles))

    if circles is not None:
        circles = circles[0, :]
        # since multiple circles might be found, the correct one
        circle_mid_colors = [pixel_color(img, x[0], x[1]) for x in circles]
        valid_idx = circle_mid_colors.index(255)
        the_sign = circles[valid_idx]

    output = (the_sign[0], the_sign[1])
    return output



def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


