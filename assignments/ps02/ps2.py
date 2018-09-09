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
    processed_image = cv2.GaussianBlur(processed_image, kernel_size, 0)
    if show_image:
        cv2.imshow('Gray Scale Image', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return processed_image


def hough_circles(img, dp, min_dist, p1, p2, min_rad, max_rad):
    return cv2.HoughCircles(
        image=img,
        # method=cv2.cv.CV_HOUGH_GRADIENT,
        method=cv2.HOUGH_GRADIENT,
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

    # if there are still more than 3 circles, that means there is a circle directly above
    # or below the actual traffic light. Need to group them into possible sets of 3. First sort by the
    # y Direction, then compare each row with the one below it and store the difference in Y. The three
    # lights of interest will be the ones that are closest to each other.
    if len(circles) > 3:
        sorted_circles = circles[np.lexsort((circles[:, 2], circles[:, 0], circles[:, 1]))]
        t_circles = np.int16(sorted_circles)
        y_diffs = np.abs(np.diff(t_circles[:, 1]))
        small_diff_idx = y_diffs.argsort()[:2] # get the 2 smallest index values.
        # if the smallest diff index is 0, get 0,1,2. if not, get smallest index + 2 rows.
        smallest_idx = min(small_diff_idx)
        circles = sorted_circles[smallest_idx: smallest_idx+3, :]

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

    img = img_in.copy()

    red_color_map = img[:, :, 2]

    cv2.imshow('Red color map', red_color_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    canny_edges = cv2.Canny(red_color_map, threshold1=120, threshold2=90, apertureSize=5)
    cv2.imshow('Canny Map', canny_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # since stop signs are octogons, we want to use an internal angle of 135.
    theta = 45
    minLineLength = 25
    maxPixelGap = 10
    hLines = cv2.HoughLinesP(image=canny_edges,
                             rho=5,
                             theta=np.pi/180*theta,
                             threshold=20,
                             minLineLength=minLineLength,
                             maxLineGap=maxPixelGap
                             )

    # to capture the top and bottom of the stop sign,
    vl1 = hLines[:, 0, 1]
    vl2 = hLines[:, 0, 3]
    all_vl = np.concatenate([vl1, vl2])
    top_y_cord = max(all_vl)
    bot_y_cord = min(all_vl)
    # to capture right side / left side.
    hl1 = hLines[:, 0, 0]
    hl2 = hLines[:, 0, 2]
    all_hl = np.concatenate([hl1, hl2])
    left_x_cord = min(all_hl)
    right_x_cord = max(all_hl)

    print 'Max cords of sign are as follows'
    print 'Highest Y: {}'.format(top_y_cord)
    print 'Lowest Y: {}'.format(bot_y_cord)
    print 'Left X: {}'.format(left_x_cord)
    print 'Right X: {}'.format(right_x_cord)

    # remove lines that aren't bounding lines.
    hLines = hLines[ hlines[0,0],:,]


    # loop over lines and place them on a new image to test.
    imgc = img.copy()
    for l in hLines:
        v = l[0]
        y_cords = (v[1], v[3])
        x_cords = (v[0], v[2])

        # Only care about "bounding lines" these are lines within a threshold of

        valid_line = True
        print 'line x1: {}, y1:{} , x2: {} , y2: {} '.format(x_cords[0], y_cords[0], x_cords[1], y_cords[1])
        x_diff = x_cords[1] - x_cords[0]

        if x_diff > 0:
            y_diff = v[3] - v[1]
            slope = np.floor(y_diff / x_diff)
            print 'Slope: {}'.format(slope)
            # do not print the line if its not within 10 of the top / bottom pixels.
            if slope == 0 and :
                valid_line = False
                print 'Ignoring a line'
        if valid_line:
            cv2.line(imgc, (v[0], v[1]), (v[2], v[3]), (255, 122, 122), 5)

    cv2.imshow('Hough Lines Map', imgc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    img_in = img_in.copy()
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


