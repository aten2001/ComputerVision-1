"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def display_img(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
        display_img(processed_image, 'Gray Scale Image')
    return processed_image


def hough_circles(img, dp, min_dist, p1, p2, min_rad, max_rad):
    return cv2.HoughCircles(
        image=img,
        method=cv2.cv.CV_HOUGH_GRADIENT,
        # method=cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=p1,
        param2=p2,
        minRadius=min_rad,
        maxRadius=max_rad
    )


def pixel_color(img, x, y):
    return img[y, x]


def calculate_line_length(x1, y1, x2, y2):
    """ Calculates the length of a line given two points

    Args:
        x1: x cord of start point
        y1: y cord of start point
        x2: x cord of end point
        y2: y cord of end point.

    Returns:
        Integer representing the length of the line.

    """
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def calculate_slope(x1, y1, x2, y2):
    return abs((y2 - y1) / (x2 - x1))


def circles_touching(x1, y1, x2, y2, r1, r2):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # https://www.geeksforgeeks.org/check-two-given-circles-touch-intersect/
    dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    rad_sum_sq = (r1 + r2) * (r1 + r2)
    if dist_sq == rad_sum_sq:
        return 5
    elif dist_sq > rad_sum_sq:
        return 0
    else:
        return 5


def circle_group_deviations(circle_group):
    # x is most important so scale that by a factor of 2.
    x_cord = np.std(circle_group[:, 0]) * 2
    y_cord = np.std(circle_group[:, 1])
    r = np.std(circle_group[:, 2]) * 1.5

    # if there is overlap, we know it isn't a solid grouping of traffic lights, add a value to indicate this
    c1_x, c1_y, c1_r = circle_group[0, :]
    c2_x, c2_y, c2_r = circle_group[1, :]
    c3_x, c3_y, c3_r = circle_group[2, :]
    # circle 1 to circle 2
    c1_c2 = circles_touching(c1_x, c1_y, c2_x, c2_y, c1_r, c2_r)
    # circle 1 to circle 3
    c1_c3 = circles_touching(c1_x, c1_y, c3_x, c3_y, c1_r, c3_r)
    # circle 2 to circle 3
    c2_c3 = circles_touching(c2_x, c2_y, c3_x, c3_y, c2_r, c3_r)

    return x_cord + y_cord + r + c1_c2 + c1_c3 + c2_c3


def remove_duplicates(lines, dist=5):
    lines = lines.tolist()
    for idx, (x1, y1, x2, y2) in enumerate(lines):
        for idx2, (x3, y3, x4, y4) in enumerate(lines):
            if idx != idx2 and y1-dist <= y3 <= y1+dist and y2-dist <= y4 <= y2+dist and x1-dist <= x3 <= x1+dist and \
                    x2-dist <= x4 <= x2+dist:
                del lines[idx2]

    return np.array(lines)


def calculate_min_max_values(lines):
    vl1 = lines[:, 1]
    vl2 = lines[:, 3]
    all_vl = np.concatenate([vl1, vl2])
    max_y = max(all_vl)
    min_y = min(all_vl)
    # to capture right side / left side.
    hl1 = lines[:, 0]
    hl2 = lines[:, 2]
    all_hl = np.concatenate([hl1, hl2])
    min_x = min(all_hl)
    max_x = max(all_hl)
    return min_x, max_x, min_y, max_y


def red_masking(img, stop_mask=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower mask
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
    # display_img(red_mask, 'The Red Mask')
    if not stop_mask:
        # make another red mask for border removal of stop sign.
        lower_red = np.array([160, 100, 100])
        upper_red = np.array([179, 255, 255])
        red_mask_2 = cv2.inRange(hsv_img, lower_red, upper_red)
        red_mask = red_mask - red_mask_2

    red = cv2.bitwise_and(img, img, mask=red_mask)
    zero = 0
    h, s, v = red[:, :, 0], red[:, :, 1], red[:, :, 2]
    # the yield sign has a red of 204, stop is 255
    if stop_mask:
        mask = (0 <= h) & (0 <= s) & (215 < v)
    else:
        mask = (0 <= h) & (0 <= s) & (215 >= v)

    red[:, :, :3][mask] = [zero, zero, zero]
    return red


def yellow_masking(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # values for the mask were obtained from Stack Overflow which lead to this blog
    # http://aishack.in/tutorials/tracking-colored-objects-opencv/
    lower_yellow = np.array([20, 100, 100])  # <- higher than orange
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    # display_img(yellow_mask, 'Yellow Mask')
    yellow = cv2.bitwise_and(img, img, mask=yellow_mask)
    # want to change all [0, 128, 128] to 0 values to remove greens
    max = 200
    zero = 0
    h, s, v = yellow[:, :, 0], yellow[:, :, 1], yellow[:, :, 2]
    # mask all values that are [ < 0, >=max, <= max] to zero. removes yellow noise
    mask = (zero <= h) & (max >= s) & (max >= v)
    yellow[:, :, :3][mask] = [zero, zero, zero]
    # display_img(yellow, ' Yellow Done')
    return yellow


def orange_masking(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # values for range were found by taking unique values of HSV'
    # print np.unique(hsv_image.reshape(-1, hsv_image.shape[2]), axis=0)
    # [[  0   0   0]
    # [  0   0 128]
    # [ 15 255 255] <- this is the color we want.
    # [ 30 255 255] <- this is the yellow suns, ignore
    # [ 60 255 204]
    # [105 153 255]]
    lower_orange = np.array([0, 175, 175])
    upper_orange = np.array([20, 255, 255])
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    orange = cv2.bitwise_and(img, img, mask=orange_mask)
    # [ 0 0 204 ] red to remove
    # [ 0 0 255 ] another red to remove
    # [0 128 255] orange we want
    # remove the red masks [0,0,255]
    red = 255
    red_2 = 175
    zero = 0
    o = 120
    h, s, v = orange[:, :, 0], orange[:, :, 1], orange[:, :, 2]
    # set values that are [ > 0, < orange, > red] to zero.
    mask = (zero <= h) & (o >= s) & (red_2 <= v)
    # mask = (zero == h) & ( zero == s) & (red_2 <= v)
    orange[:, :, :3][mask] = [zero, zero, zero]
    return orange


def traffic_light_detection(img_in, radii_range, noisy_image=False, max_x_offset=5):
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
        noisy_image (bool): If true, tweaks the threshold for circle detection
        max_x_offset (number): The Max difference allowed in x offset

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
    # the distance between the circles should be the smallest possible circles that can touch.
    min_dist = min_radii * 2 + 10

    # img, dp,  min_dist, param1, param2, minRad, maxRad
    if noisy_image:
        circles = hough_circles(img, 1.55, min_dist, 20, 15, min_radii, max_radii)
    else:
        circles = hough_circles(img, 1.15, min_dist, 30, 20, min_radii, max_radii)

    if circles is None:
        return (0, 0), None
    else:
        # cleanup circles so its easier to use.
        circles = circles[0, :]
        # round the numbers of the array to uint16 values.
        circles = np.uint16(np.around(circles))

    if len(circles) < 3:
        return (0, 0), None
    else:  # If there are more than 3 circles found, eliminate the outliers that shouldn't be detected.
        # sort the circles first by x, then by Radius value, then by Y value.
        circles = sorted(circles, key=lambda c: (c[0], c[2], c[1]))

        # since the traffic lights will be a group of 3 circles with a similar radius, then x value, then somewhat close
        # in y value, use a "window" type of sliding group to create groups of 3 circles that can then be compared
        # to each other to see if they would make up circles of a traffic light.
        circle_groups = []
        for c_idx in range(len(circles) - 2):
            circle_group = circles[c_idx: c_idx + 3]  # build the group
            circle_groups.append(circle_group)

        circle_groups = np.array(circle_groups)
        # for each circle group found, need to figure out the group with the lowest overall standard deviation.
        # for each group, calculate the std deviations.
        group_deviations = np.array([circle_group_deviations(g) for g in circle_groups])

        most_similar_idx = np.argmin(group_deviations)
        final_circles = circle_groups[most_similar_idx]

        # if the circles aren't close to each other in the X direction, return
        # none since its not a traffic light.
        x_diffs = np.diff(final_circles[:, 0])
        if np.any(x_diffs >= max_x_offset):
            return (None, None), None

        # sort the circles from top down to allow color compare.
        circles = final_circles[np.argsort(final_circles[:, 1])]  # sort by Y direction.
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
    img = img_in.copy()
    red_color_map = red_masking(img)

    # kernel = np.ones((7, 7), np.uint8)
    # red_color_map = cv2.filter2D(red_color_map, -1, kernel)
    # red_color_map = cv2.erode(red_color_map, np.ones((5, 5)))
    red_color_map = cv2.dilate(red_color_map, np.ones((5, 5)))
    canny_edges = cv2.Canny(red_color_map, threshold1=50, threshold2=250, apertureSize=5)

    min_line_length = 20
    max_pixel_gap = 5
    hough_lines = cv2.HoughLinesP(image=canny_edges,
                                  rho=.5,
                                  theta=np.pi / 180,
                                  threshold=25,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_pixel_gap
                                  )
    if hough_lines is None:
        return None, None

    hough_lines = hough_lines[0, :]
    lines = remove_duplicates(hough_lines, dist=10)

    mid_x = None
    mid_y = None

    if len(lines) >= 6:
        # Fuzzy Logic here.
        # find the line that has the longest length and has a slope of 1. This line will
        # give us the vertex point we are interested in.
        line_slopes = np.array([calculate_slope(x[0], x[1], x[2], x[3]) for x in lines])
        line_lengths = np.array([calculate_line_length(x[0], x[1], x[2], x[3]) for x in lines])
        # the lines with a slope of 2 are the ones that will be found for this sign. Get these lines and store them.
        sloped_idx = np.where(line_slopes == 2)[0]
        # loop over the valid slops and figure out max lenght
        max_length = 0
        for sid in sloped_idx:
            idx_length = line_lengths[sid]
            if idx_length > max_length:
                max_length = idx_length

        v_idx = np.where(line_lengths == max_length)[0][0]
        vertex_line = lines[v_idx]

        # once the vertex line is found, all lines that are not within the length from the x1 can be ignore.
        max_x_1 = np.int(vertex_line[0] + max_length)
        min_x_1 = np.int(vertex_line[0] - max_length)
        # from lines delete all lines where x1 < min_x_1 and all lines where x1 > max_x_1
        l_list = lines.tolist()
        for idx, line in enumerate(l_list):
            if line[0] < min_x_1:
                del l_list[idx]
            elif line[0] > max_x_1:
                del l_list[idx]
        f_line = np.array(l_list)

        if len(f_line) < 6:
            return None, None  # not a valid sign

        mid_x = vertex_line[0]
        x_point_2 = vertex_line[2]
        y_point = vertex_line[3]
        t = abs(x_point_2 - mid_x) * 2
        top_x_1, top_x_2 = [
            np.int(x_point_2 - t),
            np.int(x_point_2)
        ]
        mid_y = np.int(y_point + ((top_x_2 - top_x_1) * np.cos(np.pi / 6) -
                                  ((top_x_2 - top_x_1) / 2) / np.cos(np.pi / 6)))

    return mid_x, mid_y


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """

    img = img_in.copy()
    red_color_map = red_masking(img, stop_mask=True)
    red_color_map = cv2.dilate(red_color_map, np.ones((5, 5)))
    canny_edges = cv2.Canny(red_color_map, threshold1=50, threshold2=250, apertureSize=5)

    # display_img(canny_edges, 'Stop Sign Canny Lines')

    min_line_length = 10
    max_pixel_gap = 10
    hough_lines = cv2.HoughLinesP(image=canny_edges,
                                  rho=0.5,
                                  theta=np.pi/180,
                                  threshold=20,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_pixel_gap
                                  )
    if hough_lines is None:
        return None, None
    hough_lines = hough_lines[0, :]  # cleanup dimensionality to make it easier to work with.
    lines = remove_duplicates(hough_lines)

    if len(lines) < 8:
        return None, None

    # once given the lines of interest perform some more calculations
    min_x, max_x, min_y, max_y = calculate_min_max_values(lines)
    mid_x = min_x + ((max_x - min_x) / 2)
    mid_y = min_y + ((max_y - min_y) / 2)
    return mid_x, mid_y


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in.copy()
    yellow_color_map = yellow_masking(img)
    yellow_color_map = cv2.dilate(yellow_color_map, np.ones((5, 5)))
    canny_edges = cv2.Canny(yellow_color_map, threshold1=50, threshold2=250, apertureSize=5)
    #Hough lines.
    min_line_length = 30
    max_pixel_gap = 75

    hough_lines = cv2.HoughLinesP(image=canny_edges,
                                  rho=0.5,
                                  theta=np.pi/180,
                                  threshold=30,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_pixel_gap)

    if hough_lines is None:
        return None, None

    hough_lines = hough_lines[0, :]
    lines = remove_duplicates(hough_lines)
    if lines.shape[1] != 4:  # this wasn't a diamond, return.
        return None, None
    min_x, max_x, min_y, max_y = calculate_min_max_values(lines)
    mid_x = min_x + ((max_x - min_x) / 2)
    mid_y = min_y + ((max_y - min_y) / 2)

    # print 'Warning mid at ({}, {})'.format(mid_x, mid_y)
    return mid_x, mid_y


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    img = img_in.copy()
    orange_color_map = orange_masking(img)
    orange_color_map = cv2.dilate(orange_color_map, np.ones((5, 5)))
    canny_edges = cv2.Canny(orange_color_map, threshold1=50, threshold2=250, apertureSize=5)

    min_line_length = 20
    max_pixel_gap = 20
    hough_lines = cv2.HoughLinesP(image=canny_edges,
                                  rho=0.5,
                                  theta=np.pi/180,
                                  threshold=20,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_pixel_gap
                                  )
    if hough_lines is None:
        return None, None

    hough_lines = hough_lines[0, :]
    lines = remove_duplicates(hough_lines)
    if lines.shape[1] < 4:
        return None, None

    min_x, max_x, min_y, max_y = calculate_min_max_values(lines)
    mid_x = min_x + ((max_x - min_x) / 2)
    mid_y = min_y + ((max_y - min_y) / 2)

    return mid_x, mid_y


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
        valid_idx = circle_mid_colors.index(np.max(circle_mid_colors))
        the_sign = circles[valid_idx]
    else:
        return None, None

    output = (the_sign[0], the_sign[1])
    return output


def traffic_sign_detection(img_in, light_size=(3, 30), light_offset = 5):
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
        light_size (tuple): The min and max radii for traffic light detection
        light_offset(number): The value that traffic lights can be offset in the x direction to be valid.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    img = img_in.copy()
    dict = {} # return dictionary

    # traffic lights
    (x, y), state = traffic_light_detection(img, light_size, noisy_image=True, max_x_offset=light_offset)
    if state is not None:
        dict['traffic_light'] = (x, y)

    # orange signs ( construction )
    (x, y) = construction_sign_detection(img)
    if x is not None:
        dict['construction'] = (x, y)

    # yellow signs (warning)
    (x, y) = warning_sign_detection(img)
    if x is not None:
        dict['warning'] = (x, y)

    # stop sign
    (x, y) = stop_sign_detection(img)
    if x is not None:
        dict['stop'] = (x, y)

    # yield sign
    (x, y) = yield_sign_detection(img)
    if x is not None:
        dict['yield'] = (x, y)

    # # dne sign
    (x, y) = do_not_enter_sign_detection(img)
    if x is not None:
        dict['no_entry'] = (x, y)

    return dict


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
    img = img_in.copy()
    clean_picture = cv2.fastNlMeansDenoisingColored(
        src=img,
        dst=None,
        templateWindowSize=7,
        searchWindowSize=21,
        h=15,
        hColor=15
    )
    clean_picture = cv2.bilateralFilter(clean_picture, 9, 75, 75)
    return traffic_sign_detection(clean_picture, light_size=(8, 30), light_offset=10)


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


