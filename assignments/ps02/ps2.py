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


def remove_duplicates(lines, dist=5):
    lines = lines.tolist()
    for idx, (x1, y1, x2, y2) in enumerate(lines):
        for idx2, (x3, y3, x4, y4) in enumerate(lines):
            if idx != idx2 and y1-dist <= y3 <= y1+dist and y2-dist <= y4 <= y2+dist and x1-dist <= x3 <= x1+dist and \
                    x2-dist <= x4 <= x2+dist:
                del lines[idx2]

    return np.array(lines)


def clean_edge_lines(lines, min_x, max_x, min_y, max_y):
    # to clean up the lines we only want ones that are within a certain threshold of being at the edges.
    # the lines should have a x1 or x2 within 3 of min_x and max_x
    # should have a y1 or y2 within 3 of min_y and max_y
    valid_lines = []
    for l in lines:
        left_diff = np.abs(l[0]-min_x)
        right_diff = np.abs(l[2] - max_x)
        yt1 = np.abs(l[1]-max_y)
        yt2 = np.abs(l[3]-max_y)
        yb1 = np.abs(l[1]-min_y)
        yb2 = np.abs(l[3]-min_y)
        if (yt1 < 5 and yt2 < 5) or (yb1 < 5 and yb2 < 5):
            valid_lines.append(l)
        elif left_diff < 5 or right_diff < 5:
            valid_lines.append(l)
    # remove duplicate lines once we removed ones that are not on the border.
    valid_lines = remove_duplicates(valid_lines)
    valid_lines = np.array(valid_lines)
    return valid_lines


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
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
    red = cv2.bitwise_and(img, img, mask=red_mask)
    # print 'unique red map values'
    # print np.unique(red.reshape(-1, red.shape[2]), axis=0)
    # want to change all [0, 128, 128] to 0 values to remove greens
    max = 204
    if stop_mask:
        max = 255
    zero = 0
    h, s, v = red[:, :, 0], red[:, :, 1], red[:, :, 2]
    mask = (zero == h) & (zero == s) & (max == v)
    red[:, :, :3][mask] = [zero, zero, zero]

    return red


def yellow_masking(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # values for the mask were obtained from Stack Overflow which lead to this blog
    # http://aishack.in/tutorials/tracking-colored-objects-opencv/
    lower_yellow = np.array([20, 100, 100])  # <- higher than orange
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    yellow = cv2.bitwise_and(img, img, mask=yellow_mask)
    # want to change all [0, 128, 128] to 0 values to remove greens
    max = 128
    zero = 0
    h, s, v = yellow[:, :, 0], yellow[:, :, 1], yellow[:, :, 2]
    mask = (zero == h) & (max == s) & (max == v)
    yellow[:, :, :3][mask] = [zero, zero, zero]

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

    # cv2.imshow('Orange Mask', orange_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    orange = cv2.bitwise_and(img, img, mask=orange_mask)

    # print np.unique(orange.reshape(-1, orange.shape[2]), axis=0)
    #
    # h_values = np.unique(orange[:,:, 0])
    # s_values = np.unique(orange[:, :, 1])
    # v_values = np.unique(orange[:, :, 2])
    # print 'HValues'
    # print h_values
    #
    # print 'SValues'
    # print s_values
    #
    # print 'VValues'
    # print v_values

    # [ 0 0 204 ] red to remove
    # [ 0 0 255 ] another red to remove
    # [0 128 255] orange we want

    # cv2.imshow('Orange BW', orange)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # remove the red masks [0,0,255]
    red = 255

    red_2 = 175
    zero = 0
    #o = 125
    o = 120
    h, s, v = orange[:, :, 0], orange[:, :, 1], orange[:, :, 2]
    mask = (zero <= h) & (o >= s) & (red_2 <= v)
    # mask = (zero == h) & ( zero == s) & (red_2 <= v)
    orange[:, :, :3][mask] = [zero, zero, zero]

    # cv2.imshow('Orange Things', orange)
    # cv2.waitKey(0)

    return orange


def traffic_light_detection(img_in, radii_range, other_signs=False):
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

    if circles is None:
        # print 'No Hough Circles Found in Traffic Lights'
        return (0, 0), None
    # else:
    #     cimg = img.copy()
    #     for i in circles:
    #         # draw the outer circle
    #         cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    #         # draw the center of the circle
    #         cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #     cv2.imshow('Traffic Circles Image', cimg)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


    # If there are more than 3 circles found, eliminate the outliers.
    if len(circles) > 3:
        median_x = np.median(circles[:, 0])
        if other_signs:
            counts = np.bincount(circles[:, 0])
            mode_x = np.argmax(counts)
            if np.count_nonzero(circles[:, 0] == mode_x) >= 3:
                median_x = mode_x

        min_x = median_x - 5
        max_x = median_x + 5
        circles = circles[circles[:, 0] > min_x, :]
        circles = circles[circles[:, 0] < max_x, :]

    # cimg = img.copy()
    # for i in circles:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('Traffic Circles Image CLean 1', cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # if there are still more than 3 circles, that means there is a circle directly above
    # or below the actual traffic light. Need to group them into possible sets of 3. First sort by the
    # y Direction, then compare each row with the one below it and store the difference in Y. The three
    # lights of interest will be the ones that are closest to each other.
    if len(circles) >= 3:
        sorted_circles = circles[np.lexsort((circles[:, 2], circles[:, 0], circles[:, 1]))]

        radius_counts = np.bincount(circles[:, 2])
        mode_r = np.argmax(radius_counts)
        diff_from_mode = np.array([abs(x[2] - mode_r) for x in sorted_circles])
        # get the 3 idx of the ones closest to the mode.
        small_diff_idx = np.array(diff_from_mode.argsort()[:3])
        # remove ones with radius too large.
        sorted_circles = sorted_circles[small_diff_idx, :]
        t_circles = np.int16(sorted_circles)
        y_diffs = np.abs(np.diff(t_circles[:, 1]))

        small_diff_idx = y_diffs.argsort()[:2] # get the 2 smallest index values.
        # if the smallest diff index is 0, get 0,1,2. if not, get smallest index + 2 rows.
        smallest_idx = min(small_diff_idx)
        circles = sorted_circles[smallest_idx: smallest_idx+3, :]
        # t_circles = np.int16(circles[:, :])
        #
        # print t_circles
        #
        # rad_differences = np.diff(t_circles[:, 2])
        # print 'rad differences is'
        # print rad_differences


    # cimg = img.copy()
    # for i in circles:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow('Traffic Circles Image CLean 2', cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    if len(circles) < 3:
        return (0, 0), None

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

    # print 'Traffic Light found at: {} and it is: {}'.format(cords, state)
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
    red_color_map = cv2.dilate(red_color_map, np.ones((5, 5)))
    canny_edges = cv2.Canny(red_color_map, threshold1=50, threshold2=250, apertureSize=5)
    #
    # cv2.imshow('Canny Map', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    min_line_length = 20
    max_pixel_gap = 5
    hough_lines = cv2.HoughLinesP(image=canny_edges,
                                  rho=0.5,
                                  theta=np.pi / 180,
                                  threshold=25,
                                  minLineLength=min_line_length,
                                  maxLineGap=max_pixel_gap
                                  )
    if hough_lines is None:
        return None, None
    hough_lines = hough_lines[0, :]

    # print hough_lines
    lines = remove_duplicates(hough_lines, dist=10)


    # print 'Yield Line Count: {} '.format(len(lines))

    if len(lines) < 6:
        return None, None

    if len(lines) > 6:
        # need to remove lines that aren't close to the other ones. This is accomplished
        # by calculating the mean X1 cords of all the lines. remove one line at a time till only 6
        # are found.
        line_list = lines[:, 0].tolist()
        # print line_list
        while len(line_list) > 6:
            l_l = lines.tolist()
            x1_mean = np.mean(line_list)
            mean_diffs = [abs(x - x1_mean) for x in line_list]
            max_err = np.max(mean_diffs)
            max_loc = mean_diffs.index(max_err)
            # print 'remove from array at loc of: {}'.format(max_loc)
            del l_l[max_loc]
            lines = np.array(l_l)
            line_list = lines[:, 0].tolist()

    # imgc = img.copy()
    # for line in lines:
    #     cv2.line(imgc, (line[0], line[1]), (line[2], line[3]), (255, 122, 122), 2)
    # cv2.imshow('Hough Lines Map', imgc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # determine midpoint.
    # midpoint x will be at teh midpoint of the longest line (x2 -x1) that has the same Y values.
    sorted_lines = lines[np.argsort(lines[:, 0])]  # sort by x1 direction.
    for l in sorted_lines:
        if np.abs(l[1] - l[3]) < 5: # the line is pretty straight, good enough.
            mid_x = np.int((l[0] + l[2]) / 2)
            mid_y = np.int(l[1] + ((l[2]-l[0])*np.cos(np.pi/6) - ((l[2]-l[0])/2)/np.cos(np.pi/6)))
            return mid_x,  mid_y  # return we are done.
    return None, None


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
    # cv2.imshow('Canny Map', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    hough_lines = hough_lines[0, :]  # cleanup dimensionality to make it easier to work with.

    # to capture the top and bottom of the stop sign,
    left_x_cord, right_x_cord, bot_y_cord, top_y_cord = calculate_min_max_values(hough_lines)
    # lines = clean_edge_lines(hough_lines, left_x_cord, right_x_cord, bot_y_cord, top_y_cord)

    lines = remove_duplicates(hough_lines)

    if len(lines) < 8:
        return None, None

    # once given the lines of interest perform some more calculations
    min_x, max_x, min_y, max_y = calculate_min_max_values(lines)
    mid_x = min_x + ((max_x - min_x) / 2)
    mid_y = min_y + ((max_y - min_y) / 2)

    # print 'Mid is found at ({}, {})'.format(mid_x, mid_y)
    #
    # # loop over lines and place them on a new image to test.
    # imgc = img.copy()
    # for line in lines:
    #     cv2.line(imgc, (line[0], line[1]), (line[2], line[3]), (255, 122, 122), 2)
    # cv2.imshow('Hough Lines Map', imgc)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    if lines.shape[1] != 4: # this wasn't a diamond, return.
        return None, None
    min_x, max_x, min_y, max_y = calculate_min_max_values(lines)
    mid_x = min_x + ((max_x - min_x) / 2)
    mid_y = min_y + ((max_y - min_y) / 2)

    # print 'Warning mid at ({}, {})'.format(mid_x, mid_y)
    return mid_x, mid_y

    # # loop over lines and place them on a new image to test.
    # imgc = img.copy()
    # for line in hough_lines:
    #     cv2.line(imgc, (line[0], line[1]), (line[2], line[3]), (255, 122, 122), 2)
    # cv2.imshow('Hough Lines Map', imgc)
    # cv2.waitKey(0)
    #
    # raise NotImplementedError


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
    # cv2.imshow('Orange Map', orange_color_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    canny_edges = cv2.Canny(orange_color_map, threshold1=50, threshold2=250, apertureSize=5)

    # cv2.imshow('Canny Map', canny_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
        valid_idx = circle_mid_colors.index(255)
        the_sign = circles[valid_idx]
    else:
        return None, None

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
    img = img_in.copy()
    dict = {} # return dictionary

    # traffic lights
    (x, y), state = traffic_light_detection(img, (3, 30), other_signs=True)
    if state is not None:
        dict['traffic_light'] = (x,y)

    # orange signs ( construction )
    (x, y) = construction_sign_detection(img)
    if x is not None:
        dict['construction'] = (x, y)
    #
    # # yellow signs (warning)
    # (x, y) = warning_sign_detection(img)
    # if x is not None:
    #     dict['warning'] = (x, y)
    #
    # # stop sign
    # (x, y) = stop_sign_detection(img)
    # if x is not None:
    #     dict['stop'] = (x, y)
    #
    # # yield sign
    # (x, y) = yield_sign_detection(img)
    # if x is not None:
    #     dict['yield'] = (x, y)
    #
    # # dne sign
    # (x, y) = do_not_enter_sign_detection(img)
    # if x is not None:
    #     dict['no_entry'] = (x, y)

    return dict
    # raise NotImplementedError


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
        templateWindowSize=10,
        searchWindowSize=21,
        h=13,
        hColor=30
    )
    cv2.imshow('Cleaned Picture', clean_picture)
    cv2.waitKey(0)
    return traffic_sign_detection(clean_picture)


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


