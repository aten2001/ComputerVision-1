"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def display_img(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_base_image(img, kernel_size=(5, 5), show_image=False):
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


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    e_distance = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
    return e_distance


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    img_in = image.copy()
    top_left = (0, 0)
    bottom_left = (0, img_in.shape[0] - 1)
    top_right = (img_in.shape[1]-1, 0)
    bottom_right = (img_in.shape[1] - 1, img_in.shape[0] - 1)

    return top_left, bottom_left, top_right, bottom_right


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # w, h = template.shape[::-1]
    # print 'Template w nad h {}, {}'.format(w, h)
    # w = np.int32(w / 2.0)
    # h = np.int32(h / 2.0)
    #
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    # print res.max()
    # threshold = 0.8
    # loc = np.where(res >= threshold)
    #
    # box_cords = zip(*loc[::-1])
    #
    # # cords to midpoints of template.
    # cords = []
    # for point in zip(*loc[::-1]):
    #     cords.append([point[0] + w, point[1] + h])
    #     cv2.circle(image, (point[0] + w, point[1] + h), 4, (0,0,255), -3)
    #     # cv2.rectangle(image, point, (point[0] + w, point[1] + h), (0, 0, 255), 2)
    #
    # display_img(image, 'Image matched')
    #
    # # sort the cords first.
    # sorted_cords = sorted(cords, key=lambda x: x[0])
    # sorted_cords = np.array(sorted_cords)
    # # left side circles are going to be the last two sorted in the y direction
    # left = sorted(sorted_cords[:2], key=lambda x: x[1])
    # # right side are the first two sorted in the y direction.
    # right = sorted(sorted_cords[2:], key=lambda x: x[1])
    #
    # print sorted_cords
    #
    # return tuple(left[0]), tuple(left[1]), tuple(right[0]), tuple(right[1])

    img_in = process_base_image(image)
    harris = cv2.cornerHarris(img_in, 8, 7, 0.05)
    # display_img(harris, 'The D image 1')
    harris_dilated = cv2.dilate(harris, None)
    # harris_dilated = cv2.morphologyEx(harris, cv2.MORPH_OPEN, (2, 2))
    # display_img(harris_dilated, 'The D Image')
    # obtain the areas of interest. These will be areas within 10% of the max value
    # found in the image.
    # print 'Max is: {}'.format(harris_dilated.max())
    # np.where is going to return 2 arrays which  need to be zipeed to be tuples of (x, y)

    cord_lists = np.where(harris_dilated >= .1 * harris_dilated.max())
    cords = zip(*cord_lists[::-1])
    cords = np.array([np.float32(cords)])

    # print (cords)
    # criteria copied from OPenCV documentation found here:
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    temp, classified_points, means = cv2.kmeans(cords, K=4, criteria=criteria, attempts=50,
                                                flags=cv2.KMEANS_RANDOM_CENTERS)

    circle_centers = np.int16(means)
    # print circle_centers
    # sort the circles by the x cord first.
    circle_centers = sorted(circle_centers, key=lambda x: x[0])
    # print np.array(circle_centers)
    circle_centers = np.array(circle_centers)
    # left side circles are going to be the last two sorted in the y direction
    left = sorted(circle_centers[:2], key=lambda x: x[1])
    # right side are the first two sorted in the y direction.
    right = sorted(circle_centers[2:], key=lambda x: x[1])

    return tuple(left[0]), tuple(left[1]), tuple(right[0]), tuple(right[1])


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img_in = image.copy()
    cv2.line(img_in, markers[0], markers[1], (0, 0, 255), thickness=thickness)
    cv2.line(img_in, markers[0], markers[2], (0, 0, 255), thickness=thickness)
    cv2.line(img_in, markers[1], markers[3], (0, 0, 255), thickness=thickness)
    cv2.line(img_in, markers[2], markers[3], (0, 0, 255), thickness=thickness)

    return img_in


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    imgSource = imageA.copy()
    imgDst = imageB.copy()
    # to have the polygon area make sense, need to re-arrange the markers to go from
    # top left, top right, bottom right, bottom left. Just as if you were drawing the shape.

    # to avoid doing a bunch of loops, create an array of all valid X,Y cords that we need to transform in the image
    # markers = get_corners_list(imgDst)
    markers = find_markers(imageB)
    markers = np.int32(markers)
    markers = [markers[0], markers[2], markers[3], markers[1]]
    markers = [[np.int32(marker[0]), np.int32(marker[1])] for marker in markers]

    background = imgDst.copy()
    # fill with gray box this is going to be the "background" image with a white box so we can blend the images
    # together to get the final result.
    cv2.fillPoly(background, np.array([markers]), (0, 0, 0), 0, 0)

    # How to use remap to project the image
    # https://stackoverflow.com/questions/46520123/failing-the-simplest-possible-cv2-remap-test-aka-how-do-i-use-remap-in-pyt
    H = np.linalg.inv(homography)
    # create indices of the destination image and linearize them
    h, w = imgDst.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([x.ravel(), y.ravel(), np.ones_like(x).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1] / map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)
    # remap!
    dst = cv2.remap(imgSource, map_x, map_y, cv2.INTER_LINEAR)
    blended = cv2.addWeighted(background, 1, dst, 1, 0)
    return blended


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    # p1 and p2 represent different planes (4 points on each)
    p1 = src_points
    p2 = dst_points

    # https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    # Matrix Transformation Ax=b --> given A and B find X
    matrix_a = np.array((p1[0][0], p1[1][0], p1[2][0], p1[0][1], p1[1][1], p1[2][1], 1, 1, 1)).reshape((3, 3))
    matrix_b = np.array((p1[3][0], p1[3][1], 1))
    homogenous_coord = np.linalg.solve(matrix_a, matrix_b)
    mat_a= matrix_a * homogenous_coord

    matrix_b = np.array((p2[0][0], p2[1][0], p2[2][0], p2[0][1], p2[1][1], p2[2][1], 1, 1, 1)).reshape((3, 3))
    matrixb2 = np.array((p2[3][0], p2[3][1], 1))
    homogenous_coord2 = np.linalg.solve(matrix_b, matrixb2)
    mat_b = matrix_b * homogenous_coord2
    return np.dot(np.float32(mat_b), np.linalg.inv(mat_a)) / np.dot(np.float32(mat_b), np.linalg.inv(mat_a))[2][2]


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None