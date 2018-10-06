"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os
import itertools


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    image_cpy = image.copy()
    # sobel scale = 1/8, ksize = 3 default border type
    sobel_x = cv2.Sobel(image_cpy, cv2.CV_64F, 1, 0, scale=0.125, ksize=3)
    return sobel_x


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    image_cpy = image.copy()
    # sobel scale = 1/8, ksize = 3 default border type
    sobel_y = cv2.Sobel(image_cpy, cv2.CV_64F, 0, 1, scale=0.125, ksize=3)
    return sobel_y


def gradient_t(image1, image2):
    return image2 - image1


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    image_a = img_a.copy()
    image_b = img_b.copy()
    image_a = cv2.GaussianBlur(image_a, (5, 5), 0)
    image_b = cv2.GaussianBlur(image_b, (5, 5), 0)

    temporal = image_b - image_a
    gradient_x_a = gradient_x(image_a)
    gradient_y_a = gradient_y(image_a)


    print 'K Type is {}'.format(k_type)
    print 'K Size is {}'.format(k_size)
    # perform blurring based on k_type to get the weighted sums. box filter or a smoothing
    # filter. Auto grader will use 'uniform'.
    # gx_gx
    # gx_gy
    # gy_gx
    # gy_gy

    if k_type == 'uniform':
        k = np.ones((k_size, k_size)) / np.float64(k_size ** 2)
        gx_gx = cv2.filter2D(gradient_x_a * gradient_x_a, ddepth=-1, kernel=k)  # a
        gy_gy = cv2.filter2D(gradient_y_a * gradient_y_a, ddepth=-1, kernel=k) # d
        gx_gy = cv2.filter2D(gradient_x_a * gradient_y_a, ddepth=-1, kernel=k) # b
        gy_gx = cv2.filter2D(gradient_y_a * gradient_x_a, ddepth=-1, kernel=k) # c

        # temporal gradients.
        gx_gt = cv2.filter2D(gradient_x_a * temporal, ddepth=-1, kernel=k)
        gy_gt = cv2.filter2D(gradient_y_a * temporal, ddepth=-1, kernel=k)
    else:
        gx_gx = cv2.GaussianBlur(gradient_x_a * gradient_x_a, (k_size, k_size), sigma)  # a
        gy_gy = cv2.GaussianBlur(gradient_y_a * gradient_y_a, (k_size, k_size), sigma)  # d
        gx_gy = cv2.GaussianBlur(gradient_x_a * gradient_y_a, (k_size, k_size), sigma)  # b
        gy_gx = cv2.GaussianBlur(gradient_y_a * gradient_x_a, (k_size, k_size), sigma)  # c
        # temporal gradients.
        gx_gt = cv2.GaussianBlur(gradient_x_a * temporal, (k_size, k_size), sigma)
        gy_gt = cv2.GaussianBlur(gradient_y_a * temporal, (k_size, k_size), sigma)

    # perform some transposes to get proper shapes of matrix. a 2x2 matrix is needed.
    a_tr_a = np.array([
        gx_gx.flatten(), gx_gy.flatten(),
        gy_gx.flatten(), gy_gy.flatten()]).T.reshape(-1, 2, 2)

    # singular matrix check.
    try:
        i_a_tr_a = np.linalg.inv(a_tr_a)
    except np.linalg.linalg.LinAlgError:
        return np.zeros(image_a.shape), np.zeros(image_a.shape)


    # need 2 x 1. On the temporal shifts, need to swap x,y dir so do an inverse.
    temporal_t = np.array([-gx_gt.flatten(), -gy_gt.flatten()]).T.reshape(-1, 2, 1)

    # perform matrix multiplication.
    shifts = np.matmul(i_a_tr_a, temporal_t)
    U = shifts[:, 0, 0].reshape(image_a.shape)
    V = shifts[:, 1, 0].reshape(image_b.shape)

    return U, V


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """

    raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """

    raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """

    raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError
