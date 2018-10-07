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

    # fix to make sure the kernel size is an odd number.
    if k_size % 2 == 0 and k_size > 2:
        k_size = k_size - 1

    # copy images for use.
    image_a = img_a.copy()
    image_b = img_b.copy()

    if k_type == 'uniform':
        blur_kernel = (k_size, k_size)
        image_a = cv2.blur(image_a, blur_kernel)
        image_b = cv2.blur(image_b, blur_kernel)
    else:
        blur_kernel = cv2.getGaussianKernel(k_size=k_size, sigma=sigma)
        image_a = cv2.GaussianBlur(image_a, blur_kernel, sigma)
        image_b = cv2.GaussianBlur(image_b, blur_kernel, sigma)

    k = np.ones((k_size, k_size)) / (k_size ** 2)
    # base gradients.
    gx = gradient_x(image_a)
    gy = gradient_y(image_a)
    gt = image_b - image_a

    # calculate the weighted gradients using blurs to prevent having to loop. Makes this much easier to manage.
    if k_type == 'uniform':
        gx_x = cv2.filter2D(gx * gx, ddepth=-1, kernel=k)
        gy_y = cv2.filter2D(gy * gy, ddepth=-1, kernel=k)
        gx_y = cv2.filter2D(gx * gy, ddepth=-1, kernel=k)
        gy_x = cv2.filter2D(gy * gx, ddepth=-1, kernel=k)
        gx_t = cv2.filter2D(gx * gt, ddepth=-1, kernel=k)
        gy_t = cv2.filter2D(gy * gt, ddepth=-1, kernel=k)
    else:
        gx_x = cv2.GaussianBlur(gx * gx, blur_kernel, sigma)
        gy_y = cv2.GaussianBlur(gy * gy, blur_kernel, sigma)
        gx_y = cv2.GaussianBlur(gx * gy, blur_kernel, sigma)
        gy_x = cv2.GaussianBlur(gy * gx, blur_kernel, sigma)
        gx_t = cv2.GaussianBlur(gx * gt, blur_kernel, sigma)
        gy_t = cv2.GaussianBlur(gy * gt, blur_kernel, sigma)

    # create a new matrix from the gradients in the x, the xy, the yx, and the yy directions. To allow the use
    # of matrix multiplier provided by numpy, we flatten, then transpose to the correct shape of a 2x2 matrix.
    # np.matmul states that if either argument is N-D, N > 2, it is treated as a stack of matrices. We want a stack
    # of 2x2 matrices. residing in the last two indexes and broadcast accordingly.

    image_a_transform = np.array([
        gx_x.flatten(), gx_y.flatten(),
        gy_x.flatten(), gy_y.flatten()
    ]).T.reshape(-1, 2, 2)  # transpose and take from last a 2 x 2 matrix in the reshape.

    # 2 x 1 matrix for matmul
    time_transform = np.array([
        -gx_t.flatten(),
        -gy_t.flatten()
    ]).T.reshape(-1, 2, 1)

    try:
        image_a_transform_inv = np.linalg.inv(image_a_transform)
    except np.linalg.linalg.LinAlgError:
        # if this is a singular matrix (happens when the shift is the same as the original)
        return np.zeros(image_a.shape), np.zeros(image_a.shape)

    transform = np.matmul(image_a_transform_inv, time_transform)
    U = transform[:, 0, 0].reshape(image_a.shape)  # get hte U matrix
    V = transform[:, 1, 0].reshape(image_a.shape)  # get the V
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

    image = image.copy()
    # only select odd columns out.
    image = image.copy()
    # 1/16 4/16  6/16 4/12 1/16
    # k = np.array([1, 4, 6, 4, 1]) / 16.0
    k = np.array([np.array([1, 4, 6, 4, 1]) / 16.0])
    r_k = np.dot(k.T, k)
    filter_img = cv2.filter2D(image, -1, r_k)
    return filter_img[::2, ::2]


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
    image = image.copy()
    images = [image]   # the list for the pyramid. has base image first.
    for i in range(levels-1):
        image = reduce_image(image)  # reduce it then append. n times.
        images.append(image)
    return images


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

    # height is constant. image will never be larger than the first.
    h = img_list[0].shape[0]
    # width will be sum of all images size in the x direction
    w = sum([i.shape[1] for i in img_list])

    output = np.zeros((h, w))  # empty image.
    curr_x = 0
    for image in img_list:
        ih, iw = image.shape  # use this to determine where to place.
        output[: ih, curr_x: curr_x + iw] = normalize_and_scale(image)
        curr_x += iw

    return output


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
    image = image.copy()
    dr = 2 * image.shape[0]  # double rows.
    dc = 2 * image.shape[1]  # double columns

    output = np.zeros((dr, dc))
    # # filling alternate rows
    output[::2, ::2] = image
    # 2/16, 8/16, 12/16, 8/16, 2/16
    k = np.array([np.array([2, 8, 12, 8, 2]) / 16.0])
    r_k = np.dot(k.T, k)
    output = cv2.filter2D(output, -1, r_k)
    return output


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    images = []
    for idx in range(len(g_pyr)):
        if idx == len(g_pyr) - 1:
            image = g_pyr[idx]
        else:
            h = g_pyr[idx].shape[0]
            w = g_pyr[idx].shape[1]
            # expand the image that is found to the next images size.
            image = g_pyr[idx] - expand_image(g_pyr[idx + 1])[:h, :w]
        images.append(image)

    return images


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

    image = image.copy()

    # displacements in x-axis and y-axis as U and V for image as float32
    U = U.astype(np.float32)
    V = V.astype(np.float32)

    mesh_x, mesh_y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    # meshgrid for image as float32
    mesh_x = mesh_x.astype(np.float32)
    mesh_y = mesh_y.astype(np.float32)
    mesh_x += U
    mesh_y += V

    output = cv2.remap(src=image, map1=mesh_x, map2=mesh_y, interpolation=interpolation, borderMode=border_mode)
    return output



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
