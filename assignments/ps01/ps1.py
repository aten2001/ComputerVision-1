import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    # Since Red is last index, we want all rows, columns, and the last channel.
    return np.copy(image[:, :, 2])


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    # Return green channel, all rows, columns
    return np.copy(image[:, :, 1])


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    # Since blue is the first index, get first channel.
    return np.copy(image[:, :, 0])


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    temp_image[:, :, 0] = extract_green(image)
    temp_image[:, :, 1] = extract_blue(image)
    return temp_image

def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    src = np.copy(src)
    dst = np.copy(dst)

    # height is rows, width is columns
    src_rows, src_cols = src.shape
    dst_rows, dst_cols = dst.shape

    # shape size mid points.
    shape_mid_rows = int(np.floor(shape[0] / 2))
    shape_mid_cols = int(np.floor(shape[1] / 2))

    # mid point of the "copy" image
    copy_mid_row = int(np.floor(src_rows / 2))
    copy_mid_col = int(np.floor(src_cols / 2))

    # mid points of the paste image.
    paste_mid_row = int(np.floor(dst_rows / 2))
    paste_mid_col = int(np.floor(dst_cols / 2))

    # calculate the shifts to make sure copy is correct.
    r1_dst, r2_dst, c1_dst, c2_dst, r1_src, r2_src, c1_src, c2_src = [
        paste_mid_row - shape_mid_rows,
        paste_mid_row + shape_mid_rows,
        paste_mid_col - shape_mid_cols,
        paste_mid_col + shape_mid_cols,
        copy_mid_row - shape_mid_rows,
        copy_mid_row + shape_mid_rows,
        copy_mid_col - shape_mid_cols,
        copy_mid_col + shape_mid_cols
    ]

    dst[r1_dst: r2_dst, c1_dst: c2_dst] = src[r1_src: r2_src, c1_src: c2_src]
    return dst


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    return 1.*np.min(image), 1.*np.max(image), 1.*np.mean(image), 1.*np.std(image)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    i_min, i_max, i_mean, i_std = image_stats(image)
    # take the mean from the image, then divide by the std deviation. We then scale by the
    # scale factor and then add the mean back into the image.
    normal_image = (((image-i_mean) / i_std) * scale) + i_mean
    return normal_image


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    # take the temp image, all rows, from column defined in shift to end, move shift using border replicate.
    return cv2.copyMakeBorder(temp_image[:, shift:], 0, 0, 0, shift, cv2.BORDER_REPLICATE)



def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    difference = img1.astype(np.float) - img2.astype(np.float)
    output_image = np.zeros(difference.shape)
    cv2.normalize(difference, output_image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # print("Max Value is ", max(output_image.flatten()))
    # print("Min Value is ", min(output_image.flatten()))
    return output_image


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    # generate random noise using the image.shape tuple as the dimensions.
    gaussian_noise = np.random.randn(*image.shape) * sigma
    temp_image = np.copy(image)
    temp_image = (temp_image * 1.0)  # make it a float
    temp_image[:, :, channel] += gaussian_noise[:, :, channel]
    return temp_image
