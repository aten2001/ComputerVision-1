"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "output"
video_dir = "input_videos"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0), base_image=None):
    if base_image is None:
        img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)
    else:
        img_out = base_image.copy()

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):
            try:
                cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                           y + int(v[y, x] * scale)), color, 1)
                cv2.circle(img_out, (x + int(u[y, x] * scale),
                                     y + int(v[y, x] * scale)), 1, color, 1)
            except OverflowError:
                cv2.circle(img_out, (x, y), 1, color, 1)

    return img_out


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


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(output_dir, filename), image)


def helper_for_part_6(video_name, fps, frame_ids, output_prefix, counter_init):

    video = os.path.join(video_dir, video_name)
    image_gen = video_frame_generator(video)

    image_a = image_gen.next()  # frame 1
    image_b = image_gen.next()



    h, w, d = image_a.shape

    levels = 8
    k_size = 41
    k_type = 'uniform'
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    out_path = "output/ar_{}-{}".format(output_prefix[4:], video_name)

    video_out = mp4_video_writer(out_path, (w, h), fps)
    output_counter = counter_init

    frame_num = 1
    output_counter = 0
    output_frame = frame_ids[output_counter]

    while image_b is not None:
        a = cv2.cvtColor(image_a.copy(), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(image_b.copy(), cv2.COLOR_BGR2GRAY)

        u, v = ps4.hierarchical_lk(a, b, levels, k_size,
                                   k_type, 0, interpolation, border_mode)

        image = quiver(u, v, scale=3, stride=10, base_image=image_a)

        if output_counter < len(frame_ids) and frame_num == output_frame:
            print 'Saving image of {}'.format(frame_num)
            out_str = "frame-{}.png".format(frame_num)
            save_image(out_str, image)

            output_counter += 1
            if output_counter < len(frame_ids):
                output_frame = frame_ids[output_counter]

            print 'Saving Next {} : Data of {} and {}'.format( output_frame, output_counter, len(frame_ids))

        # video_out.write(image)
        image_a = image_b.copy()
        image_b = image_gen.next()
        # save_image('b.png', image_b)
        # image_b = cv2.imread('output/b.png', 0) / 1.
        frame_num += 1

        print 'Frame: {}'.format(frame_num)

    video_out.release()


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MPEG')
    filename = filename.replace('mp4', 'mpeg')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    for i in range(level-1, -1, -1 ):
        u = 2.0 * ps4.expand_image(u)
        v = 2.0 * ps4.expand_image(v)

    h, w = pyr[0].shape
    return u[:h, :w], v[:h, :w]


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.




    k_size = 19  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel


    # Optional: smooth the images if LK doesn't work well on raw images
    blur_kernel = (k_size, k_size)
    shift_0 = cv2.blur(shift_0, blur_kernel)
    shift_r2 = cv2.blur(shift_r2, blur_kernel)
    shift_r5_u5 = cv2.blur(shift_r5_u5, blur_kernel)


    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.
    k_size = 27
    # k_type = ""  # TODO: Select a kernel type
    # sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


# used for trackbar tuning
def nothing(x):
    pass


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    k_type = 'uniform'
    sigma = 0.5

    # k_size = "kSize"
    # window = "Params"
    # cv2.namedWindow(window)
    # cv2.createTrackbar(k_size, window, 1, 100, nothing)
    # while 1:
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    #
    #     k_size = cv2.getTrackbarPos('kSize', 'Params')
    #     u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    #     u_v = quiver(u, v, scale=3, stride=10)
    #     cv2.imshow("Params", u_v)

    # change k size per thing to see impact.
    k_size = 41
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    k_size = 63
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    k_size = 63
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4 # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    # k_size = "kSize"
    # window = "Params"
    # cv2.namedWindow(window)
    # cv2.createTrackbar(k_size, window, 1, 100, nothing)
    # while 1:
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    #
    #     level_id = 0  # TODO: Select the level number (or id) you wish to use
    #     # k_size = 11  # TODO: Select a kernel size
    #     k_size = cv2.getTrackbarPos('kSize', 'Params')
    #     k_type = 'uniform'  # TODO: Select a kernel type
    #     sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    #     u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
    #                              yos_img_02_g_pyr[level_id],
    #                              k_size, k_type, sigma)
    #
    #     u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)
    #
    #     interpolation = cv2.INTER_CUBIC  # You may try different values
    #     border_mode = cv2.BORDER_REFLECT101  # You may try different values
    #     yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)
    #
    #     diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    #
    #     cv2.imshow("Params", diff_yos_img_01_02)


    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 23  # TODO: Select a kernel size
    k_type = 'uniform'  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 23  # TODO: Select a kernel size
    k_type = 'uniform'  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 4  # TODO: Define the number of levels
    k_size = 29  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    k_size = 71
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 4
    k_size = 57
    k_type = 'uniform'
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    # k_size = "kSize"
    # window = "Params"
    # cv2.namedWindow(window)
    # cv2.createTrackbar(k_size, window, 1, 100, nothing)
    # while 1:
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    #
    #     level_id = 1  # TODO: Select the level number (or id) you wish to use
    #     # k_size = 11  # TODO: Select a kernel size
    #     k_size = cv2.getTrackbarPos('kSize', 'Params')
    #     k_type = 'uniform'  # TODO: Select a kernel type
    #
    #     u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
    #                                k_type, sigma, interpolation, border_mode)
    #
    #     u_v = quiver(u, v, scale=3, stride=10)
    #     cv2.imshow("Params", u_v)

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    k_type = 'uniform'
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    interpolation = cv2.INTER_CUBIC

    images = []
    # need intervals for  0.2, 0.4, 0.6, 0.8. arange is not end inclusive.
    intervals = np.arange(0.2, 1, 0.2)

    image_1 = cv2.imread('input_images/TestSeq/Shift0.png', 0) / 1.
    image_2 = cv2.imread('input_images/TestSeq/ShiftR10.png', 0) / 1.

    images.append(image_1) # add t0
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(0))), image_1)

    k_size = 21


    # loop over time intervals.
    counter = 1
    img = image_1.copy()
    for i in intervals:
        U, V = ps4.hierarchical_lk(img, image_2, levels=4, k_size=k_size, k_type=k_type, sigma=0,
                                   interpolation=interpolation, border_mode=border_mode)
        # orientate and scale
        U = -U * i
        V = -V * i
        warped = ps4.warp(img, U, V, interpolation=interpolation, border_mode=border_mode)

        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), warped)
        images.append(warped)
        img = warped
        counter += 1

    images.append(image_2) # add t1
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), image_2)

    # build output image
    # r1 0, 0.2, 0.4
    # r2 0.6, 0.8, 1
    row_1 = np.concatenate((images[0], images[1], images[2]), axis=1)
    row_2 = np.concatenate((images[3], images[4], images[5]), axis=1)
    output = np.concatenate((row_1, row_2), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps4-5-1-a-1.png'), output)


def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    k_type = 'uniform'
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    interpolation = cv2.INTER_CUBIC

    images = []
    # need intervals for 0, 0.2, 0.4, 0.6, 0.8 and 1. arange is not end inclusive.
    intervals = np.arange(0.2, 1, 0.2)

    image_1 = cv2.imread('input_images/MiniCooper/mc01.png', 0) / 1.
    image_2 = cv2.imread('input_images/MiniCooper/mc02.png', 0) / 1.
    images.append(image_1)
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(10))), image_1)

    k_size = 45

    # # loop over time intervals.
    counter = 11
    img = image_1
    for i in intervals:

        # from the current interval image to desired output.
        U, V = ps4.hierarchical_lk(img, image_2, levels=4, k_size=k_size, k_type=k_type, sigma=0,
                                   interpolation=interpolation, border_mode=border_mode)

        # orientate.
        U = -U # * i
        V = -V # * i
        warped = ps4.warp(img, U, V, interpolation=interpolation, border_mode=border_mode)
        img = warped  # reset for next iteration
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), warped)
        images.append(warped)
        counter += 1

    images.append(image_2)
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), image_2)

    # build output image
    # r1 0, 0.2, 0.4
    # r2 0.6, 0.8, 1
    row_1 = np.concatenate((images[0], images[1], images[2]), axis=1)
    row_2 = np.concatenate((images[3], images[4], images[5]), axis=1)
    output = np.concatenate((row_1, row_2), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps4-5-1-b-1.png'), output)

    # part 2

    images = []
    image_1 = cv2.imread('input_images/MiniCooper/mc02.png', 0) / 1.
    image_2 = cv2.imread('input_images/MiniCooper/mc03.png', 0) / 1.
    images.append(image_1)

    k_size = 45

    # # loop over time intervals.
    counter = 21
    img = image_1
    for i in intervals:
        # from the current interval image to desired output.
        U, V = ps4.hierarchical_lk(img, image_2, levels=4, k_size=k_size, k_type=k_type, sigma=0,
                                   interpolation=interpolation, border_mode=border_mode)
        # orientate.
        U = -U # * i
        V = -V # * i
        warped = ps4.warp(img, U, V, interpolation=interpolation, border_mode=border_mode)
        img = warped  # reset for next iteration
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), warped)
        images.append(warped)
        counter += 1

    images.append(image_2)
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(str(counter))), image_2)

    # build output image
    # r1 0, 0.2, 0.4
    # r2 0.6, 0.8, 1
    row_1 = np.concatenate((images[0], images[1], images[2]), axis=1)
    row_2 = np.concatenate((images[3], images[4], images[5]), axis=1)
    output = np.concatenate((row_1, row_2), axis=0)
    cv2.imwrite(os.path.join(output_dir, 'ps4-5-1-b-2.png'), output)



def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    # video_file = "ps3-4-a.mp4"
    # frame_ids = [355, 555, 725]
    # fps = 40
    #
    # helper_for_part_6(video_file, fps, frame_ids, "ps3-5-a", 1)

    video_file = 'short_video.mp4'
    frame_ids = [7, 12]
    fps = 30

    helper_for_part_6(video_file, fps, frame_ids, 'ps4-6-a', 1)


if __name__ == "__main__":
    # part_1a()
    # part_1b()
    # part_2()
    # part_3a_1()
    # part_3a_2()
    # part_4a()
    # part_4b()
    # part_5a()
    # part_5b()
    part_6()
