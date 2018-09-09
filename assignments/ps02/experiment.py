"""
CS6476: Problem Set 2 Experiment file

This script contains a series of function calls that run your ps2
implementation and output images so you can verify your results.
"""


import cv2

import ps2


def draw_tl_center(image_in, center, state):
    """Marks the center of a traffic light image and adds coordinates
    with the state of the current image

    Use OpenCV drawing functions to place a marker that represents the
    traffic light center. Additionally, place text using OpenCV tools
    that show the numerical and string values of the traffic light
    center and state. Use the following format:

        ((x-coordinate, y-coordinate), 'color')

    See OpenCV's drawing functions:
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    Make sure the font size is large enough so that the text in the
    output image is legible.
    Args:
        image_in (numpy.array): input image.
        center (tuple): center numeric values.
        state (str): traffic light state values can be: 'red',
                     'yellow', 'green'.

    Returns:
        numpy.array: output image showing a marker representing the
        traffic light center and text that presents the numerical
        coordinates with the traffic light state.
    """
    img = image_in.copy()
    # draw the midpoint.
    cv2.circle(img, (center[0], center[1]), 4, (0, 0, 0), -3)
    # add the state
    disp_str = '(({}, {} ), {} )'.format(center[0], center[1], state)
    cv2.putText(img, disp_str, (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    return img


def mark_traffic_signs(image_in, signs_dict):
    """Marks the center of a traffic sign and adds its coordinates.

    This function uses a dictionary that follows the following
    structure:
    {'sign_name_1': (x, y), 'sign_name_2': (x, y), etc.}

    Where 'sign_name' can be: 'stop', 'no_entry', 'yield',
    'construction', 'warning', and 'traffic_light'.

    Use cv2.putText to place the coordinate values in the output
    image.

    Args:
        image_in (numpy.array): the image to draw on.
        signs_dict (dict): dictionary containing the coordinates of
        each sign found in a scene.

    Returns:
        numpy.array: output image showing markers on each traffic
        sign.
    """
    img = image_in.copy()

    # loop over the dictionary, add all KVP as lables and mid points.
    for key, value in signs_dict.iteritems():
        cv2.circle(img, value, 4, (0, 0, 0), -3)
        # each character is about 10 pixels, shift to fit
        word_width = 10 * len(key)

        # place the text below the midpoint value
        key_offset = value[0] - word_width / 2

        # print 'Img Width is: {}'.format(img.shape[1])
        # print 'key offset is: {}'.format(key_offset)
        # print 'Word with is: {}'.format(word_width)
        # print 'With words edge will be at {} '.format(key_offset + (word_width/2))
        cv2.putText(img, key, (key_offset, value[1] + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    return img


def part_1():

    input_images = ['simple_tl', 'scene_tl_1', 'scene_tl_2', 'scene_tl_3']
    output_labels = ['ps2-1-a-1', 'ps2-1-a-2', 'ps2-1-a-3', 'ps2-1-a-4']

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):

        tl = cv2.imread("input_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("output/{}.png".format(label), img_out)

def part_1_tests():
    input_images = ['simple_tl_test',
                    'tl_green_299_287_blank',
                    'tl_red_199_137_blank',
                    'tl_yellow_199_237_blank'
                    ]
    output_labels = ['simple_tl_out',
                     'tl_green_299_287_out',
                     'tl_red_199_137_out',
                     'tl_yellow_199_237_out'
                     ]

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):
        tl = cv2.imread("input_images/test_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        print 'Cords for image name: {}.png are {} , state is: {}'.format(img_in, coords, state)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_1_scenes_test():
    input_images = ['scene_all_signs',
                    'tl_green_299_287_background',
                    'tl_red_199_137_background',
                    'tl_yellow_199_237_background',
                    ]
    output_labels = ['sscene_all_signs_out',
                     'tl_green_299_287_background_out',
                     'tl_red_199_137_background_out',
                     'tl_yellow_199_237_background_out'
                     ]

    # Define a radii range, you may define a smaller range based on your
    # observations.
    radii_range = range(10, 30, 1)

    for img_in, label in zip(input_images, output_labels):
        tl = cv2.imread("input_images/test_images/{}.png".format(img_in))
        coords, state = ps2.traffic_light_detection(tl, radii_range)

        print 'Cords for image name: {}.png are {} , state is: {}'.format(img_in, coords, state)

        img_out = draw_tl_center(tl, coords, state)
        cv2.imwrite("output/{}.png".format(label), img_out)

def part_2():

    input_images = ['scene_dne_1',
                    'scene_stp_1',
                    'scene_constr_1',
                    'scene_wrng_1',
                    'scene_yld_1'
                    ]

    output_labels = ['ps2-2-a-1',
                     'ps2-2-a-2',
                     'ps2-2-a-3',
                     'ps2-2-a-4',
                     'ps2-2-a-5'
                     ]

    sign_fns = [ps2.do_not_enter_sign_detection,
                ps2.stop_sign_detection,
                ps2.construction_sign_detection,
                ps2.warning_sign_detection,
                ps2.yield_sign_detection
                ]

    sign_labels = ['no_entry',
                   'stop',
                   'construction',
                   'warning',
                   'yield'
                   ]

    for img_in, label, fn, name in zip(input_images, output_labels, sign_fns,
                                       sign_labels):

        sign_img = cv2.imread("input_images/{}.png".format(img_in))
        coords = fn(sign_img)

        temp_dict = {name: coords}
        img_out = mark_traffic_signs(sign_img, temp_dict)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_3():

    input_images = [ # 'scene_some_signs',
                    'scene_all_signs']
    output_labels = [ # 'ps2-3-a-1',
                     'ps2-3-a-2']

    for img_in, label in zip(input_images, output_labels):

        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_4():
    input_images = ['scene_some_signs_noisy', 'scene_all_signs_noisy']
    output_labels = ['ps2-4-a-1', 'ps2-4-a-2']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_noisy(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_5a():
    input_images = ['img-5-a-1', 'img-5-a-2', 'img-5-a-3']
    output_labels = ['ps2-5-a-1', 'ps2-5-a-2', 'ps2-5-a-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


def part_5b():
    input_images = ['img-5-b-1', 'img-5-b-2', 'img-5-b-3']
    output_labels = ['ps2-5-b-1', 'ps2-5-b-2', 'ps2-5-b-3']

    for img_in, label in zip(input_images, output_labels):
        scene = cv2.imread("input_images/{}.png".format(img_in))
        coords = ps2.traffic_sign_detection_challenge(scene)

        img_out = mark_traffic_signs(scene, coords)
        cv2.imwrite("output/{}.png".format(label), img_out)


if __name__ == '__main__':

    # part_1()
    # part_1_tests()
    part_1_scenes_test()

    #part_2()
    # img_name = 'yield_173_358_blank'
    # sign_img = cv2.imread("input_images/test_images/{}.png".format(img_name))
    # ps2.yield_sign_detection(sign_img)

    # part_3()
    # part_4()
    # part_5a()
    # part_5b()

