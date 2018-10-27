"""
CS6476 Computer Vision
Problem Set 0 
Simple script to verify your installation
"""
import unittest

class TestInstall(unittest.TestCase):
    def setUp(self):
        pass

    def test_numpy(self):
        import numpy

    def test_cv2(self):
        import cv2

    def test_cv2_version(self):
        import cv2
        v = cv2.__version__.split(".")
        self.assertTrue(v[0] == '2' and v[1] == '4', 'Wrong OpenCV version. '
                                                     'Make sure you installed OpenCV 2.4.x.'
                                                     'Any other OpenCV versions i.e. 3.+ are not supported.')

    def test_ORB(self):
        from cv2 import ORB
        test_orb = ORB()

    def test_load_mp4_videos(self):
        import cv2
        video = cv2.VideoCapture("turtle.mp4")
        okay, frame = video.read()
        self.assertTrue(okay, "Loading mp4 video failed")

    def test_show_image(self):
        import cv2
        video = cv2.VideoCapture("turtle.mp4")
        okay, frame = video.read()
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()

