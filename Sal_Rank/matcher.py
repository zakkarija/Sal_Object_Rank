from PIL import Image
import cv2
from matplotlib import pyplot as plt
import imagehash
import numpy as np


def hash_matcher(c1, c2):
    # img1 = cv2.cvtColor(c1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.cvtColor(c2, cv2.COLOR_BGR2RGB)

    hash0 = imagehash.average_hash(Image.fromarray(c1))
    hash1 = imagehash.average_hash(Image.fromarray(c2))
    cutoff = 10

    if hash0 - hash1 < cutoff:
        # print("Similar")
        return True
    else:
        # print("Not similar")
        return False


img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")
large_img = cv2.imread("large_img.jpg")


def orb_matcher(img1, img2):
    # orb = cv2.ORB_create(nfeatures=1500)
    orb = cv2.ORB_create()

    # keypoints_orb, descriptorsOrb = orb.detectAndCompute(img1, None)
    # keypoints_orb2, descriptorsOrb2 = orb.detectAndCompute(img2, None)
    #
    # bookOrb = cv2.drawKeypoints(img1, keypoints_orb, None)
    # no_bookOrb = cv2.drawKeypoints(img2, keypoints_orb, None)
    #
    # cv2.imshow("ORB Book", bookOrb)
    # cv2.imshow("ORB no Book", no_bookOrb)
    # cv2.waitKey(0)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], flags=2, outImg=None)
    cv2.imshow("Matching", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_matcher(im1, im2):
    # There are 6 comparison methods to choose from:
    # TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED
    # You can see the differences at a glance here:
    # https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
    # Note that the values are inverted for TM_SQDIFF and TM_SQDIFF_NORMED
    result = cv2.matchTemplate(im1, im2, cv2.TM_SQDIFF_NORMED)

    # I've inverted the threshold and where comparison to work with TM_SQDIFF_NORMED
    threshold = 0.2
    # The np.where() return value will look like this:
    # (array([482, 483, 483, 483, 484], dtype=int32), array([514, 513, 514, 515, 514], dtype=int32))
    locations = np.where(result <= threshold)
    # We can zip those up into a list of (x, y) position tuples
    locations = list(zip(*locations[::-1]))
    print(locations)

    if locations:
        print('Found Object')
        needle_w = im2.shape[1]
        needle_h = im2.shape[0]
        line_color = (0, 255, 0)
        line_type = cv2.LINE_4

        # Loop over all the locations and draw their rectangle
        for loc in locations:
            # Determine the box positions
            top_left = loc
            bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)
            # Draw the box
            cv2.rectangle(im1, top_left, bottom_right, line_color, line_type)

        cv2.imshow('Matches', im1)
        cv2.waitKey()
        # cv2.imwrite('result.jpg', im1)
        return True
    else:
        print('Object not found.')
        return False

# hash_matcher(img1, img2)
# orb_matcher(img1, img2)
# template_matcher(img1, img2)
