import argparse

import cv2
import saraRC_rankIndex as rs

# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=-1,
                help="which saliency ranked segment to show")
args = vars(ap.parse_args())

IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]
# -------------------------------------------------
# -------------------------------------------------
# Start Main Code
# -------------------------------------------------
# -------------------------------------------------

# Get List of ROI of Mask R-CNN

imgPath = "sal.jpeg"

s1 = cv2.imread(IMAGE_DIR)
cv2.imshow("Input Image", s1)
print("Generating SaRa")

outS1, saraListS1 = rs.returnSARA(s1, RANK_TO_SHOW)
cv2.imshow("SaRa Output for S1", outS1)
cv2.imwrite("out.jpg", outS1)
print(saraListS1)

cv2.waitKey()
