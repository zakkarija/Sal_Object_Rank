import argparse
import cv2
import saraRC_mrcnn as rs
import mask_rcnn as rcnn

# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default="eval_images\\3_hats.jpeg",
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=0,
                help="which saliency object rank to show")
args = vars(ap.parse_args())

IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]
# -------------------------------------------------
# -------------------------------------------------
# Start Main Code
# -------------------------------------------------
# -------------------------------------------------
img = cv2.imread(IMAGE_DIR)

# Get List of ROI of Mask R-CNN
results = rcnn.detect_objects(IMAGE_DIR)
print("results", results)
rois = results['rois']
ids = results['class_ids']

# Show detected Objects
# for roi in rois:
#     obj = img[roi[0]:roi[2], roi[1]:roi[3]]
#     cv2.imshow("Object", obj)
#     cv2.waitKey()


s1 = cv2.imread(IMAGE_DIR)
cv2.imshow("Input Image", s1)
print("Generating SaRa")

outS1, saraListS1 = rs.returnSARA(s1, RANK_TO_SHOW, results)
cv2.imshow("SaRa Output for S1", outS1)
cv2.imwrite("out.jpg", outS1)
print(saraListS1)

cv2.waitKey()
