import argparse
import cv2
import SR as psr
import time
import mask_rcnn as rcnn

# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default="images\\3_books.jpeg",
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=-1,
                help="which saliency object rank to show"),
ap.add_argument("-g", "--gaussian", required=False, default="True",
                help="factor in gaussian map on saliency score")
args = vars(ap.parse_args())

IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]

if args["gaussian"] == "False":
    GAUSSIAN = False
else:
    GAUSSIAN = True
# -------------------------------------------------
# Start Main Code
# -------------------------------------------------
start = time.process_time()

img = cv2.imread(IMAGE_DIR)

# Get List of Products ROIs using Mask R-CNN
results = rcnn.detect_objects(IMAGE_DIR)
# print("results", results)


# cv2.imshow("Input Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Show detected Objects
rois = results['rois']
ids = results['class_ids']
# for roi in rois:
#     obj = img[roi[0]:roi[2], roi[1]:roi[3]]
#     cv2.imshow("Object", obj)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

objectRanked = psr.returnObjects(img, RANK_TO_SHOW, results, GAUSSIAN)

i = 0
for ranked_object in objectRanked:
    object_class = rcnn.getClassNameByObject(ranked_object[1])
    print("Rank ", ranked_object[3], ": ", object_class, " with a saliency score ", ranked_object[2])
    i += 1

print("Time took: ", time.process_time() - start)

cv2.waitKey(0)
cv2.destroyAllWindows()
