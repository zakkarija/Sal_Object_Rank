import argparse
import cv2
import time
import SR_sal as pr
import mask_rcnn as rcnn
import prUtils as util

# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default="images\\3_books.jpeg",
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=-1,
                help="which saliency object rank to show"),
ap.add_argument("-g", "--gaussian", type=util.str2bool, required=False, default="False",
                help="factor in gaussian map on saliency score")
ap.add_argument("-m", "--mask", type=util.str2bool, required=False, default="False",
                help="factor in gaussian map on saliency score")
ap.add_argument("-s", "--sal", type=util.sal_map_ver, required=False, default="sr",
                help="Type of saliency map excepted")
args = vars(ap.parse_args())

# Convert arguments to variables
IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]
pr.GAUSSIAN = args["gaussian"]
pr.MASK = args["mask"]
pr.SAL_TYPE = args["sal"]

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

print("Showing Rank:\t", RANK_TO_SHOW)
print("Image:\t", IMAGE_DIR)
print("Gaussian:\t", pr.GAUSSIAN)
print("Mask:\t", pr.MASK)
print("Saliency Map:\t", pr.SAL_TYPE)

objectRanked = pr.returnObjects(img, RANK_TO_SHOW, results)

i = 0
for ranked_object in objectRanked:
    object_class = rcnn.getClassNameByObject(ranked_object[1])
    print("Rank ", ranked_object[3], ": ", object_class, "saliency score ", ranked_object[2], " gaussian score ",
          ranked_object[4])
    # print("Rank ", ranked_object[3], ": ", object_class, "saliency score ", ranked_object[2])
    i += 1

print("Time took: ", time.process_time() - start)

cv2.waitKey(0)
cv2.destroyAllWindows()
