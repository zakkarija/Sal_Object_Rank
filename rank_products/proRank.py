import argparse
import cv2
import SR as psr
import SR_mask as psrm
import time
import mask_rcnn as rcnn


# -------------------------------------------------
# Parse Arguments
# -------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default="images\\3_books.jpeg",
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=-1,
                help="which saliency object rank to show"),
ap.add_argument("-g", "--gaussian", type=str2bool, required=False, default="True",
                help="factor in gaussian map on saliency score")
ap.add_argument("-m", "--mask", type=str2bool, required=False, default="True",
                help="factor in gaussian map on saliency score")
args = vars(ap.parse_args())

# Convert arguments to variables
IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]
psrm.GAUSSIAN = args["gaussian"]
psrm.MASK = args["mask"]

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
print("Gaussian:\t", psrm.GAUSSIAN)
print("Mask:\t", psrm.MASK)

# objectRanked = psr.returnObjects(img, RANK_TO_SHOW, results, GAUSSIAN)
objectRanked = psrm.returnObjects(img, RANK_TO_SHOW, results)

i = 0
for ranked_object in objectRanked:
    object_class = rcnn.getClassNameByObject(ranked_object[1])
    print("Rank ", ranked_object[3], ": ", object_class, " with a saliency score ", ranked_object[2])
    i += 1

print("Time took: ", time.process_time() - start)

cv2.waitKey(0)
cv2.destroyAllWindows()
