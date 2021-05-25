import argparse
import os

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
ap.add_argument("-i", "--image", required=False, default="images",
                help="path to input image")
ap.add_argument("-r", "--rank", required=False, default=-1,
                help="which saliency object rank to show"),
ap.add_argument("-g", "--gaussian", type=util.str2bool, required=False, default="True",
                help="factor in gaussian map on saliency score")
ap.add_argument("-m", "--mask", type=util.str2bool, required=False, default="False",
                help="factor in gaussian map on saliency score")
ap.add_argument("-s", "--sal", type=util.sal_map_ver, required=False, default="Itti",
                help="Type of saliency map excepted")
ap.add_argument("-o", "--output", required=False, default="output.txt",
                help="Directory of product list output text file\n"
                     "(appends to existing file or creates a file if the specified file does not exist)")
args = vars(ap.parse_args())

# Convert arguments to variables
IMAGE_DIR = args["image"]
RANK_TO_SHOW = args["rank"]
OUTPUT = args["output"]
pr.GAUSSIAN = args["gaussian"]
pr.MASK = args["mask"]
pr.SAL_TYPE = args["sal"]


def rank(image):
    start = time.process_time()

    img = cv2.imread(image)

    # Get List of Products ROIs using Mask R-CNN
    results = rcnn.detect_objects(image)
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
    print("Image:\t", image)
    print("Gaussian:\t", pr.GAUSSIAN)
    print("Mask:\t", pr.MASK)
    print("Saliency Map:\t", pr.SAL_TYPE)

    objectRanked = pr.returnObjects(img, RANK_TO_SHOW, results)

    outF = open(OUTPUT, "a")
    outF.write("Ranking: " + image + "\n\n")

    i = 0
    for ranked_object in objectRanked:
        object_class = rcnn.getClassNameByObject(ranked_object[1])
        print("Rank ", ranked_object[3], ": ", object_class, "saliency score ", ranked_object[2], " gaussian score ",
              ranked_object[4])
        # print("Rank ", ranked_object[3], ": ", object_class, "saliency score ", ranked_object[2])
        outF.write(
            "Rank " + str(ranked_object[3]) + ": " + object_class + " saliency score: "
            + str(ranked_object[2]) + " gaussian score: " + str(ranked_object[4]) + "\n")

        i += 1

    outF.write("\n\n")
    print("Result printed to output.txt")
    print("Time took: ", time.process_time() - start)


for filename in os.listdir(IMAGE_DIR):
    if filename.endswith(".jpeg"):
        print("________________________________________________________________________________")
        print("Executing: ", filename)
        print("________________________________________________________________________________")
        img = os.path.join(IMAGE_DIR, filename)
        print(img)
        rank(img)
        print("________________________________________________________________________________")
        print("Executed: ", filename)
        print("________________________________________________________________________________")
    else:
        continue
