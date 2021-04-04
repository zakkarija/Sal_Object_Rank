import itertools

import cv2
import numpy as np
import math
import operator

import matcher
import pySaliencyMap
import saliency_models

# -------------------------------------------------
# Global Variables
# -------------------------------------------------

objectEntropies = []
indexed_objects = []

GAUSSIAN = True
MASK = False

# -------------------------------------------------
# Class Names: Class's index in  list is its ID.
# -------------------------------------------------


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# -------------------------------------------------
# SaRa Initial Functions
# -------------------------------------------------

def returnIttiSaliency(img):
    imgsize = img.shape
    img_width = imgsize[1]
    img_height = imgsize[0]
    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
    saliency_map = sm.SMGetSM(img)

    # Scale pixel values to 0-255 instead of float (approx 0, hence black image)
    # https://stackoverflow.com/questions/48331211/how-to-use-cv2-imshow-correctly-for-the-float-image-returned-by-cv2-distancet/48333272
    saliency_map = cv2.normalize(saliency_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return saliency_map


# -------------------------------------------------
# Saliency Ranking
# -------------------------------------------------

segmentsCoords = []
segments = []


def generateSegments(img, seg_count, depth=None):
    segments = []
    segmentCount = seg_count
    index = 0

    wInterval = int(img.shape[1] / segmentCount)
    hInterval = int(img.shape[0] / segmentCount)

    for i in range(segmentCount):
        for j in range(segmentCount):
            # Note: img[TopRow:BottomRow, FirstColumn:LastColumn]
            tempSegment = img[int(hInterval * i):int(hInterval * (i + 1)), int(wInterval * j):int(wInterval * (j + 1))]
            # cv2.imshow("Crop" + str(i) + str(j), tempSegment)
            # coordTup = (index, x1, y1, x2, y2)
            coordTup = (
                index, int(wInterval * j), int(hInterval * i), int(wInterval * (j + 1)), int(hInterval * (i + 1)))
            segmentsCoords.append(coordTup)
            segments.append(tempSegment)
            index += 1


def calculatePixelFrequency(img):
    flt = img.flatten()
    unique, counts = np.unique(flt, return_counts=True)
    pixelsFrequency = dict(zip(unique, counts))

    return pixelsFrequency


def point_in_roi(x, y, object_roi):
    # print(object_roi[0], "<", x, "<", object_roi[2])
    # print(object_roi[1], "<", y, "<", object_roi[3])
    return object_roi[0] <= y <= object_roi[2] and object_roi[1] <= x <= object_roi[3]


def show_coords(img, x1, y1, x2, y2, label):
    roi = img[x1:x2, y1:y2]
    # print("roi", roi)
    cv2.imshow(label, roi)


def intersection(a, b, img):
    # print("b", b)
    # print("a", a)
    # show_coords(img, a[0], a[1], a[2], a[3], "a")
    # show_coords(img, b[0], b[1], b[2], b[3], "b")

    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y

    if w < 0 or h < 0: return tuple((0, 0, 0, 0))
    return tuple((x, y, w, h))


def getGaussianWeight(coords, kernel, img):
    gaussian_weights = []
    i = 0
    for seg in kernel:

        # print("seg", seg)
        # print("coords", coords)
        # print("segmentsCoords", segmentsCoords)

        # seg_coords(index,y1,x1,y2,x2)
        seg_coords = segmentsCoords[i]
        a = (coords[0], coords[1], coords[2], coords[3])
        b = (seg_coords[2], seg_coords[1], seg_coords[4], seg_coords[3])
        ans = intersection(a, b, img)

        if ans != (0, 0, 0, 0):
            gaussian_weights.append(float(seg))
            # print("Intersection", ans, "i:", i, "seg:", seg)
        # else:
        #     print("Box not in", i)
        # cv2.waitKey(0)
        i += 1
    return sum(gaussian_weights) / len(gaussian_weights)


def calculateEntropy(img, w, dw):
    flt = img.flatten()

    # c = flt.shape[0]
    # totalPixels = 0
    # tprob = 0
    # sumOfProbs = 0
    entropy = 0
    wt = w * 10

    # if imgD=None then proceed normally
    # else calculate its frequency and find max
    # use this max value as a weight in entropy

    pixelsFrequency = calculatePixelFrequency(flt)

    totalPixels = sum(pixelsFrequency.values())

    for px in pixelsFrequency:
        tprob = (pixelsFrequency.get(px)) / totalPixels
        # probs[px] = tprob
        entropy += entropy + (tprob * math.log(2, (1 / tprob)))

        entropy = entropy * wt * dw

    # print("Saliency Score:", entropy, " Gaussian weight:", wt)
    return entropy, wt


# -------------------------------------------------
# Product Ranking
# -------------------------------------------------

def makeGaussian(size, fwhm=10, center=None):
    # https://gist.github.com/andrewgiessel/4635563
    """ Make a 2D gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


# def findMostSalientObject(objs, kernel, dws):


def rankProductsWithSaliency(objs, kernel, sal_map, img):
    # objs is array of obj (roi, class_id, image mask, sal map mask)
    maxEntropy = 0
    index = 0
    i = 0
    for obj in objs:

        coords = obj[0]  # Mask

        if MASK:
            roi = obj[3][coords[0]:coords[2], coords[1]:coords[3]]
            # cv2.imshow("entropy roi", roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            # coords[0] is x1 or y1??
            roi = sal_map[coords[0]:coords[2], coords[1]:coords[3]]

        # print("roi", roi)
        # cv2.imshow("roi", roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if np.all((kernel == 0)):
            # calculateEntropy(sal_segment, gaussian weight, depth weight)
            # gaussian weight is 0.1 since it will be multiplied by 10
            entropy, gaussian = calculateEntropy(roi, 0.1, 1)
        else:
            gaussian_weight = getGaussianWeight(coords, kernel, img)
            entropy, gaussian = calculateEntropy(roi, gaussian_weight, 1)

        # objectEntropies is list of  (index, entropy)
        Tup = (i, entropy, gaussian)
        objectEntropies.append(Tup)
        # objects is list of (index, object)
        indexed_object = (i, obj)
        indexed_objects.append(indexed_object)
        if entropy > maxEntropy:
            maxEntropy = entropy
            index = i
        i += 1
        print("_________________________________")
    return maxEntropy, index


def getClassName(salient_object):
    return class_names[salient_object[1][1]]


def getRoiWithIndex(index, img):
    obj = getObjectWithIndex(index)
    object_coords = obj[1][0]
    return img[object_coords[0]:object_coords[2],
           object_coords[1]: object_coords[3]]


def getObjectWithIndex(index):
    # print('objects', indexed_objects)
    for objct in indexed_objects:
        # objct is  index , roi
        if int(objct[0]) == int(index):
            return objct


def getObjectWithRank(rank, img, obj_entropies):
    rank_iter = 0
    for obj in obj_entropies:
        if int(rank) == int(rank_iter):
            return obj[0], getObjectWithIndex(obj[0])
        rank_iter += 1


def duplicateObjects(img):
    for objA, objB in itertools.combinations(indexed_objects, 2):
        if objA[0] != objB[0]:
            index1 = objA[0]
            index2 = objB[0]
            print("Comparing index: {} with {} ".format(index1, index2))

            roi = objA[1][0]
            roi2 = objB[1][0]
            match1 = img[roi[0]:roi[2], roi[1]:roi[3]]
            match2 = img[roi2[0]:roi2[2], roi2[1]:roi2[3]]

            # if matcher.hash_matcher(match1, match2):
            # if matcher.template_matcher(match1, match2):
            if matcher.feature_match(match1, match2):
                print("Object Matches: ", getClassName(getObjectWithIndex(objA[0])),
                      " with ", getClassName(getObjectWithIndex(objB[0])))

                # print("indexed_objects", indexed_objects)
                # Instead of removing the object (Combine their averaged entropy)
                # indexed_objects.pop(objA[0])
                for obj_ent in objectEntropies:
                    if obj_ent[0] == objA[0]:
                        objectEntropies.remove(obj_ent)
                # print("indexed_objects After pop", indexed_objects)

                # cv2.imshow("Object Matches1", match1)
                # cv2.imshow("Object Matches2", match2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                print("\n\n")
            else:
                print("No Match Found\n\n")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # print("No Match found between", getClassName(getObjectWithIndex(objA[0])),
                #       " and", getClassName(getObjectWithIndex(objB[0])))


def show_ranked_objects(rankedObjs, img):
    i = 0
    for ranked_object in rankedObjs:
        if MASK:
            ranked_object_coords = ranked_object[1][1][0]
            ranked_object_img = ranked_object[1][1][2][ranked_object_coords[0]:ranked_object_coords[2],
                                ranked_object_coords[1]: ranked_object_coords[3]]
        else:
            ranked_object_coords = ranked_object[1][1][0]
            ranked_object_img = img[ranked_object_coords[0]:ranked_object_coords[2],
                                ranked_object_coords[1]: ranked_object_coords[3]]

        img_caption = "Rank {}".format(i)
        cv2.imshow(img_caption, ranked_object_img)
        i += 1
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def getImageFromMaskList(image, mask, i):
    img = image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for j in range(image.shape[2]):
        img[:, :, j] = img[:, :, j] * mask[:, :, i]


def getMasksImages(image, results):
    onlyObj = []
    i = 0
    mask = results["masks"]
    # print("mask.shape", mask.shape)
    # print("img.shape", image.shape)
    for i in range(mask.shape[2]):
        img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j in range(image.shape[2]):
            img[:, :, j] = img[:, :, j] * mask[:, :, i]
        onlyObj.append(img)
        i += 1
    return onlyObj


def generateObjects(img, sal_map, rank_to_show, results):
    gaussian_kernel_array = makeGaussian(9)
    gaussian1d = gaussian_kernel_array.ravel()

    masks = getMasksImages(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), results)
    salMasks = getMasksImages(cv2.cvtColor(sal_map, cv2.COLOR_RGB2BGR), results)

    # Show Image Masks
    # for mask_img in masks:
    #     cv2.imshow("Image Mask", mask_img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Get rois and labels
    objs = []
    ids = results['class_ids']
    i = 0
    for r in results['rois']:
        # obj = (roi, call_is, image mask, saliency mask)
        obj = (r, ids[i], masks[i], salMasks[i])
        objs.append(obj)
        i += 1

    # Generate Object Saliency Ranking
    if not GAUSSIAN:
        maxObj, indexObj = rankProductsWithSaliency(objs, np.zeros(1), sal_map, img)
    else:
        maxObj, indexObj = rankProductsWithSaliency(objs, gaussian1d, sal_map, img)

    # dictObjEntropies = dict(objectEntropies)
    # sortedObjEntropies = sorted(dictObjEntropies.items(), key=operator.itemgetter(1), reverse=True)
    objectEntropies.sort(key=operator.itemgetter(1), reverse=True)

    # print("\nChecking for duplicate objects...\n")
    # duplicateObjects(img)

    if rank_to_show == -1:
        salient_object = getObjectWithIndex(indexObj)
    else:
        indexObj, salient_object = getObjectWithRank(rank_to_show, img, objectEntropies)

    salient_object_roi = salient_object[1][0]
    salientObjectImg = img[salient_object_roi[0]:salient_object_roi[2],
                       salient_object_roi[1]: salient_object_roi[3]]

    # object_class = getClassName(salient_object)
    # print("\n-----------------------------------------\n")
    # print("The most salient product is ", object_class)
    # cv2.imshow("Most Salient", salientObjectImg)

    # Create and return List of final objects
    # (index, obj, entropy, rank)  -> obj is (roi, class_id, mask), ranked by entropies
    final_objects = []
    i = 0
    for obj_entropies in objectEntropies:
        final_object = (obj_entropies[0], getObjectWithIndex(obj_entropies[0]), obj_entropies[1], i, obj_entropies[2])
        final_objects.append(final_object)
        i += 1

    # show_ranked_objects(final_objects, img)
    return final_objects


def returnObjects(input_img, rank_to_show, results):
    generateSegments(input_img, 9)

    # Itti Salinecy Map
    sal_map = returnIttiSaliency(input_img)

    # Spectral Residual Saliency Map
    # sal_map = saliency_models.SpectralResidualSaliency(input_img)

    # Boolean Map Saliency Map
    # sal_map = saliency_models.bms(input_img)
    print("\n\n\n\nComputed Saliency Map\n\n\n\n")

    rankedObjs = generateObjects(input_img, sal_map, rank_to_show, results)
    return rankedObjs
