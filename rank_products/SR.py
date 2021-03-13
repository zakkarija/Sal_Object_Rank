import itertools

import cv2
import numpy as np
import math
import operator

import matcher

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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


def calculatePixelFrequency(img):
    flt = img.flatten()
    unique, counts = np.unique(flt, return_counts=True)
    pixelsFrequency = dict(zip(unique, counts))

    return pixelsFrequency


def calculateEntropy(img, w):
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

        entropy = entropy * wt * 1

    return entropy


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


objectEntropies = []
indexed_objects = []


# def findMostSalientObject(objs, kernel, dws):
def findMostSalientObject(objs, kernel):
    # objs is array of obj (roi, class_id)
    maxEntropy = 0
    index = 0
    i = 0
    for obj in objs:
        # print("roi", roi, kernel[i], dws)
        print("roi", obj[0])
        entropy = calculateEntropy(obj[0], kernel[i])
        # objectEntropies is list of  (index, entropy)
        Tup = (i, entropy)
        objectEntropies.append(Tup)
        # objects is list of (index, object)
        indexed_object = (i, obj)
        indexed_objects.append(indexed_object)
        if entropy > maxEntropy:
            maxEntropy = entropy
            index = i
        i += 1
    return maxEntropy, index


def getClassName(salient_object):
    return class_names[salient_object[1][1]]


def duplicateObjects(img):
    for objA, objB in itertools.combinations(indexed_objects, 2):
        if objA[0] != objB[0]:
            roi = objA[1][0]
            roi2 = objB[1][0]
            match1 = img[roi[0]:roi[2], roi[1]:roi[3]]
            match2 = img[roi2[0]:roi2[2], roi2[1]:roi2[3]]

            # if matcher.template_matcher(match1, match2):
            if matcher.hash_matcher(match1, match2):
                print("Object Matches: ", getClassName(getObjectWithIndex(objA[0])),
                      " with ", getClassName(getObjectWithIndex(objA[1])))
                # cv2.imshow("Object Matches1", match1)
                # cv2.imshow("Object Matches2", match2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                print("Not Match found between ", objA[0], ":", getClassName(getObjectWithIndex(objA[0])),
                      " and ", objA[0], ":", getClassName(getObjectWithIndex(objA[1])))


def getObjectWithIndex(index):
    print('objects', indexed_objects)
    for objct in indexed_objects:
        # objct is  index , roi
        if int(objct[0]) == int(index):
            return objct


def getObjectWithRank(rank, img, obj_entropies):
    rank_iter = 0
    for obj in obj_entropies:
        if int(rank) == int(rank_iter):
            print("object returned: ", obj)
            return obj[0], getObjectWithIndex(obj[0])
        rank_iter += 1


def generateObjects(img, rank_to_show, results):
    # Generate Gaussian Weights
    gaussian_kernel_array = makeGaussian(9)
    gaussian1d = gaussian_kernel_array.ravel()

    # Get rois and labels
    objs = []
    ids = results['class_ids']
    i = 0
    for r in results['rois']:
        obj = (r, ids[i])
        objs.append(obj)
        i += 1

    # Generate Object Saliency Ranking
    maxObj, indexObj = findMostSalientObject(objs, gaussian1d)
    dictObjEntropies = dict(objectEntropies)
    sortedObjEntropies = sorted(dictObjEntropies.items(), key=operator.itemgetter(1), reverse=True)

    duplicateObjects(img)

    if rank_to_show == -1:
        salient_object = getObjectWithIndex(indexObj)
    else:
        indexObj, salient_object = getObjectWithRank(rank_to_show, img, sortedObjEntropies)

    salient_object_roi = salient_object[1][0]
    salientObjectImg = img[salient_object_roi[0]:salient_object_roi[2],
                       salient_object_roi[1]: salient_object_roi[3]]

    object_class = getClassName(salient_object)
    print("mostSalientObject is ", object_class)
    cv2.imshow("mostSalientObject", salientObjectImg)
    cv2.waitKey()

    # Create and return List of final objects
    # (index, obj, entropy)  -> obj is (roi, class_id), ranked by entropies
    final_objects = []
    for obj_entropies in sortedObjEntropies:
        final_object = (obj_entropies[0], getObjectWithIndex(obj_entropies[0]), obj_entropies[1])
        final_objects.append(final_object)
    return final_objects


def returnObjects(input_img, rank_to_show, results):
    rankedObjs = generateObjects(input_img, rank_to_show, results)
    return rankedObjs
