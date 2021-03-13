import itertools

import cv2
import numpy as np
import math
import scipy.stats as st
import matplotlib.pyplot as plt
import operator
import time
import os
from enum import Enum
import pandas as pd

# Import Akisato Kimura <akisato@ieee.org> implementation of Itti's Saliency Map Generator
# Original Source: https://github.com/akisatok/pySaliencyMap
import matcher
import pySaliencyMap

# -------------------------------------------------
# Start Global Variables
# -------------------------------------------------

segmentsEntropies = []
segmentsCoords = []

segDim = 9
segments = []
gtSegments = []
dws = []
saraList = []

evalList = []
labelsEvalList = ['Image', 'Index', 'Rank', 'Quartile', 'isGT', 'Outcome']

outcomeList = []
labelsOutcomeList = ['Image', 'FN', 'FP', 'TN', 'TP']

dataframeCollection = {}
errorCount = 0

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


# -------------------------------------------------
# SaRa Initial Functions
# -------------------------------------------------

def generateSegments(img, segCount, depth=None):
    segments = []
    segmentCount = segCount
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

    return segments


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

def calculatePixelFrequency(img):
    flt = img.flatten()
    unique, counts = np.unique(flt, return_counts=True)
    pixelsFrequency = dict(zip(unique, counts))

    return pixelsFrequency


def calculateEntropy(img, w, dw):
    flt = img.flatten()

    c = flt.shape[0]
    totalPixels = 0
    tprob = 0
    sumOfProbs = 0
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

    return (entropy)


objectEntropies = []
objects = []


def findMostSalientObject(objs, kernel, dws):
    # objs is array of obj (roi, class_id)
    maxEntropy = 0
    index = 0
    i = 0
    for obj in objs:
        # print("roi", roi, kernel[i], dws)
        print("roi", obj[0])
        entropy = calculateEntropy(obj[0], kernel[i], dws[i])
        # objectEntropies is list of  (index, entropy)
        Tup = (i, entropy)
        objectEntropies.append(Tup)
        # objects is list of (index, object)
        object = (i, obj)
        objects.append(object)
        if entropy > maxEntropy:
            maxEntropy = entropy
            index = i
        i += 1
    return maxEntropy, index


def findMostSalientSegment(segments, kernel, dws):
    maxEntropy = 0
    index = 0
    i = 0
    for segment in segments:
        # tempEntropy = calculateEntropy(segment, kernel[i])
        # print("roi", roi, kernel, dws)
        tempEntropy = calculateEntropy(segment, kernel[i], dws[i])
        tempTup = (i, tempEntropy)
        segmentsEntropies.append(tempTup)
        if tempEntropy > maxEntropy:
            maxEntropy = tempEntropy
            index = i
        i += 1

    return maxEntropy, index


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


def get_last_non_zero_index(d, default=None):
    rev = (len(d) - idx for idx, item in enumerate(reversed(d), 1) if item)
    return next(rev, default)


def get_first_non_zero_indox(list):
    return next((i for i, x in enumerate(list) if x), None)


def genDepthWeights(dSegments, depthMap):
    histD, binsD = np.histogram(depthMap, 256, [0, 256])
    firstNZ = get_first_non_zero_indox(histD)
    lastNZ = get_last_non_zero_index(histD)
    mid = (firstNZ + lastNZ) / 2

    for seg in dSegments:
        hist, bins = np.histogram(seg, 256, [0, 256])
        # print(hist)
        dw = 0
        ind = 0
        for s in hist:
            if (ind > mid):
                dw = dw + (s * (1))
            ind = ind + 1
        dws.append(dw)

    return dws


def genBlankDepthWeight(dSegments):
    for seg in dSegments:
        dw = 1
        dws.append(dw)
    return dws


def generateHeatMap(img, mode, sortedSegScores, SegmentsCoords):
    # mode0 prints just a white grid
    # mode1 prints prints a colour-coded grid

    font = cv2.FONT_HERSHEY_SIMPLEX
    printIndex = 0
    set = int(0.25 * len(sortedSegScores))
    color = (0, 0, 0)

    saraListOut = []

    # rank = 0

    for ent in sortedSegScores:
        quartile = 0
        if (mode == 0):
            color = (255, 255, 255)
            t = 4
        elif (mode == 1):
            if (printIndex + 1 <= set):
                color = (0, 0, 255)
                t = 8
                quartile = 4
            elif (printIndex + 1 <= set * 2):
                color = (0, 128, 255)
                t = 6
                quartile = 3
            elif (printIndex + 1 <= set * 3):
                color = (0, 255, 255)
                t = 4
                quartile = 2
            elif (printIndex + 1 <= set * 4):
                color = (0, 250, 0)
                t = 2
                quartile = 1

        x1 = segmentsCoords[ent[0]][1]
        y1 = segmentsCoords[ent[0]][2]
        x2 = segmentsCoords[ent[0]][3]
        y2 = segmentsCoords[ent[0]][4]
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)

        cv2.putText(img, str(printIndex), (x - 2, y), font, .5, color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)

        # print("\nText Index:" + str(printIndex))
        # print("Rank:" + str(ent[0]))
        # print("Quartile:" + str(quartile))

        # cv2.putText(gtSara, str(printIndex), (x-2,y), font, .5, (255,255,255) ,1 ,cv2.LINE_AA)
        # cv2.rectangle(gtSara, (x1,y1), (x2,y2), color , t)

        # saraTuple = (index, rank, quartile)
        saraTuple = (ent[0], printIndex, quartile)
        # print("\nSara Tuple: " + str(saraTuple))
        saraListOut.append(saraTuple)
        printIndex += 1

    # print(saraListOut)
    return img, saraListOut


def printObjects(entropies, rois, img):
    saraListOut = []
    printIndex = 0
    # rois is list of rois
    # entropies is list of (index, entropy)
    for entropy in entropies:
        x1 = rois[entropy[0]][0]
        y1 = rois[entropy[0]][1]
        x2 = rois[entropy[0]][2]
        y2 = rois[entropy[0]][3]
        # saraTuple is index and rank
        saraTuple = (entropy[0], printIndex)
        saraListOut.append(saraTuple)
        printIndex += 1

    # obj_roi = getObjectWithRank(saraListOut, rois, 0, img)
    # cv2.imshow("obj_roi", obj_roi)
    # cv2.waitKey()

    return saraListOut


sortedObjEntropies = []


def getClassName(salient_object):
    return class_names[salient_object[1][1]]


def duplicateObjects(img):
    for objA, objB in itertools.combinations(objects, 2):
        if objA[0] != objB[0]:
            roi = objA[1][0]
            roi2 = objB[1][0]
            match1 = img[roi[0]:roi[2], roi[1]:roi[3]]
            match2 = img[roi2[0]:roi2[2], roi2[1]:roi2[3]]
            
            # if matcher.template_matcher(match1, match2):
            if matcher.hash_matcher(match1, match2):
                print("Object Matches: ", getClassName(getObjectWithIndex(objA[0], img)),
                      " with ", getClassName(getObjectWithIndex(objA[1], img)))
                cv2.imshow("Object Matches1", match1)
                cv2.imshow("Object Matches2", match2)
                cv2.waitKey()
                cv2.destroyAllWindows()
            else:
                print("Not Match found between ", objA[0], " and ", objB[0])


def generateSaRa(img, texSegments, results, rank_to_show):
    # Generate Gaussian Weights
    gaussian_kernel_array = makeGaussian(segDim)
    gaussian1d = gaussian_kernel_array.ravel()

    # Generate Depth scores
    # dSegments = generateSegments(gt, segDim)
    dws = genBlankDepthWeight(texSegments)

    # Generate Saliency Ranking
    maxH, index = findMostSalientSegment(texSegments, gaussian1d, dws)
    dictEntropies = dict(segmentsEntropies)
    sortedEntropies = sorted(dictEntropies.items(), key=operator.itemgetter(1), reverse=True)

    # Get rois and labels
    objs = []
    ids = results['class_ids']
    i = 0
    for r in results['rois']:
        obj = (r, ids[i])
        objs.append(obj)
        i += 1

    # Generate Object Saliency Ranking
    maxObj, indexObj = findMostSalientObject(objs, gaussian1d, dws)
    dictObjEntropies = dict(objectEntropies)
    sortedObjEntropies = sorted(dictObjEntropies.items(), key=operator.itemgetter(1), reverse=True)

    # print("maxObj", maxObj)
    # print("indexObj", indexObj)
    # print("sortedObjEntropies", sortedObjEntropies)
    # print("rank_to_show", rank_to_show)

    duplicateObjects(img)

    if rank_to_show == -1:
        salient_object = getObjectWithIndex(indexObj, img)
    else:
        indexObj, salient_object = getObjectWithRank(rank_to_show, img, sortedObjEntropies)

    salient_object_roi = salient_object[1][0]
    salientObjectImg = img[salient_object_roi[0]:salient_object_roi[2],
                       salient_object_roi[1]: salient_object_roi[3]]

    object_class = getClassName(salient_object)
    print("mostSalientObject is ", object_class)
    cv2.imshow("mostSalientObject", salientObjectImg)
    cv2.waitKey()

    # Generate Heatmap and display it
    texOut, saraListOut = generateHeatMap(img, 1, sortedEntropies, segmentsCoords)
    return texOut, saraListOut, indexObj


def getIndexWithRank(saraListOutput, rank, img):
    # sal_tuple is (index, rank, quartile)
    for sal_tuple in saraListOutput:
        # print(sal_tuple)
        if sal_tuple[1] == int(rank):
            for coords in segmentsCoords:
                # coords is index, x1, y1, x2, y2
                if coords[0] == sal_tuple[0]:
                    x1 = coords[1]
                    y1 = coords[2]
                    x2 = coords[3]
                    y2 = coords[4]
                    roi = img[y1:y2, x1: x2]
                    return roi


def getObjectWithIndex(index, img):
    print('objects', objects)
    for coords in objects:
        # coords is  index , roi
        if int(coords[0]) == int(index):
            # x1 = coords[1][0][0]
            # y1 = coords[1][0][1]
            # x2 = coords[1][0][2]
            # y2 = coords[1][0][3]
            # roi = img[y1:y2, x1: x2]
            # to get roi coords[1][0]
            return coords


def getObjectWithRank(rank, img, objEntropies):
    rank_iter = 0
    for obj in objEntropies:
        if int(rank) == int(rank_iter):
            print("object returned: ", obj)
            return obj[0], getObjectWithIndex(obj[0], img)
        rank_iter += 1

    # -------------------------------------------------
    # Evaluation Functions
    # -------------------------------------------------


def returnSARA(inputImg, rank_to_show, results):
    texSegments = generateSegments(returnIttiSaliency(inputImg), 9)
    saraOutput, saraListOutput, indexObj = generateSaRa(inputImg, texSegments, results, rank_to_show)

    return saraOutput, saraListOutput
