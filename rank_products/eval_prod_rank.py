import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
import time
import csv
import pandas as pd
from IPython.display import display, HTML

import mask_rcnn as rcnn
import SR as sr

segmentsScores = []
segmentsCoords = []
sortedSegScores = []
sortedSegScoresOut = []


def loadImages(dataset):
    img, depthImg, salImg, name, image_path = 0, 0, 0, 0, 0
    if dataset == 1:
        img = cv2.imread('eval/eval_images/books/3_colour.jpeg')
        depthImg = cv2.imread('eval/eval_images/books/3_depth8.png')
        salImg = cv2.imread('eval/eval_images/books/sal.jpg')
        name = "Books"
        image_path = "/academic_book_no/3_colour.jpeg"
    elif dataset == 2:
        img = cv2.imread('eval/eval_images/shoes/3_colour.jpeg')
        depthImg = cv2.imread('eval/eval_images/shoes/3_depth8.png')
        salImg = cv2.imread('eval/eval_images/shoes/sal.jpg')
        name = "Footwear"
        image_path = "/footwear_no/3_colour.jpeg"
    return img, depthImg, salImg, name, image_path


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


def point_in_roi(x, y, object_roi):
    # print(object_roi[0], "<", x, "<", object_roi[2])
    # print(object_roi[1], "<", y, "<", object_roi[3])
    return object_roi[0] <= y <= object_roi[2] and object_roi[1] <= x <= object_roi[3]


def getObjectIndex(x, y, products):
    for final_object in products:
        object_roi = final_object[1][1][0]
        if point_in_roi(x, y, object_roi):
            return final_object
    return -1


def getScaledCoordinates(img, inMaxX, inMaxY, tempX, tempY):
    "scales the coordinate in the csv file to match the size of the input img"

    maxX = int(img.shape[1])
    maxY = int(img.shape[0])

    tempX = (int)(tempX * (maxX / inMaxX))
    tempY = (int)(tempY * (maxY / inMaxY))
    return tempX, tempY


def zerolistmaker(n):
    # https://stackoverflow.com/questions/8528178/list-of-zeros-in-python
    listofzeros = [0] * n
    return listofzeros


def rankProductsFromCSV(csvName, task, img, products):
    df = pd.read_csv('eval/click_movement.csv')

    print("id", id)
    image = df['image'] == id
    df = df[image]

    # print(df.head())
    # print(df.describe())

    wasted_click = 0
    total_clicks = 0

    for index, row in df.iterrows():
        total_clicks += 1
        # Get X and Y coords of the click
        coords = df.at[index, 'click']
        chunks = coords.split(',')
        tempX = int(chunks[0].strip(',()'))
        tempY = int(chunks[1].strip(',()'))

        # 299,255 are the dimensions used in the experiment represented in csv.
        X, Y = getScaledCoordinates(imgT, 300, 180, tempX, tempY)
        # print(tempX, tempY, "scaled to", X, Y)

        # Get Index of Object present at that point
        tempObj = getObjectIndex(X, Y, products)

        if tempObj != -1:
            tempObjIndex = int(tempObj[0])
            # print("Object with index:", tempObjIndex)
            objectScores[tempObjIndex] = objectScores[tempObjIndex] + 1
        else:
            # print("Point (", X, ",", Y, ") not in any object")
            wasted_click += 1

    print("Clicks on no products:", wasted_click, "out of", total_clicks)

    new_products = []
    for product in objects:
        new_product = (product[0], product[1], product[2], objectScores[product[0]])
        new_products.append(new_product)

    return new_products


# imgT, imgD, imgS, name, id = loadImages(2)
cots_path = '/footwear_no/3_colour.jpeg'
imgT = cv2.imread('eval/cots_2' + cots_path)
id = cots_path

objects = sr.returnObjects(imgT, -1, rcnn.detect_objects('eval/cots_2' + cots_path), True)

objectScores = []
objectScores = zerolistmaker(len(objects))

objects.sort(key=operator.itemgetter(0))

print("Products: ")
for product in objects:
    print("\t", product)

prodScores = rankProductsFromCSV('click_movement.csv', 1, imgT, objects)

prodScores.sort(key=operator.itemgetter(3), reverse=True)

# print("objectScores", objectScores)
print("Final Product Ranking\n")
i = 0
for final_product in prodScores:
    object_class = rcnn.getClassNameByObject(final_product[1])
    print("Rank", i, ": [", final_product[0], "]", object_class, " clicks:",
          final_product[3], " sal_score:", final_product[2])
    i += 1
