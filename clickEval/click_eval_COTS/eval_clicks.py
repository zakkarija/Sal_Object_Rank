import cv2
import numpy as np
import math
# import scipy.stats as st
import matplotlib.pyplot as plt
import operator
import time
import csv
import pandas as pd
from IPython.display import display, HTML

segmentsScores = []
segmentsCoords = []
sortedSegScores = []
sortedSegScoresOut = []


def loadImages(dataset):
    if dataset == 1:
        img = cv2.imread('eval_images/books/3_colour.jpeg')
        depthImg = cv2.imread('eval_images/books/3_depth8.png')
        salImg = cv2.imread('eval_images/books/sal.jpg')
        name = "Books"
        image_path = "/academic_book_no/3_colour.jpeg"
    elif dataset == 2:
        img = cv2.imread('eval_images/shoes/3_colour.jpeg')
        depthImg = cv2.imread('eval_images/shoes/3_depth8.png')
        salImg = cv2.imread('eval_images/shoes/sal.jpg')
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


def getSegmentIndex(img, x, y, n):
    # https://math.stackexchange.com/questions/528501/how-to-determine-which-cell-in-a-grid-a-point-belongs-to

    a = int(img.shape[1])
    b = int(img.shape[0])

    i = math.floor((x * n) / a)
    j = math.floor((y * n) / b)

    segmentIndex = i + j * n
    return segmentIndex, i, j


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


def segmentsFromCsv(csvName, task, img):
    df = pd.read_csv('click_movement.csv')

    # cleanTime = (df['TimeUsed'] > 100) & (df['TimeUsed'] < 50000)
    # image = df['Image'] == id
    # task = df['Task'] == task
    # df = df[image & task]
    print("id", id)
    image = df['image'] == id
    df = df[image]

    print(df.head())
    print(df.describe())

    for index, row in df.iterrows():
        coords = df.at[index, 'click']
        chunks = coords.split(',')
        tempX = int(chunks[0].strip(',()'))
        tempY = int(chunks[1].strip(',()'))
        # tempX = df.at[index, 'X']
        # tempY = df.at[index, 'Y']
        # 299,255 are the dimensions used in the experiment represented in csv.
        tempX, tempY = getScaledCoordinates(imgT, 300, 180, tempX, tempY)
        tempSegment = getSegmentIndex(img, tempX, tempY, numSegmentsRow)[0]
        segmentsScores[tempSegment] = segmentsScores[tempSegment] + 1

    i = 0
    for segScore in segmentsScores:
        tempScoreTup = (i, segScore)
        sortedSegScoresOut.append(tempScoreTup)
        i = i + 1

    return sortedSegScoresOut


def generateHeatMap(img, mode, sortedSegScores, SegmentsCoords):
    font = cv2.FONT_HERSHEY_SIMPLEX
    printIndex = 0
    set = int(0.25 * len(sortedSegScores))
    color = (0, 0, 0)

    saraListOut = []

    for ent in sortedSegScores:
        if (mode == 0):
            color = (255, 255, 255)
            t = 4
        elif (mode == 1):
            if (printIndex + 1 <= set):
                color = (0, 0, 255)
                t = 8
            elif (printIndex + 1 <= set * 2):
                color = (0, 128, 255)
                t = 6
            elif (printIndex + 1 <= set * 3):
                color = (0, 255, 255)
                t = 4
            elif (printIndex + 1 <= set * 4):
                color = (0, 250, 0)
                t = 2

        x1 = segmentsCoords[ent[0]][1]
        y1 = segmentsCoords[ent[0]][2]
        x2 = segmentsCoords[ent[0]][3]
        y2 = segmentsCoords[ent[0]][4]
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        cv2.putText(img, str(printIndex), (x - 2, y), font, .5, color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
        saraTuple = (printIndex, ent[0])
        # print("\nSara Tuple: " + str(saraTuple))
        saraListOut.append(saraTuple)
        printIndex += 1

    return img, saraListOut


numSegmentsRow = 9
segments = []
dws = []
saraList = []

imgT, imgD, imgS, name, id = loadImages(2)
# imgT =

# segments = generateSegments(imgT, numSegmentsRow)

# segIndex = returnSegmentIndex(imgT, Xtarget, Ytarget, numSegmentsRow)
# cv2.imshow("seg", segments[segIndex[0]])
# cv2.waitKey()

segmentsScores = zerolistmaker(numSegmentsRow * numSegmentsRow)
generateSegments(imgS, numSegmentsRow)
segmentsZeros = zerolistmaker(numSegmentsRow * numSegmentsRow)

segScores = segmentsFromCsv('click_movement.csv', 1, imgT)

dictScores = dict(segScores)
sortedSegScores = sorted(dictScores.items(), key=operator.itemgetter(1), reverse=True)

imgT, saraListOut = generateHeatMap(imgT, 1, sortedSegScores, segmentsCoords)

# print(segmentsScores)
# print(segmentsCoords[sortedSegScores[0][0]][1])
print("Rank: \n")

outS1 = cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)
plt.imshow(outS1)
plt.savefig('output.png')

df = pd.DataFrame(saraListOut, columns=['Segment', 'Rank'])
# df1.columns = ["Segment", "Rank"]
# pd.set_option('display.max_rows', 81)
# df
df.to_csv("outClicks.csv", index=False)
