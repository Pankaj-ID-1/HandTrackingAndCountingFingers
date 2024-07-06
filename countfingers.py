import cv2
import time
import numpy as np
import handTrackingProjectmodule as htm
import math

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prevtime = 0

detector = htm.handDetector(detectionCon=0.7, maxHands=1)



area = 0
lmlist = []
colorVol = [255, 0, 0]
fingercnt = 0
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=True)
    if len(lmlist) != 0:
        area = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))//100
        if 250 < area < 1000:
            # length, img, lineinfo = detector.findDistance(lmlist, 4, 8, img)
            fingers = detector.fingersUp(lmlist)

            for finger in fingers:
                if finger:
                    fingercnt += 1
            cv2.putText(img, "Count: {:.2f}".format(int(fingercnt)), (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, colorVol)
            fingercnt = 0

    currtime = time.time()
    fps = 1 / (currtime - prevtime)
    prevtime = currtime

    cv2.putText(img, "FPS: {:.2f}".format(int(fps)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Img', img)
    cv2.waitKey(1)