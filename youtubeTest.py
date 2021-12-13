import cv2 as cv
import numpy as np
import math

capture = cv.VideoCapture(0)

while(capture.isOpened()):

    #captures webcam
    ret, img = capture.read()

    #creates rectangle on webcam feed which acts as a zone for user's hand#
    cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
    imgCrop = img[100:300, 100:300]

    #used to created black/white version of webcam feed (only for hand zone)#
    gray = cv.cvtColor(imgCrop, cv.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv.GaussianBlur(gray, value, 0)
    _, thresh1 = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cv.imshow('Thresholded', thresh1)

    (version, _, _) = cv. __version__.split('.')

    if version == '3': 
        image, contours, hierarchy = cv.findContours(thresh1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    elif version =='4':
        contours, hierarchy = cv.findContours(thresh1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    #detects and draws contours of webcam feed on secondary windows#
    #creates accurate map of user's hand when it is in the hand zone#   
    cnt = max(contours, key = lambda x: cv.contourArea(x))   
    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(imgCrop, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv.convexHull(cnt)
    drawing = np.zeros(imgCrop.shape, np.uint8)
    cv.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    hull = cv.convexHull(cnt, returnPoints = False)
    defects = cv.convexityDefects(cnt, hull)
    defectCount = 0
    cv.drawContours(thresh1, contours, -1, (0, 255, 0), 3)

    #calculates angle of defects and decides if they are to be interpreted as a finger#
    for i in range(defects.shape[0]):
        s, e, f, d = defects [i, 0]

        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        #if angle of defect is above 90 degrees it is considered to be a defect caused by a finger#
        if angle <= 90: 
            defectCount += 1
            cv.circle(imgCrop, far, 1, [0, 0, 255], -1)
    
        cv.line(imgCrop, start, end, [9, 255, 0], 2)

    #conditional variable to decide if 1 or 0 fingers are being held up#
    check1 = int(a) - int(b)

    #based on defectCount text will be displayed based on how many fingers are detected#
    #special case for 0/1 fingers are shown in final two elif statements with check1 above used as condition#
    if defectCount == 1: 
        cv.putText(img, "2 Finger Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defectCount == 2:
       cv.putText(img, "3 Fingers Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defectCount == 3:
       cv.putText(img, "4 Fingers Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defectCount == 4:
       cv.putText(img, "5 Fingers Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defectCount == 0 and check1 < 8:
       cv.putText(img, "1 Fingers Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif defectCount == 0:
       cv.putText(img, "0 Fingers Detected", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, 2)

    cv.imshow('Gesture', img)
    all_img = np.hstack((drawing, imgCrop))
    cv.imshow('Contours', all_img)

    #once user presses 'Q' on keyboard while loop will break and program will terminate#
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

#program termination procedures#
capture.release()
cv.destroyAllWindows()