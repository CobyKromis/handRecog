import cv2 as cv
import sys


img = cv.imread(r"C:\Users\cobyk\OneDrive\Fall 2021\plab3\opencv02test.jpg")
if img is None:
    sys.exit("Couldn't read image")

cv.imshow("Display Window", img)
k = cv.waitKey(0)

if k==ord("s"):
    cv.imwrite("opencv02test.jpg", img)

