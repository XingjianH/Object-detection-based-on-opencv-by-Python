
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt



def miniprojet(img1,img2):
    # convert the frames to grayscale
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    grayA=cv2.GaussianBlur(grayA,(21,21),0)
    grayB=cv2.GaussianBlur(grayB,(21,21),0)
    diff_image = cv2.absdiff(grayA, grayB)

    # perform image thresholding
    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
 
    # plot image after thresholding

    # plt.show()

    # apply image dilation
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(thresh,kernel,iterations = 1)



    # find contours
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]

    valid_cntrs = []

    for i,cntr in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cntr)
        if  (cv2.contourArea(cntr) >= 50):
            valid_cntrs.append(cntr)

    # count of discovered contours        

    dmy = img2.copy()

    cv2.drawContours(dmy, valid_cntrs, -1, (0,200,0), -1)
   

    #kernel=np.ones((5,5),np.uint8)
    #dmy=cv2.erode(dmy,kernel,iterations=1)

    return dmy







# frame0 = False

cap = cv2.VideoCapture("428197376-1-16.mp4")
i = 0

frame0 = None

while (1):
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    res = miniprojet(frame1,frame2)


    cv2.imshow('f',res)

    if cv2.waitKey(50) == 27:
        break


cap.release()

cv2.destroyAllWindows()