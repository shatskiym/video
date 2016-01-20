import cv2
import numpy as np
import os

templates = os.listdir('templates')
templs = []

for template in templates:
    templs.append(cv2.imread('templates//'+template,0))
    
    
cap = cv2.VideoCapture('video//4.flv')

while (cap.isOpened()):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for templ in templs:
        w,h = templ.shape[::-1]
        res = cv2.matchTemplate(img_gray,templ,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(min_val)
        print(max_val)
        print(min_loc)
        print(max_loc)
        if max_val<0.85:
            break
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
    cv2.imshow('cap1',img)
    if cv2.waitKey(10) == 0x1b:
        break

cap.release()
cv2.destroyAllWindows()

