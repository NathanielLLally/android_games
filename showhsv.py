import cv2
import numpy as np
import sys

file = sys.argv[1]

image = cv2.imread(file)
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])
##blue = cv2.inRange(hsvImage, np.array([200, 35, 60]), np.array([230,105,110]))
blue = cv2.inRange(hsvImage, np.array([100, 50, 50]), np.array([120,255,255]))
red_low = cv2.inRange(hsvImage, np.array([0, 90, 75]), np.array([10,255,255]))
red_high = cv2.inRange(hsvImage, np.array([175, 90, 75]), np.array([185,255,255]))
red = red_low + red_high
purple = cv2.inRange(hsvImage, np.array([120, 50, 50]), np.array([140,255,255]))

bmask = cv2.bitwise_and(image,image, mask = blue)
rmask = cv2.bitwise_and(image,image, mask = red)
pmask = cv2.bitwise_and(image,image, mask = purple)

blue = [
[91,50,18],
[100,81,63],
[99,48,6],
[95,72,55],
[58,35,18],
[64,26,0],
[69,35,7],
[50,17,6],
]

red = [
[40,39,68],
[30,32,60],
[0,0,60],
[28,26,87],
[7,5,60],
[53,53,86],
[40,38,70],
[9,9,34]
]

redno = [
[15,17,25],
[10,10,13],
[12,13,18],
[5,6,12],
[22,21,40],
[33,36,48],
[38,44,60],
[39,49,75]
]

purple = [
[62,46,54],
[71,8,42],
[100,43,73],
[81,6,45],
[200,50,80]
]

white = [
[139,170,186],
[101,148,170],
[101,148,170],
[81,142,177],
]

l = white
for v in l:
    b = np.uint8([[v]])
    hsvb = cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
    print( hsvb )


cv2.imshow('Original image',image)
cv2.imshow('HSV image', hsvImage)
cv2.imshow('Blue Masked image', bmask)
cv2.imshow('Red Masked image', rmask)
cv2.imshow('Purple Masked image', pmask)

  

cv2.waitKey(0)
cv2.destroyAllWindows()
