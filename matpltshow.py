import cv2 as cv
import matplotlib.pyplot as plt
import sys

file = sys.argv[1]

img = cv.imread(cv.samples.findFile(file), 0)
plt.imshow(img)
plt.show()
