#!/usr/bin/python
# Import required packages
import cv2
import pytesseract
import sys

# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'

# Read image from which text needs to be extracted
imagename = sys.argv[1]
pfxd = imagename.split(".")[0]
d = pfxd.split("/")
pfx = "work/%s" % d[-1]
img = cv2.imread(imagename)

# img = cv2.resize(img, None, fx=2, fy=2)

# Preprocessing the image starts

# neg = cv2.bitwise_not(img)

# cv2.imwrite('negative.png', neg)

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(gray,(5,5),0)
# Perform threshold
ret, thresh1 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
# ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)

# gray_three = cv2.merge([thresh1,thresh1,thresh1])

# thresh2 = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

cv2.imwrite("%s.grayscale.png" % pfx, thresh1)
# cv2.imwrite('adaptive.png', adaptive_threshold)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

cv2.imwrite("%s.dilation.png" % pfx, dilation)

# Creating a copy of image
im2 = img.copy()

# A text file is created and flushed
file = open("%s.recognized.txt" % pfx, "w+")
file.write("")
file.close()


pois = [
    [2138, 58, "energy"],
    [2191, 1009, "loadout"],
    [2184, 90, "wave"],
    [1682, 55, "gold"],
    [1945, 55, "munitions"],
    [2223, 55, "oil"],
    [784, 454, "defense rating"],
    [1119, 457, "outpost level"],
    [789, 349, "player"],
    [1070, 372, "player number"],
    [2184, 916, "moveout"],
]
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = thresh1[y : y + h, x : x + w]

    # Open the file in append mode
    file = open("%s.recognized.txt" % pfx, "a")

    cfile = "crop_%d_%d.png" % (x, y)

    text = pytesseract.image_to_string(cropped, config="--oem 3 --psm 7")
    for poi in pois:
        if poi[0] > x and poi[0] < (x + w) and poi[1] > y and poi[1] < (y + h):
            # Apply OCR on the cropped image
            cfile = "crop_%s_%d_%d.png" % (poi[2], x, y)
            text = poi[2] + ":" + text
            # cv2.imwrite(cfile, cropped)

    # Appending the text into file
    file.write("x %d y %d w %d h %d\n" % (x, y, w, h))
    file.write(text)
    file.write("\n\n")

    # Close the file
    file.close

cv2.imwrite("%s.rectangles.png" % pfx, im2)
