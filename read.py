import pytesseract
import sys
import os
import cv2

file = sys.argv[1]
img = cv2.imread(file)

whitelist = ""
chars= "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_."
if len(whitelist) > 0:
    chars = whitelist
tconf = "--oem 3 --psm 8 "+ \
         "-c tessedit_char_whitelist=\"%s\"" % chars


text = pytesseract.image_to_data(img, config=tconf, output_type='data.frame')
text = text[text.conf != -1]
print(text)


