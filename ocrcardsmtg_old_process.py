#!/usr/bin/python
# Import required packages
import cv2 as cv
import cv2 as cv2
import pytesseract
import sys
import subprocess
import os
import time
import threading
import queue
import signal
import re
from pathlib import Path
import numpy as np
import math
import inspect #pass frame to sig handler
from skimage.feature import hog

# TODO: handle skewing
#   contrast / enhance on process_card
#   other methods on process card (threshold, edge, inversion etc..)
#
#    what is best results tesseract for text? white on black or vice versa?
#
#   print result text on image
#    do not overwrite text in image with a higher confidence
#  
#  perform price lookup on text, get title back, perform confidence on match
#   take title from price if confidence above X (n letter difference etc..)
#
#  stop processing those tiles upon match, leave title and prices on screen
#
#  try 
#

scale_percent = 60 # percent of original size
#serial = "FAMVRW9D9HDEHEWS" #BLU Pure
serial="XG9LGEZX6L75QGMJ" #Oppo Realme RMX2001

debug = False
page = 1
num_cards = 4

# Adaptive threshold levels
BKG_THRESH = 30
CARD_THRESH = 30


#  for some reason forking then using subprocess run will block the main process
#
def fork(func=None):
    r_out, w_out = os.pipe()
    r_err, w_err = os.pipe()
    pid = os.fork()
  
    # n greater than 0  means parent process
    if pid > 0:
        print("Parent process and id is : ", os.getpid())
        # Parent process
        os.close(w_out)
        os.close(w_err)

        r1 = os.fdopen(r_out)
        r2 = os.fdopen(r_err)
        for i in range(10):
            # Note that r1.read(), and r2.read() are non-blocking calls
            # You can run this while loop as long as you want.
            print("Read text (sysout):", r1.read())
            print("Read text (syserr):", r2.read())
            time.sleep(0.5)        
    else:
        print("Child process and id is : ", os.getpid(), flush=True)
            # Child process
        os.close(r_out)
        os.close(r_err)

        w1 = os.fdopen(w_out, "w")
        w2 = os.fdopen(w_err, "w")
        sys.stdout = w1
        sys.stderr = w2

        # Note that flush=True is necessary only if you want to ensure the order of messages printed
        # across method1, and method2 is maintained

        # Method 1: Standard Python print messages
        print("Redirected to stdout #2", flush=True)
        print("Redirected to stderr #2", file=sys.stderr, flush=True)

        # Method 2: Using system file descriptors
        stdout_fd = sys.stdout.fileno()
        os.write(stdout_fd, b'Redirected to stdout')

        stderr_fd = sys.stderr.fileno()
        os.write(stderr_fd, b'Redirected to stderr')

        if func != None:
            ret=func()
            print("result of func %s: %d" % (func, ret), flush=True)
        print("Child returning", flush=True)

        # Restore original stdout, and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Close the file descriptors
        w1.close()
        w2.close()
        sys.exit()

class AsyncLineReader(threading.Thread):
    def __init__(self, fd, outputQueue):
        threading.Thread.__init__(self)

        assert isinstance(outputQueue, queue.Queue)
        assert callable(fd.readline)

        self.fd = fd
        self.outputQueue = outputQueue

    def run(self):
        map(self.outputQueue.put, iter(self.fd.readline, ''))

    def eof(self):
        return not self.is_alive() and self.outputQueue.empty()

    @classmethod
    def getForFd(cls, fd, start=True):
        q = queue.Queue()
        reader = cls(fd, q)

        if start:
            reader.start()

        return reader, q

def stdout_reader(proc, q):
    for line in iter(proc.stdout.readline, b''):
        q.put(line.decode('utf-8'))

def stderr_reader(proc, q):
    for line in iter(proc.stderr.readline, b''):
        q.put(line.decode('utf-8'))

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

actualtap = True
def tap(tag):
    if not actualtap:
        print("TAP %s" % tag)
        return
    x, y = pois[tag]
    x = int(x / (scale_percent / 100))
    y = int(y / (scale_percent / 100))
    cmd = "/home/nathaniel/android/sdk/platform-tools/adb -s %s shell input tap %d %d" % (serial, x, y)
    print(cmd)

    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

def checkCurrentFocus():
    cmd = "adb -s %s shell dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp|mHoldScreenWindow'"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    return result.stdout
#
#  for future reference,
#   setting up open camera
#  video mode, hide all gui elements
#  set resolution to that of phone screen size
#  enable digital video stabilization
#  preference_auto_stabilise = true (auto level in gui)
#    images mat be smaller resolution due to rotating and cropping
#
def isRunningOrStart(package="net.sourceforge.opencamera"):
    focus=checkCurrentFocus()
    m = re.match(r".*?%s.*" % package,focus)
    if (m):
        return
    else:
        cmd = "adb -s %s shell monkey -p %s -c android.intent.category.LAUNCHER 1" % (serial,package)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        

def itap(coord):
    (x,y) = coord
    cmd = "/home/nathaniel/android/sdk/platform-tools/adb -s %s shell input tap %d %d" % (serial, x, y)
    print(cmd)

    result = subprocess.run(cmd.split(), capture_output=True, text=True)

#  check for existing scrcpy using v42l-sink
#  if non, start one
#    TODO: handle errors from scrcpy
#
def v4l2sink():

    #  check for process being run externally
    #
    try:
        ps = subprocess.run(['ps', '-ef'], check=True, capture_output=True)
#    print(ps.stdout)
        processNames = subprocess.run(['grep', 'scrcpy'],
                              input=ps.stdout, check=True, capture_output=True)
        pidnfo = subprocess.run(['grep', 'v4l2-sink'],
                              input=processNames.stdout, capture_output=True)
        print(pidnfo.stdout,len(pidnfo.stdout))
        if (len(pidnfo.stdout) > 0):
            return (False,False)
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")
        pass

    #  begin v4l2_sink with scrcpy as Popen thread with threaded output readers
    #
    print("scrcpy is not running, starting process")
#    cmd = "scrcpy -s %s --lock-video-orientation=3 --v4l2-sink=/dev/video0 --stay-awake --power-off-on-close --no-display --show-touches" % serial
    cmd = "scrcpy -s %s --lock-video-orientation=3 --v4l2-sink=/dev/video0 --stay-awake --power-off-on-close --show-touches" % serial
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    pq = queue.Queue()
    t1 = threading.Thread(target=stdout_reader, args=(proc, pq))
    t1.start()
    t2 = threading.Thread(target=stderr_reader, args=(proc, pq))
    t2.start()

    wait = isinstance(proc, subprocess.Popen)
    waited = 0
    timeout = 5
    while wait and waited < timeout:
        try:
            line = pq.get(block=False)
            print('{0}'.format(line), end='')
        #INFO: v4l2 sink started to device: /dev/video0
            m = re.match(r".*?v4l2 sink started to device: (?P<device>[\/\w]+)", line)
    #        m = re.match(r".*?v4l2 sink started to device: ", line)
            if (m):
                print("scrcpy v4l2 sink started using device %s" % m.group('device'))
                wait = False
            #
            #  TODO what are the error strings from scrcpy or v4l2?
            #
        except queue.Empty:
            pass
        time.sleep(0.01) 
        waited += 0.01

    return (proc, pq)

def handler(signum, frame):
    try:
        if (isinstance(proc, subprocess.Popen)):
            proc.terminate()
    except:
        pass
    sys.exit()

def nothing(*arg):
    pass

def setScaleFromTrack(x):
    scale_percent = cv.getTrackbarPos('scale', 'edge')

def get_tile(x,y,w,h, tilesize):
    TILE_W = tilesize[1]
    TILE_H = tilesize[0]

    x1 = int(x / TILE_W)
    x2 = (x + w) / TILE_W

    y1 = int(y / TILE_H)
    y2 = (y + h) / TILE_H

    if int(x2) == x2:
        x2 = int(x2 - 1)
    else:
        x2 = int(x2)

    if int(y2) == y2:
        y2 = int(y2 - 1)
    else:
        y2 = int(y2)

    tw = x2 - x1 + 1
    th = y2 - y1 + 1

    return x1, y1, tw, th

#cardsByTile = {tile: {image:data, text:text, conf:conf}} 

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

cardsByTile = {}
def process_card(crop,bounds,tilesize, page):
    #thresh
    # poi's 
    #tesseract
    w = crop.shape[0]
    h = crop.shape[1]
    pro = 0
    if (h > w):
        pro = h / w
    else:
        pro = w / h
#    area = h * w
    #round

    #  if dimensions are within the size of 1 and 5 tiles
    #  proportion of L to W is within 1.3 to 1.5 (typical card)
    #  attempt a few methods for processing to OCR with tesseract
    #  return (text, confidence) of the method with highest confidence
    #  save all images used by page, tile, and type in pwd/cards
    #
    pros = "%.1f" % pro
    if ((h > tilesize[0] and h < tilesize[0] * 5) and
        (w > tilesize[1] and w < tilesize[1] * 5) and 
        (pros == "1.3" or pros == "1.4" or pros == "1.5")):
        tile = get_tile(bounds[0], bounds[1], w, h, tilesize)


        #  go for white text on darker background
        #
        crop = cv.bitwise_not(crop)

        #  correct skew

        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv.threshold(cv.cvtColor(crop,cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

  # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))

    # Applying dilation on the threshold image
        dilation = cv.dilate(thresh, rect_kernel, iterations=1)

    # Finding contours
        contours, hierarchy = cv.findContours(
            dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )

#        cv.drawContours(t2, contours, 0, (0, 255, 0), 1)


        # grab the (x, y) coordinates of all pixel values that
        # are greater than zero, then use these coordinates to
        # compute a rotated bounding box that contains all
        # coordinates
        coords = np.column_stack(np.where(thresh > 0))
        crect = cv.minAreaRect(coords)
#        box=cv.boxPoints(crect)
#        box=np.int0(box)
#        cv.drawContours(crop, [box], -1, (0, 255, 0), 1)

        angle = crect[-1]
        oangle = angle
        rw, rh = crect[1]
        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = -angle

        mangle = angle

        #https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        #
        if (rw > rh):
            pass
#            angle+=180
        else:
            angle+=90
        

        # rotate the image to deskew it
        (h, w) = crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(crop, M, (w, h),
            flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)



        title = rotated[0: int(h/27 * 5), 0: w]
#        cv.rectangle(img=tt2, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)


        ilines = rotated.copy()
        if (True):


           # crop outer black edge with pad 
            tthresh = cv.threshold(cv.cvtColor(title,cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            erode = cv.erode(tthresh, rect_kernel, iterations=1)
            dilation = cv.dilate(erode, rect_kernel, iterations=1)
#            tthresh = cv.adaptiveThreshold(cv.cvtColor(title,cv.COLOR_BGR2GRAY), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
            y_nonzero, x_nonzero = np.nonzero(dilation)
            pad = 1
            title= title[np.min(y_nonzero)+pad:np.max(y_nonzero)-(2*pad), np.min(x_nonzero)+pad:np.max(x_nonzero)-(2*pad)]

            #crop white edge
#            tthresh = cv.threshold(cv.cvtColor(title,cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
#            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
#            erode = cv.erode(tthresh, rect_kernel, iterations=1)
#            dilation = cv.dilate(erode, rect_kernel, iterations=1)
#            dilation = cv.bitwise_not(dilation)

#            pad = 0
#            title= title[np.min(y_nonzero)+pad:np.max(y_nonzero)-(2*pad), np.min(x_nonzero)+pad:np.max(x_nonzero)-(2*pad)]

#            coords = np.column_stack(np.where(tthresh > 0))
#            crect = cv.minAreaRect(coords)
#            box=cv.boxPoints(crect)
#            box=np.int0(box)
#            cv.drawContours(t2, [box], -1, (0, 255, 0), 1)

            tthresh = cv.threshold(cv.cvtColor(rotated,cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            dilation = cv.dilate(tthresh, rect_kernel, iterations=1)
            rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
            contours, hierarchy = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

            t2 = title.copy()
            cv.drawContours(t2, contours, -1, (255, 0, 0), 1)


            lines = cv.HoughLines(tthresh, 1, np.pi / 180, 150, None, 0, 0)
            
            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    if ((abs(pt1[0] - pt2[0]) < 15) or (abs(pt1[1] - pt2[1]) < 15)):
                        cv.line(ilines, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
        else:
            tthresh = cv.adaptiveThreshold(cv.cvtColor(title,cv.COLOR_BGR2GRAY), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)
            linesP = cv.HoughLinesP(tthresh, 1, np.pi / 180, 50, None, 50, 10)
        
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv.line(ilines, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA)

        lab = cv.cvtColor(cv.GaussianBlur(title, (3, 3), 1), cv.COLOR_BGR2Lab)

        im32f = np.array(lab[:, :, 1:3], dtype=np.float32)
        k = 2 # 2 clusters
        term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
        ret, labels, centers = cv.kmeans(im32f.reshape([lab.shape[0]*lab.shape[1], -1]), 
                                      k, None, term_crit, 10, 0)
        # segmented image
        labels = labels.reshape([lab.shape[0], lab.shape[1]]) * 255

        im2 = title.copy()

        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')
        edge = cv.Canny(title, thrs1, thrs2, apertureSize=5)

        # dilate canny output to remove potential
        # holes between edge segments
#        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
#        edge = cv.dilate(edge, rect_kernel, iterations=1)

#        cv.imshow('canny', edge)

        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        edge = cv.dilate(edge, rect_kernel, iterations=1)

        im2 = np.uint8(im2/2.)
#        im2 = np.uint8(im2/2.)
        im2[edge != 0] = (0,255,0)

        contours, hierarchy = cv.findContours(edge, 
            cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

#        contours, hierarchy = cv.findContours(edge, 
#            cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#        cv.drawContours(im2, contours, -1, (255, 0, 0), 3)
#        cv.imshow('contours', im3)

        count=0
        rect=[]
        for cnt in contours:
            approx = cv.approxPolyDP(cnt,0.02*cv.arcLength(cnt,True),True)
            if (len(approx)==4 and abs(cv.contourArea(cnt)) > 100):
                #and cv.isContourConvex(cnt)):
#                cnt = cnt.reshape(-1, 2)
#                max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
#                if max_cos < 0.3:
                rect.append(cnt)
                cv.drawContours(im2,[approx],0,(0,0,255),1)

        cv.putText(crop, '%.4f' % (oangle),
                   (40, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv.putText(crop, '%.4f' % (mangle),
                   (40, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv.putText(crop, '%.4f' % (angle),
                   (40, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv.putText(crop, 'w%d h%d' % (rw,rh),
                   (40, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        text = pytesseract.image_to_data(title, config="--oem 3 --psm 7", output_type='data.frame')
        text = text[text.conf != -1]
        try:
            #level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text        
            lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                                         .apply(lambda x: ''.join(list(str(x)))).tolist()
            confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
        
            line_conf = []
            for i in range(len(lines)):
                if lines[i].strip():
                    line_conf.append((lines[i], round(confs[i],3)))

            line = "".join(lines)

            conf = 0
            if (np.any(confs)):
                conf = np.mean(confs)

            key = "%d%d%d%d" % (tile[0], tile[1], tile[2], tile[3])

            write = False
            card = {"conf":conf}
            if key not in cardsByTile:
                cardsByTile[key] = card
                write = True

            if key in cardsByTile:
                if conf > cardsByTile[key]["conf"]:
                    cardsByTile[key] = card
                    write = True

            if write:
                tf = open("cards/%d_%d_%d_%d.txt" % (tile[0], tile[1], tile[2], tile[3]), "a")
                tf.write(line+"\n")
                tf.write("%.4f\n" % conf)
                tf.close()
                orig = cv.bitwise_not(rotated)
                cv.imwrite("cards/%d_%d_%d_%d.card.png" % (tile[0], tile[1], tile[2], tile[3]) , orig)
                cv.imwrite("cards/%d_%d_%d_%d.rotated.png" % (tile[0], tile[1], tile[2], tile[3]) , rotated)
                cv.imwrite("cards/%d_%d_%d_%d.otsu.png" % (tile[0], tile[1], tile[2], tile[3]) , tthresh)
                cv.imwrite("cards/%d_%d_%d_%d.title.png" % (tile[0], tile[1], tile[2], tile[3]) , title)
                cv.imwrite("cards/%d_%d_%d_%d.titlee.png" % (tile[0], tile[1], tile[2], tile[3]) , edge)
                cv.imwrite("cards/%d_%d_%d_%d.titlec.png" % (tile[0], tile[1], tile[2], tile[3]) , im2)
                cv.imwrite("cards/%d_%d_%d_%d.png" % (tile[0], tile[1], tile[2], tile[3]) , crop)
            return (line, conf)
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print(tile)
            print(text)
            return (None, None)

        #  
        #

        crop_labels = labels[0: int(h/27 * 4), 0: w]
        text = pytesseract.image_to_data(crop_labels, config="--oem 3 --psm 7", output_type='data.frame')
        text = text[text.conf != -1]
        #level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text
        lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                                     .apply(lambda x: ''.join(list(str(x)))).tolist()
        confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
    
        line_conf = []
        for i in range(len(lines)):
            if lines[i].strip():
                line_conf.append((lines[i], round(confs[i],3)))

        lblline = "".join(lines)
        lblconf = 0
        if (np.any(confs)):
            lblconf = np.mean(confs)

        #  
#        lines = text.groupby(['page_num','block_num','par_num','line_num'])['text'].apply(list)
#        confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()

        cv.imwrite("cards/%d_%d_%d_%d.label.title.png" % (tile[0], tile[1], tile[2], tile[3]) , crop_labels)
        cv.imwrite("cards/%d_%d_%d_%d.label.png" % (tile[0], tile[1], tile[2], tile[3]) , labels)


    return (None, None) 

#def reshape_split(image, kernel_size: tuple):
#
#    img_height, img_width, channels = image.shape
#    tile_height, tile_width = kernel_size
#
#    tiled_array = image.reshape(channels,
#        img_height // tile_height,
#                                tile_height,
#                                img_width // tile_width,
#                                tile_width,
#                                channels)
#    tiled_array = tiled_array.swapaxes(1,2)
#    return tiled_array

#
#  MAIN
#

signal.signal(signal.SIGINT, handler)

#
#
try:
    vdev = sys.argv[1]
except IndexError:
    vdev = 0

print("capture %s\n" % vdev)

#fork(v42lsink)
#if (os.path.exists(vdev)):
#path = Path(vdev)
#if (path.is_file):
if (type(vdev) == str):
    actualtap=False
else:
    #  will block for either timeout or time until success string outputs
    #
    #  proc used in handler
    #
    (proc, q) = v4l2sink()

isRunningOrStart() #open camera on android device

vid = cv.VideoCapture(vdev)
#if 'size' in params:
#    w, h = map(int, params['size'].split('x'))
#vid.set(cv.CAP_PROP_FRAME_WIDTH, 2400)
#vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

cv.namedWindow('edge')
cv.createTrackbar('scale', 'edge', scale_percent, 100, nothing)
cv.createTrackbar('thrs1', 'edge', 750, 5000, nothing)
cv.createTrackbar('thrs2', 'edge', 2855, 5000, nothing)

timeout=2
waited=0
method="edge"
previous_frame = None
frame_count = 0
prev_scale = None
while True:

    _ret, orig = vid.read()
    if (orig is None):
        time.sleep(0.1) 
        waited=waited+0.1
        if (waited > timeout):
            vid = cv.VideoCapture(vdev)
        continue
    else:
        frame_count += 1

#    if ((frame_count % 2) != 0):
#        continue

    #  classically OCR uses white as text foreground and black for background
    #  
    #  MTG typically has black text on a lighter color bg
    #
#    print('Original Dimensions : ',img.shape)
    prev_scale = scale_percent
    scale_percent = cv.getTrackbarPos('scale', 'edge')
    width = int(orig.shape[1] * scale_percent / 100)
    height = int(orig.shape[0] * scale_percent / 100)
    dim = (width, height)

    t = clock()

    #  split frame into tiles so that we can have an area 
    #  for detected cards to occupy for reference
    #
    ih, iw, ic = orig.shape

    img = cv.resize(orig, dim, interpolation = cv.INTER_AREA)

    #try 10 by 10 grid
    rows, cols = 10,10
    tilesize = (int(ih//rows), int(iw//cols))
#    print(ih,iw,tilesize)
#    print("rows %d, cols %d" % (orig.rows, orig.cols))

    #  first part is card shape detection
    #

#    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
#                	cells_per_block=(2, 2), visualize=True, multichannel=True)

#    cv.imshow('hog', hog_image)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # denoise before edge detection
    blur = cv.GaussianBlur(gray,(15,15),0)

    if (previous_frame is None):
        previous_frame = blur
        continue

    if (prev_scale != scale_percent):
        previous_frame = blur
    # motion detection
    #
    # calculate difference and update previous frame
    diff_frame = cv.absdiff(src1=previous_frame, src2=blur)
    previous_frame = blur

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    kernel = np.ones((5, 5))

    #  since this application is supposed to be a still vantage point, we will erode 
    #  to mitigate motion detection as opposed to dilate 
    #
    diff_frame = cv.erode(diff_frame, kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv.threshold(src=diff_frame, thresh=20, maxval=255, type=cv.THRESH_BINARY)[1]

    #    cv.imshow('dilate', diff_frame)
    #    cv.imshow('motion', thresh_frame)

    #    contours, _ = cv.findContours(image=thresh_frame, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    #    imgm = img.copy()
    #    for c in contours:
    #        if cv.contourArea(c) < 50:
    #            continue
    #        (x, y, w, h) = cv.boundingRect(c)
    #        cv.rectangle(img=imgm, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    #        cv.drawContours(thresh_frame, [c], 0, (0, 255, 0), -1)

#    cv.imshow('motion', thresh_frame)

#    cv.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

#    ret, thresh1 = cv.threshold(gray, 175, 255, cv.THRESH_BINARY)

    #  normalize color histogram for greater contrast
    #
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(blur)

    im2 = img.copy()

#        sh = t.shape
#        rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #  i know of three methods so far for processing images towards
    #  shape detection
    #
    #  the first is to use Canny edge detection combined 
    #
    if (method == "edge"):
        thrs1 = cv.getTrackbarPos('thrs1', 'edge')
        thrs2 = cv.getTrackbarPos('thrs2', 'edge')
        edge = cv.Canny(contrast, thrs1, thrs2, apertureSize=5)

        # dilate canny output to remove potential
        # holes between edge segments
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        edge = cv.dilate(edge, rect_kernel, iterations=1)
        im2 = np.uint8(im2/2.)
        im2[edge != 0] = (0,255,0)

        contours, hierarchy = cv.findContours(edge, 
            cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        im3 = img.copy()
#        contours, hierarchy = cv.findContours(edge, 
#            cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(im3, contours, -1, (255, 0, 0), 3)
#        cv.imshow('contours', im3)

        count=0
        rect=[]
        for cnt in contours:
            approx = cv.approxPolyDP(cnt,0.02*cv.arcLength(cnt,True),True)
            if (len(approx)==4 and abs(cv.contourArea(cnt)) > 100):
                #and cv.isContourConvex(cnt)):
                cnt = approx.reshape(-1, 2)
                max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                if max_cos < 0.3:
                    count=count+1
                    rect.append(cnt)
                    cv.drawContours(im2,[cnt],0,(0,0,255),3)

        if (count >= num_cards):
            cardnum=0
            for cnt in rect:
                cardnum += 1
                box=cv.boundingRect(cnt)
#                print(box)
                unscaled = tuple(int(n / (scale_percent / 100)) for n in box)
#                print(unscaled)
                x, y, w, h = unscaled
                #crop
                cardimage=orig[y: y + h, x : x + w]
                (line, conf) = process_card(cardimage, unscaled, tilesize, page)
                if (line != None and conf != None):
                    cv.putText(im2, '%s' % (line),
                               (box[0] - 40, box[1] + int(box[3] / 10)),
                               cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    cv.putText(im2, '[%0.3f]' % (conf), 
                               (box[0] - 40, box[1] + int(box[3] / 10) * 2),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#        lines = cv.HoughLinesP(edge, 1, math.pi/180.0, 40, np.array([]), 50, 10)
#        a, b, _c = lines.shape
#        for i in range(a):
#            cv.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 3, cv.LINE_AA)
        
    #  the second is essentially centered around a binary threshold 
    #  this one then is followed by dilation and findContours
    #
    if (method == "threshold"):
        bkg_level = blur[int(height/2)][int(width/100)]
        thresh_level = bkg_level + BKG_THRESH
        #    print(thresh_level)

        ret, thresh1 = cv.threshold(blur, thresh_level, 255, cv.THRESH_BINARY)

        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 100))

        # Applying dilation on the threshold image
        dilation = cv.dilate(thresh1, rect_kernel, iterations=1)

        cnts = cv.findContours(
            dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for cnt in cnts:
            approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
    #        print(len(approx))
            if len(approx)==5:
    #            print("Blue = pentagon")
                cv.drawContours(img,[cnt],0,255,-1)
            elif len(approx)==3:
    #            print("Green = triangle")
                cv.drawContours(img,[cnt],0,(0,255,0),-1)
            elif len(approx)==4:
    #            print("Red = square")
                cv.drawContours(img,[cnt],0,(0,0,255),-1)
            elif len(approx) == 6:
    #            print("Cyan = Hexa")
                cv.drawContours(img,[cnt],0,(255,255,0),-1)
            elif len(approx) == 8:
    #            print("White = Octa")
                cv.drawContours(img,[cnt],0,(255,255,255),-1)
            elif len(approx) > 12:
    #            print("Yellow = circle")
                cv.drawContours(img,[cnt],0,(0,255,255),-1)
        cv.imshow('dilation', dilation)
        cv.imshow('thresh', thresh1)

    if (method == "lines"):
        #https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
        # Apply HoughLinesP method to
        # to directly obtain line end points
        lines_list =[]
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=100, # Min number of votes for valid line
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
                    )
         
        # Iterate over points
        for points in lines:
              # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])


    if (debug):
        cv.imshow('hist a eq', blur)

    #  display tile grid
    #
    showts = tuple(int(n * scale_percent / 100) for n in tilesize)
    tilecount = 0
    for row in range(rows):
        for col in range(cols):
            tilecount += 1
            y0 = row * showts[0]
            y1 = y0 + showts[0]
            x0 = col * showts[1]
            x1 = x0 + showts[1]

            #show tile grids with tile number text
            rect = cv.rectangle(im2, (x0, y0), (x1, y1), (127, 127, 127), 1)
            cv.putText(im2, '%d' % (tilecount),
#                       (x0 + int(showts[1] / 2), y0 + int(showts[0] / 2)),
                       (x0 + 5, y1 - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (127,127,127), 1)

    #  show framerate (time between processed frames)
    #
    dt = clock() - t
    cv.putText(im2, 'time: %.1f ms' % (dt*1000), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    cv.imshow(method, im2)

    #  escape key
    if cv.waitKey(5) == 27:
        break

cv.destroyAllWindows()

#  cleanup after Popen
frame = inspect.currentframe()
handler(signal.SIGINT, frame)
