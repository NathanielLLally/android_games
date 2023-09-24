#!/usr/bin/python
# Import required packages
import cv2
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

scale_percent = 60 # percent of original size
#serial = "FAMVRW9D9HDEHEWS" #BLU Pure
#serial="XG9LGEZX6L75QGMJ" #Oppo Realme RMX2001
serial="0123456789ABCDEF" #Oppo Realme RMX2001


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
    return cv2.getTickCount() / cv2.getTickFrequency()

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

def itap(coord):
    (x,y) = coord
    cmd = "/home/nathaniel/android/sdk/platform-tools/adb -s %s shell input tap %d %d" % (serial, x, y)
    print(cmd)

    result = subprocess.run(cmd.split(), capture_output=True, text=True)

def v4l2sink():
    cmd = 'ps -ef | grep scrcpy | grep v4l2-sink'
    ps = subprocess.run(['ps', '-ef'], check=True, capture_output=True)
    processNames = subprocess.run(['grep', 'scrcpy'],
                              input=ps.stdout, capture_output=True, check=True)
    pidnfo = subprocess.run(['grep', 'v4l2-sink'],
                              input=processNames.stdout, capture_output=True)
    if (len(pidnfo.stdout) == 0):
        print("scrcpy is not running, starting process", flush=True)
        cmd = "scrcpy -s %s --lock-video-orientation=3 --v4l2-sink=/dev/video0 --stay-awake --power-off-on-close --no-display --show-touches" % serial
        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pq = queue.Queue()
        t1 = threading.Thread(target=stdout_reader, args=(proc, pq))
        t1.start()
        t2 = threading.Thread(target=stderr_reader, args=(proc, pq))
        t2.start()
        return (proc, pq)
    else:
        return (False,False)

def handler(signum, frame):
    try:
        if (isinstance(proc, subprocess.Popen)):
            proc.terminate()
    except:
        pass
    sys.exit()

signal.signal(signal.SIGINT, handler)

# Read image from which text needs to be extracted
# filename = sys.argv[1] if len(sys.argv) >= 2 else 'defaultfile.txt'
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
    print("should start scrcpy")
    (proc, q) = v4l2sink()
    wait = isinstance(proc, subprocess.Popen)
    while wait:
        try:
            line = q.get(block=False)
            print('{0}'.format(line), end='')
        #INFO: v4l2 sink started to device: /dev/video0
            m = re.match(r".*?v4l2 sink started to device: (?P<device>[\/\w]+)", line)
    #        m = re.match(r".*?v4l2 sink started to device: ", line)
            if (m):
                print("scrcpy v4l2 sink started using device %s" % m.group('device'))
                time.sleep(1)
                wait = False
        except queue.Empty:
            pass
        time.sleep(0.05) 

# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'

vid = cv2.VideoCapture(vdev)
#if 'size' in params:
#    w, h = map(int, params['size'].split('x'))
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2400)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#(child_stdin,
# child_stdout,
 #child_stderr) = os.popen3(cmd, mode, 80)

screen = "raid"

count = [2,4,20,20]
while True:
    _ret, img = vid.read()

#    print('Original Dimensions : ',img.shape)
    
    if (screen == "raid"):
        t = clock()

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#        buttons = ((897,974),(1135,974),(1375,974),(1615,974))
#        buttons = ((897,974),(1135,974),(1375,974),(1615,974))
        colors = []
        labels = []
        for b in buttons:
            (x,y) = b
            hue = hsv[y,x,0]
            colors.append(hue)

        for hue in colors:
            label = "green"
            if hue < 5:
                label = "black"
            elif hue < 35:
                label = "unknown"
            elif hue < 50:
                label = "unknown"
            elif hue < 70:
                label = "green"
            elif hue < 255:
                label = "dark red"
            labels.append(label)

        print(labels)
        for i in range(len(buttons)):
            if (labels[i] == "green"):
                count[i] = count[i] -1
                if (count[i] > 0):

                    print("count[%d] %d, push %d" % (i, count[i], i))
                    itap(buttons[i])


        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #print("%d %d %d %d" % (int(img[897,974]),int(img[1135,974]),int(img[1375,974]),int(img[1615,974])))
        dt = clock() - t
        cv2.putText(img, 'time: %.1f ms' % (dt*1000), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.imshow('gunsup', img)

    else:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        t = clock()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

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

    # Creating a copy of image
        im2 = img.copy()

    # A text file is created and flushed
    #    file = open("%s.recognized.txt" % pfx, "w+")
    #    file.write("")
     #   file.close()


        pois = {
            'energy': [2138, 58],
            'loadout': [2191, 1023],
            'acceptcontinue': [2101, 984],
            'wave': [2184, 90],
            'gold': [1682, 55],
            'munitions': [1945, 55],
            'oil': [2223, 55],
            'defense rating': [784, 454],
            'outpose level': [1119, 457],
            'player': [789, 349],
            'player number': [1070, 372],
            'moveout': [2184, 916],
            }

        additional_rects = [
            [2118,994, 2229, 1050, "war"],
        ]

        for p in pois.keys():
            pois[p][0] = int(pois[p][0] * scale_percent / 100)
            pois[p][1] = int(pois[p][1] * scale_percent / 100)

        for a in additional_rects:
            for i in range(0,4):
                a[i] = int(a[i] * scale_percent / 100)
            pois[a[4]] = [int(a[0] + (a[2] - a[0]) / 2), int(a[1] + (a[3] - a[1]) / 2)]
            print(pois[a[4]])

        boxes = []
        for cnt in contours:
            boxes.append(cv2.boundingRect(cnt))

        for r in additional_rects:
            boxes.append((int(r[0]),int(r[1]),int(r[2]-r[0]),int(r[3]-r[1])))
            pois[r[4]] = [int(r[0] + (r[2] - r[0]) / 2), int(r[1]+(r[3] - r[1]) / 2)]

        recognized = dict()

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
        for box in boxes:
            x, y, w, h = box

    #        print("x %d y %d w %d h %d" % (x, y, w, h))

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            for k in pois.keys():
                poi = pois[k]
                if poi[0] > x and poi[0] < (x + w) and poi[1] > y and poi[1] < (y + h):
                    cv2.circle(im2, (poi[0], poi[1]), 3, (255,0,0), 2)
                    # Cropping the text block for giving input to OCR
                    cropped = thresh1[y : y + h, x : x + w]
                    text = pytesseract.image_to_string(cropped, config="--oem 3 --psm 7")

                    # Apply OCR on the cropped image
                    recognized[k] = text
                    cv2.putText(im2, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


        dt = clock() - t
        cv2.putText(im2, 'time: %.1f ms' % (dt*1000), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.imshow('gunsup', im2)

        for k in recognized.keys():
            if (len(recognized[k]) > 0):
                print("%s = %s" % (k, recognized[k]))

        if 'war' in recognized:
            if (recognized['war'].lower().find("war") != -1):
                tap('war')

        if 'acceptcontinue' in recognized:
            if (recognized['acceptcontinue'].lower().find("accept") != -1):
                print(recognized['acceptcontinue'].lower())
                print("Click accept")
                tap('acceptcontinue')
            elif (recognized['acceptcontinue'].lower().find("continue") != -1):
                print(recognized['acceptcontinue'].lower())
                print("Click continue")
                tap('acceptcontinue')

    if cv2.waitKey(5) == 27:
        break
