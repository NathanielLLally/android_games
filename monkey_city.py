#!/usr/bin/python
# Import required packages
from functools import reduce
import cv2 as cv
import cv2 as cv2
import pytesseract
import sys
import optparse
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
import imutils
from imutils.object_detection import non_max_suppression
from PyQt5 import QtGui
from PyQt5 import QtCore
#from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QTable, QTableItem
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import *
from PyQt5 import QtTest
#pyqtSignal, pyqtSlot, Qt, QThread
import json
import webbrowser
import requests
import urllib
from ppretty import ppretty
import traceback
from datetime import datetime
from datetime import timedelta
from enum import auto
from enum import Flag

class LogLevel(Flag):
    ACTIONS = auto()
    OCRTEXT = auto()
    ADB = auto()
    IMAGE = auto()
    MAIN = auto()
    @classmethod
    def is_actions(cls, v):
        return (cls.ACTIONS in cls(v))
    @classmethod
    def is_ocrtext(cls, v):
        return (cls.OCRTEXT in cls(v))
    @classmethod
    def is_adb(cls, v):
        return (cls.ADB in cls(v))
    @classmethod
    def is_image(cls, v):
        return (cls.IMAGE in cls(v))
    @classmethod
    def is_main(cls, v):
        return (cls.MAIN in cls(v))

#verbosity = LogLevel.MAIN | LogLevel.ADB | LogLevel.OCRTEXT | LogLevel.ACTIONS



#
page = 1

# Adaptive threshold levels
BKG_THRESH = 30
CARD_THRESH = 30
SERIAL = ""

FARMCOORDS = [(920,310),(920,620),(920,930),(1415,310),(1415,620),(1415,930),(1838,607),(1838,930)]

#  bounds box of building info popup upperl left (x,y),(w,h)
#
ADBLDGBOUNDS = [(384,474),(531,176)]
#925 (900 maybe), 650
#
#  these coords are to shorten
#    the build time for a particular building by 15 minutes by ad watching
#ADCOORDS = [(655,749),(778,711)]

#  this is for 5 premium currency from watching aD
#
ADCOORDS = [(160,130),(2172,515), (250,470), (1240, 950)]
bb = []
delay = 0.2
bb.append({"action":"self.hold(%s,%s,0.1)" % (160,130), "name": "tap ruby coords from pointerLocationBar 160,130."})
bb.append({"action":"self.waitFor(%s)" % 0.5 , "name":"noop"})
bb.append({"action":"self.swipe(%s,%s,%s,%s,0.3)" % (2172,515,250,470), "name": "swipe left" })
bb.append({"action":"self.waitFor(%s)" % 2 , "name":"noop"})
bb.append({"action":"self.hold(%s,%s,0.1)" % (1240,950), "name": "then hit watch ad button"})

PREMIUM_CURRENCY_AD_ACTIONS = bb


#  Code derived and adapted from optical character recognition based playing card software that I wrote 
#   in order to aid in managing and trading tangible collectible cards
#


def stdout_reader(proc, q):
    for line in iter(proc.stdout.readline, b''):
        q.put(line.decode('utf-8'))

def stderr_reader(proc, q):
    for line in iter(proc.stderr.readline, b''):
        q.put(line.decode('utf-8'))

def handler(signum, frame):
    try:
        if (isinstance(proc, subprocess.Popen)):
            proc.terminate()
        sys.exit(app.exec_())
    except:
        pass
    sys.exit()

def nothing(*arg):
    pass

class Actions(object):
    def __init__(self, *args, **kwargs):
        self.verbose = V.ACTIONS | V.ADB
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        if LogLevel.is_actions(self.verbose):
            print("Actions __init begin")
        self.forceModeActionsFlag = False
        self.currentAction = None
        self.currentActionStart = None
        self.waitForStart = None
        self.waitForEnd = None
        self.currentActionEnd = None
        self.timeout = 5
        self._scale = None
        self._device_width = None
        self._device_height = None
        self._X11_width = None
        self._X11_height = None
        self._mOption = None
        self._serial = None
        self.iActions = []
        self.cActions = []
        self.actionResult = ""
        if 'serial' in kwargs:
            self._serial = kwargs['serial']
        if LogLevel.is_actions(self.verbose):
            print("Actions __init end")

    #  ms
    def clock(self):
        return (cv.getTickCount() / cv.getTickFrequency()) * 1000

    def actions(self,Pactions=[]):
        if hasattr(Pactions,'__len__'):
            if len(Pactions) == 0:
                raise Exception("derive this class and override this method to provide a default action list")
            else:
                if (not self.iActions):
                    self.iActions = Pactions.copy()
                    self.cActions  = self.iActions.copy()
        else:
            raise Exception("invalid parameter to actions")

        return self.cActions

    def insertNextAction(self, action):
        if not self.iActions:
            raise Exception('insertNextAction called before actions initialized')
        self.cActions.insert(0,action);

    def nextAction(self):
        actions = self.actions()
        if not len(actions):
            self.reset()
            actions = self.actions()

        self.currentAction = actions.pop(0)
        self.currentActionStart = None
        self.currentActionEnd = None
        return self.currentAction

    def action(self, Paction=[]):
        if (self.currentAction == None):
            self.nextAction()
        return self.currentAction

    def do(self,force=None):
        action = self.action()
        if self.currentActionStart == None or self.currentActionEnd == None:
            self.currentAction = None
            self.currentActionStart = self.clock()

        # only waitFor action can set waitForStart
            if self.waitForStart == None:
                if isinstance(action, dict):
                    if LogLevel.is_actions(self.verbose):
                        if ('name' in action and action['name'] != 'noop'):
                            print("do: %s - %s" % (action["name"], action["action"]))
                        else:
                            print("do: anon - %s" % (action["action"]))
                else:
                    if LogLevel.is_actions(self.verbose):
                        print("self.currentAction: [%s]" % action)

            if self.forceModeActionsFlag != force and force != None:
                self.forceModeActionsFlag = force
            if isinstance(action, dict):
                exec(action["action"])
            else:
                exec(action)

        else:
            print("do(): doing nothing, action started at [%s]" % self.currentActionStart)
            raise Exception("action apparently did not handle start & end properly")

    def getResolutions(self):
        try:
            serial = self.serial()
        except:
            return (None, None, None, None)

    #  get android device physical size
        cmd = "adb -s %s shell wm size" % serial
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        x = 0
        y = 0
        X11x = 0
        X11y = 0
        
        if (len(result.stdout) > 0):
            m = re.search(r"^(Physical size: )(?P<y>\d+)x(?P<x>\d+)\s", result.stdout, re.M)
            x = str(m.group('x'))
            y = str(m.group('y'))

        print("yo!")

    #get X11 resolution
        cmd = "xrandr --current"
    # | grep '*' | uniq | awk '{print $1}' | cut -d 'x' -f1"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if (len(result.stdout) > 0):
            m = re.search(r".*?(?P<x>\d+)x(?P<y>\d+)\s+.*?\*", result.stdout, re.M)
            X11x = str(m.group('x'))
            X11y = str(m.group('y'))

        if LogLevel.is_actions(self.verbose):
            print("device %s has resolution (%s,%s), X11 has resolution (%s, %s)" % (serial,x,y,X11x,X11y))
        print("device %s has resolution (%s,%s), X11 has resolution (%s, %s)" % (serial,x,y,X11x,X11y))
        self._device_width = x
        self._device_height = y
        self._X11_width = X11x
        self._X11_height = X11y
        return (int(x),int(y),int(X11x),int(X11y))

    def resolutions(self):
        if (self._device_width == None or self._device_height == None or self._X11_width == None or self._X11_height == None):
            (self._device_width,self._device_height,self._X11_width,self._X11_height) = self.getResolutions()
        return (self._device_width,self._device_height,self._X11_width,self._X11_height)

    def device_width(self):
        if (self._device_width == None):
            (self._device_width,self._device_height,self._X11_width,self._X11_height) = self.getResolutions()
        return self._device_width

    def device_height(self):
        if (self._device_height == None):
            (self._device_width,self._device_height,self._X11_width,self._X11_height) = self.getResolutions()
        return self._device_height
    def X11_width(self):
        if (self._X11_width == None):
            (self._device_width,self._device_height,self._X11_width,self._X11_height) = self.getResolutions()
        return self._X11_width

    def X11_height(self):
        if (self._X11_height == None):
            (self._device_width,self._device_height,self._X11_width,self._X11_height) = self.getResolutions()
        return self._X11_height

    def scale(self):
        (device_width,device_height,X11_width,X11_height) = self.resolutions()
        if ((device_width > X11_width) or (device_height > X11_height)):
            self._scale=0.5
        else:
            self._scale = 1

        return self._scale

    #  option that will be used to start video for linux
    #
    def mOption(self):
        if (self._mOption == None):
            print(self.device_width())
            print(self.scale())
            self._mOption = int(float(self.device_width()) * float(self.scale()))
        return self._mOption

    def serial(self):
        if (not self._serial):
#len(self._serial) <= 0):
            self.getDeviceSerial()
        return self._serial

    def getDeviceSerial(self,Pserial=""):
        if (len(Pserial)):
            self._serial = Pserial
        try:
            self._serial = os.environ['ANDROID_SERIAL']
        except:
            pass

        cmd = "adb devices"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if (len(result.stdout) > 0):
            m = re.search(r"^(?!(List|\s+))(?P<serial>.*?)\s", result.stdout, re.M)
            if m:
                if (self._serial):
                    if (str(m.group('serial')) == self._serial):
                        if LogLevel.is_actions(self.verbose):
                            print("using connected device [%s]" % self.serial)
                        return self._serial
                else:
                    self._serial = str(m.group('serial'))
                    if LogLevel.is_actions(self.verbose):
                        print("using connected device [%s]" % self._serial)
                    return self._serial
        raise Exception("no connected android device!") 

#  high level actions begin here
#
    def tap(self, x,y):
        cmd = "adb -s %s shell input tap %s %s" % (self.serial(),x / self.scale(),y / self.scale())
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        self.currentActionEnd = self.clock()
        if LogLevel.is_adb(self.verbose):
            print(result)
        return result

    def swipe(self, x,y, x2, y2, seconds):
        cmd = "adb -s %s shell input swipe %s %s %s %s %s" % (self.serial(),int(x / self.scale()),int(y / self.scale()), int(x2 / self.scale()), int(y2 / self.scale()), int(seconds * 1000))
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        self.currentActionEnd = self.clock()
        if LogLevel.is_adb(self.verbose):
            print(result)
        return result

    def backButton(self):
        cmd = "adb -s %s shell input keyevent 4"
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        self.currentActionEnd = self.clock()
        return result

    def hold(self, x,y, seconds):
        cmd = "adb -s %s shell input swipe %s %s %s %s %s" % (self.serial(),int(x / self.scale()),int(y / self.scale()), int(x / self.scale()), int(y / self.scale()), int(seconds * 1000))
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        self.currentActionEnd = self.clock()
        if LogLevel.is_adb(self.verbose):
            print(result)
        return result

    def mForeground(self):
        cmd = "adb -s %s shell dumpsys window" % self.serial()
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)

        if (len(result.stdout) > 0):
#mFocusedApp=AppWindowToken{841c7c6 token=Token{2e439a1 ActivityRecord{4c22e08 u0 com.ninjakiwi.monkeycity/com.google.android.gms.ads.AdActivity t27}}}
#mFocusedWindow=Window{569a398 u0 com.ninjakiwi.monkeycity/com.google.android.gms.ads.AdActivity}
            m = re.search(r"^\s+mFocusedWindow=Window{(?P<windowId>\S+) \S+ (?P<package>\S+)\/(?P<activity>\S+)}", result.stdout, re.M)
            if m:
                    self.fgPackage = str(m.group('package'))
                    self.fgActivity = str(m.group('activity'))
                    print("foreground package %s activity %s" % (self.fgPackage, self.fgActivity))
                    return (self.fgPackage, self.fgActivity)
            else:
                    print("cannot determing foreground package/activity")
                    print(result.stdout)
        return ("","")


        #  TODO Implement me
        #
    def waitCondition(self, condition=None, value=None):
        print("wait for conditon- implement me")
        print(condition)

    def waitFor(self, seconds):
        #print("%s.waitFor(%s)"%(type(self), seconds))

        #re-entrant version
        if self.waitForStart:
            delta = self.clock() - self.waitForStart
           # print("delta: [%s]" % delta)
            if delta >= int(float(seconds) * 1000):
                self.currentAction = None
                self.currentActionEnd = self.clock()
                self.waitForStart = None
                print("action is ending, actual wait time was %s ms" % delta)
            else:
            #    print("action is ongoing, setting currentAction")
                self.currentAction = "self.waitFor(%s)" % (seconds)
        else:
            self.currentAction = "self.waitFor(%s)" % (seconds)
            self.waitForStart = self.currentActionStart

            #once set, let do() handle clearing
            self.waitForEnd = self.waitForStart + int(float(seconds) * 1000)
            #if self.waitForStart == None:
            #raise Exception("method not called with %s.do()" % type(self))

    def reset(self):
        if LogLevel.is_actions(self.verbose):
            print("Actions.reset")
        self.waitForStart = None
        self.cActions = self.iActions.copy()
        self.currentAction = None
        self.currentActionStart = None
        self.currentActionEnd = None

class MonkeyCity(Actions):
    def __init__(self, *args, **kwargs):
        super(MonkeyCity, self).__init__(*args, **kwargs)
        self.verbose = vv

        if LogLevel.is_actions(self.verbose):
            print("MonkeyCity __init__ begin")

        self.names = []
        if "mode" in kwargs:
            if hasattr(self,kwargs["mode"]):
                self.mode = kwargs["mode"]

        self.parent = None
        if "parent" in kwargs:
            self.parent = kwargs["parent"]

        self.modeOptions = {}
        self.modeOptions['WatchAd'] = { "disabled": False }
        #  can override from constructor
        if 'modeOptions' in kwargs:
            self.modeOptions = kwargs['modeOptions']


        self.mode = "CollectGold"
        a=[
            {"action":"self.clickForceModeActions()", "name":"click force actions"},
            {"action":"self.startGame()", "name":"start game or bail"}
        ]

        i=1
        delay = 0.1
        for (x,y) in FARMCOORDS:
            a.append({"action":"self.hold(%s,%s,0.1)" % (x,y), "name": "farm %s" % i})
            i=i+1
            a.append({"action":"self.waitFor(%s)" % delay , "name":"noop"})

        time_repeat =  120
        time_between_modes = 2
        #time_between_modes = time_repeat
        a.append({"action":"self.waitFor(\"%s\")" % time_between_modes , "name":"mode switch delay"})
        a.append({"action":"self.setMode(\"%s\")" % 'WatchAd' , "name":"setMode"})
        self.CollectGold = a.copy()

        #b=[{"action":"self.startGameIfClosed()", "name":"start game if not currently foreground"}]

        #  TODO cover case where watchad mode happens without a collectgold first (or just bail if that happens

        b=[]
        #WatchAd
        delay = 0.2
        b = PREMIUM_CURRENCY_AD_ACTIONS

        time_shortest_ad = 14
        time_fg_activity_change = 3

        #WatchAd first step
        b.insert(0,{"action":"self.disablePrivateDNS()", "name":"set private dns off"})

        #TODO finish
        conditionFGActivityChange = compile("self.mForeground() == ('com.ninjakiwi.monkeycity', 'com.ninjakiwi.MainActivityGradle')", 'compare_tuples_package_activity', 'exec')
#        b.append({"action":"self.waitCondition(\"%s\")" % time_shortest_ad , "name":"shortest ad time"})

        b.append({"action":"self.waitFor(%s)" % 45 , "name":"noop"})
        b.append({"action":"self.backButton()", "name":"backbutton"})
        b.append({"action":"self.waitFor(%s)" % 17 , "name":"noop"})
        b.append({"action":"self.backButton()", "name":"backbutton"})
        b.append({"action":"self.waitFor(%s)" % 17 , "name":"noop"})
        b.append({"action":"self.backButton()", "name":"backbutton"})
        b.append({"action":"self.waitFor(\"%s\")" % 2 , "name":"noop"})
        #b.append({"action":"self.waitFor(\"%s\")" % time_shortest_ad , "name":"shortest ad time"})


        #WatchAd last step
        b.append({"action":"self.enablePrivateDNS()"})
#        b.append({"action":"self.waitFor(\"%s\")" % time_repeat , "name":"all mode actions repeat delay"})
        b.append({"action":"self.stopApp('com.ninjakiwi.monkeycity')", "name":"stop game", "flags":"once"})
        b.append({"action":"self.setMode(\"%s\")" % 'CollectGold' , "name":"setMode"})
        self.WatchAd = b.copy()

        #self.CollectGold = [{"action":"self.waitFor(15)", "name":"Noop"}, {"action":"self.tap(920,648)", "name":"farm 1"}, {"action":"self.waitFor(1)","name":"noop"},{"action":"self.tap(892,930)", "name":"farm 2"}, {"action":"self.tap(1415,620)", "name":"farm 3"}, {"action":"self.tap(1415,930)", "name":"farm 5"}, {"action":"self.tap(1838,607)", "name":"farm 4"}, {"action":"self.tap(1838,930)", "name":"farm 6"} ]
        #self.CollectGold = [{"action":"self.startGame()", "name":"start monkey city"},{"action":"self.waitFor(1)", "name":"Noop"}]
        self.modeSteps = getattr(self,self.mode)
        self.sleep = 8
        self.doing = ""
        if LogLevel.is_actions(self.verbose):
            print("MonkeyCity __init end")

    def disablePrivateDNS(self):
        self.adbShell("settings put global private_dns_mode off")
        #self.adbShizuku("settings put global private_dns_mode off")
        dnsmode = self.adbShell("settings get global private_dns_mode")
        print("private dns mode %s " % dnsmode)

    def enablePrivateDNS(self, dnsServer='dns.adguard.com'):
        #self.adbShizuku("settings put private_dns_specifier %s" % dnsServer)
        #self.adbShizuku("settings put global private_dns_mode %s" % "hostname")
        self.adbShell("settings put global private_dns_specifier %s" % dnsServer)
        self.adbShell("settings put global private_dns_mode %s" % "hostname")
        dnsmode = self.adbShell("settings get global private_dns_mode")
        dnsserver = self.adbShell("settings get global private_dns_specifier")
        print("private dns mode %s server %s" % (dnsmode,dnsserver))


    def adbShizuku(self, Pcmd="", comment=""):
        shizukuInstructions = "open Shizuku, find 'Use Shizuku in terminal apps'.  export to any directory under\n/emulated/storage/0/\ncopy both rish and dex file to:\n/data/local/tmp/\nchmod +x rish\n\n"
        shizukuCmd = "adb shell sh /storage/emulated/0/Android/data/moe.shizuku.privileged.api/start.sh"
        #TODO:
        # check if server is running.
        # on non rooted devices, something like the following is required to be run every reboot
        # 
        #  note, this was run over adb wifi connection 
        #  adb shell sh /storage/emulated/0/Android/data/moe.shizuku.privileged.api/start.sh
        #
#        self.adbShell('sh /storage/emulated/0/Download/rish')
        rish = '/data/local/tmp/rish'
        cmd = "adb -s %s shell %s -c '%s'" % (self.serial(), rish, Pcmd)
        if LogLevel.is_adb(self.verbose):
            print("input: %s, %s" % (Pcmd,cmd))
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
#        echo 'settings get global private_dns_mode' | adb -s XG9LGEZX6L75QGMJ shell sh /storage/emulated/0/Download/ris
        if LogLevel.is_adb(self.verbose):
            print("%s, %s" % (result.stdout,result.stderr))

        if len(result.stderr) > 0:
            checkForShizukiPid = 'pidof grep moe.shizuku.privileged.api'

            cmd = "adb -s %s shell %s" % (self.serial(), checkForShizukiPid)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            m = re.search(r"^(?P<pid>\d+)", result.stdout, re.M)
            if m:
                if LogLevel.is_adb(self.verbose):
                    print("found Shizuku process %s" % (m.group('pid')))
                raise Exception("that means there is some other issue with command...  error is [%s]: %s" % (cmd, result.stderr))
            else:
                raise Exception("Shizuku server process not found, try running the following:\n%s\n\n%s" % (shizukuCmd, shizukuInstructions))

        if len(comment) > 0:
            print(comment)

        return result.stdout



    def adbShell(self, Pcmd, comment=""):
        cmd = "adb -s %s shell %s" % (self.serial(), Pcmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if LogLevel.is_adb(self.verbose):
            print(cmd)

        if len(result.stderr) > 0:
            raise Exception("error with command [%s]: %s" % (cmd, result.stderr))

        if LogLevel.is_actions(self.verbose):
            if len(comment):
                print(comment)

        return result.stdout

    # determine if we should toggle mute based on state,
    #  then return or toggle
    #
    def mute(self, state = True):
        muted = self.isMuted()
        if LogLevel.is_actions(self.verbose):
            print("is muted: %s" % muted)
        if state == True:
            if 'MUSIC' in muted:
                return
        if state == False:
            if 'MUSIC' not in muted:
                return
    
        try:
            self.adbShell("input keyevent 164", "toggle mute")
        except:
            pass

    def isMuted(self):
        cmd = "adb -s %s shell dumpsys audio" % (self.serial())
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        if (len(result.stdout) > 0):
# works            m = re.compile(r"^- STREAM_(?P<stream>\S+):", re.MULTILINE)
            m = re.compile(r"- STREAM_(?P<stream>\S+):.*?Muted: (?P<muted>\S+).*?Min: ?(?P<min>\S+).*?Max: ?(?P<max>\S+).*?streamVolume:(?P<volume>\S+).*?", re.DOTALL | re.MULTILINE)
            #m = re.compile(r"^- STREAM_(?P<stream>\S+):.*?\nMuted: (?P<muted>\S+).*?streamVolume:(?P<volume>\S+)", re.MULTILINE)
            nomatch = True
            streams = {}
            for match in m.finditer(result.stdout):
                #stream,muted,volume = match.groups()
                r = match.groupdict()
                streams[r['stream']] = r.copy() 
#                print("stream %s muted %s volume %s" % (r['stream'], r['muted'], r['volume']))
                nomatch = False
            if nomatch:
                raise Exception("cannot determine audio muted state")

#        mobs.sort(key = lambda e: int(np.linalg.norm(np.array((e[0],e[1])) - npcenter))) 
            return list(filter(lambda s: streams[s]['muted'] == 'true', streams.keys())) 

        raise Exception("cannot determine audio muted state")


    def stopApp(self, package):
        cmd = "adb -s %s shell am force-stop %s" % (self.serial(), package)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if LogLevel.is_adb(self.verbose):
            print(cmd)
        if len(result.stdout) > 0:
            print(result.stdout)

    def startApp(self, package):
        cmd = "adb -s %s shell monkey -p %s -c android.intent.category.LAUNCHER 1" % (self.serial(), package)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if LogLevel.is_adb(self.verbose):
            print(cmd)

    def setMode(self, mode):
        if self.mode != mode:
            if hasattr(self,mode):
#
#  handle disabled Flag
#                
                if mode in self.modeOptions:
                    o = self.modeOptions[mode]
                    print(o)
                    print("MonkeyCity.setMode: modeOptions.%s.disabled = %r" % (mode, o['disabled']))
                    if o['disabled']:
                        print("\n!!!!mode %s has been disabled, actions instead will be waitFor(repeat_time), reset actions to mode[%s]" % (mode, self.mode))
                        return

                self.mode = mode
                self.modeSteps = getattr(self,self.mode)
                self.iActions = self.modeSteps.copy()
                self.reset()
            else:
                raise Exception("mode %s doesn't exist" % mode)

    #  avoid re-entrant loop 
    #  check for flag etc... remove this call from actions (during reset)
    #
    #
    def clickForceModeActions(self):
        print("clickForceModeActions: enter")
        if (not self.forceModeActionsFlag):
            self.parent.forceModeActions(True)


    def startGame(self):
        print("startGame: enter")
        (p, a) = self.mForeground()
        force = self.forceModeActionsFlag
        print("forceModeActionsFlag = %r" % force)
        if (p == 'com.ninjakiwi.monkeycity' and force == False):
            self.actionResult = "game in use, bail"
            self.cActions = [{"action":"self.waitFor(180)", "name":"check start"}]
            return self.actionResult

        self.stopApp("com.ninjakiwi.monkeycity")
        self.mute()
        self.startApp("com.ninjakiwi.monkeycity")
        self.insertNextAction( {"action":"self.waitFor(17)", "name":"wait for game to start"})

        return ""

    def actions(self, actions=[]):
        if hasattr(actions, "__len__"):
            # get
            if (len(actions) == 0):
                #initialize
                if not self.iActions:
#                    self.iActions = self.make_button_taps(["summon", "summon", "ward", "summon", "heal", "summon", "buff", "debuff", "ward", "summon", "offhand"])
#                    self.iActions = self.make_button_taps(self.modeSteps)
                    self.iActions = self.modeSteps.copy()

#                    print(self.mode)
#                    print(self.modeSteps)
#                    print(self.iActions)
                    #let reset() handle this, otherwise when event loop calls do() it will begin
                    #self.cActions = self.iActions
            # set
            else:
                print("len of parameter to actions is %s" % len(actions))
                raise Exception("please use constructor for setting default actions")
        else:
            raise Exception("implement me. %s.actions() passed a %s, val [%s]" % (type(self), type(actions), actions))
        return self.cActions

    def disable_scale(self):
        self.scale_disabled = 1

    def enable_scale(self):
        self.scale_disabled = 0

    def scale(self):
        if hasattr(self, "scale_disabled") and self.scale_disabled:
            return 1
        return super(MonkeyCity, self).scale()

    def do(self,force=None):
        #doesn't work
        self.disable_scale()
        super(MonkeyCity,self).do(force)
        self.enable_scale()

    def doToken(self, token):
        self.disable_scale()
        print("do %s" % token)
        actions = []
#        actions = self.make_button_taps([token], False)
        for action in actions:
            if isinstance(action, dict):
                #print("%s - %s" % (action["name"], action["action"]))
                QtTest.QTest.qWait(100)
                exec(action["action"])
        self.enable_scale()
        return token


def rectContains(rect,pt):
    logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return logic

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_text_signal = pyqtSignal(dict)
    canny_lower_signal = pyqtSignal(int)
    canny_upper_signal = pyqtSignal(int)
    thresh_lower_signal = pyqtSignal(int)
    thresh_upper_signal = pyqtSignal(int)
    battle_signal = pyqtSignal(tuple)
    force_mode_actions_signal = pyqtSignal(bool)

    def __init__(self,devN=0,lower=None, upper=None, lowerT=None, upperT=None,verbose=vv, actions=None):
        super().__init__()
        self._run_flag = True
        self.frame_count = 0
        self.devN = devN
        self.reopenVideoFlag = False
        self.timeout = 5
        self.cardsByTile = {}
        self.show_tile_grid = False
        self.scale_percent = 100 # percent of original size
        if (actions):
            self.actions = actions
        else:
            self.actions = Actions()
        #self.scale = self.actions.scale()
        #self.device_width = int(self.actions.device_width())
        #self.device_height = int(self.actions.device_height())
        self.method="edge"
        self.verbose = verbose
        self.image_options = None
        self.mode_options = None
        self.cannyLower = lower
        self.cannyUpper = upper
        self.threshLower = lowerT
        self.threshUpper = upperT
        self.worldThreshLower = 42 #great for high contrast mode balor
        self.bars_hpmpward = [((168,320),"red"),((168,332),"blue"), ((168,344), "purple")]
        self.barwh = (46,6)
        self.bars_hpsummons = [ (24,368), (24,272), (96,236), (240,248), (240,392) ]
        self.health_summons = []
        self.battleStart = datetime.now()
        self.battleEnd = None
        self.ward_turns = 0
        self.health_percent = 0
        self.mana_percent = 0
        self.ward_percent = 0
        self.summons = 0
        self.turns = 0
        self.action = ""
        self.actionStack = ["db"]
        self.lastTurn = None
        self.red_x_btn_template = cv.imread('templates/red_x_button.png', 0)
        self.xbtn_template = cv.imread('templates/x_btn_black_10px_border.png', 0)
        self.origintown_template = cv.imread('templates/origin_town.png', 0)
        self.balor_template = cv.imread('templates/balor.png', 0)
        self.warpgate_template = cv.imread('templates/warp_gate.png', 0)
        self.continuebtn_template = cv.imread('templates/continue_button.png', 0)
        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        self.net = cv.dnn.readNet("frozen_east_text_detection.pb")


    def fast_east(self,image,min_confidence=0.5):
        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (224, 64)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
    #    print("[INFO] text detection took {:.6f} seconds".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_confidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x])) 
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # scale the bounding box coordinates based on the respective
                # ratios
                #startX = int(startX * rW)
                #startY = int(startY * rH)
                #endX = int(endX * rW) + 2
                #endY = int(endY * rH) + 2

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        rects = []
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW) + 2
            endY = int(endY * rH) + 2

            rects.append((startX, startY, endX, endY))

        return rects

    def angle_cos(self,p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

    def tess(self, img,whitelist=""):
        words = []
        chars= "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_."
        if len(whitelist) > 0:
            chars = whitelist
        tconf = "--oem 3 --psm 8 "+ \
            "-c tessedit_char_whitelist=\"%s\"" % chars
        
        try:
            text = pytesseract.image_to_data(img, config=tconf, output_type='data.frame')
            text = text[text.conf != -1]
            if LogLevel.is_ocrtext(self.verbose):
                print(text)
#                                    print()
            try:
                #level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text        
                lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: x.values.tolist()).tolist()
                confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
            
                tl = []
                tc = []
                for i in range(len(lines[0])):
                    if str(lines[0][i]).strip():
                        tl.append(str(lines[0][i]))

                line = " ".join(tl)  #more likely multiple words, but still can be single word
#                        print(line)

                conf = 0
                if (np.any(confs)):
                    conf = round(np.mean(confs),2)
                    if LogLevel.is_ocrtext(self.verbose):
                        print(conf)
                if (conf >= 10):
                    words.append(line)

            except BaseException as err:
                if LogLevel.is_ocrtext(self.verbose):
                    print(f"Unexpected {err=}, {type(err)=}")
                    print(text)
        except BaseException as err:
            if LogLevel.is_ocrtext(self.verbose):
                print(f"Unexpected {err=}, {type(err)=}")

        return words

    #  returns cb []
    #      center coords of bounding box which fast_east algo returns
    #  words []
    #    the words tesseract read inside that box with a 50% confidence or above
    
    def fast_east_tess(self, image):
        cb = []
        words = []
        boxes = self.fast_east(image)

        boxc = 0
        for box in sorted(boxes, key=lambda x: x[0]):
            boxc += 1
           
            (x, y, x2, y2) = box
#                x, y, w, h = unscaled
                #crop
 #               cardimage=orig[y: y + h, x : x + w]
            if LogLevel.is_ocrtext(self.verbose):
                print("x[%d] y[%d] x2[%d] y2[%d]" % (x,y,x2,y2))
            img = image[y:y2,x:x2]

#            if (img is None):
#                print("Empty!")
#            print(box, boxc,"cards/%d_%d_%d_%d.title.%d.png" % (tile[0], tile[1], tile[2], tile[3], boxc))
#            print(type(img))
            
            # draw the bounding box on the image
#            cv.rectangle(im2, (x, y), (x2, y2), (0, 255, 0, 1))

            tconf = "--oem 3 --psm 7 "+ \
                "-c tessedit_char_whitelist=\"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-_.\""
            
            try:
                text = pytesseract.image_to_data(img, config=tconf, output_type='data.frame')
                text = text[text.conf != -1]
#                                    print()
                if LogLevel.is_ocrtext(self.verbose):
                    print(text)
                try:
                    #level,page_num,block_num,par_num,line_num,word_num,left,top,width,height,conf,text        
                    lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(lambda x: x.values.tolist()).tolist()
                    confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
                
                    tl = []
                    tc = []
                    for i in range(len(lines[0])):
                        if str(lines[0][i]).strip():
                            tl.append(str(lines[0][i]))

                    line = " ".join(tl)  #more likely multiple words, but still can be single word
#                        print(line)

                    conf = 0
                    if (np.any(confs)):
                        conf = round(np.mean(confs),2)
#                        print(conf)
                    if (conf > 50):
                        words.append(line)
                        cb.append((x+(x2-x)/2,y+(y2-y)/2))

                except BaseException as err:
                    if LogLevel.is_ocrtext(self.verbose):
                        print(f"Unexpected {err=}, {type(err)=}")
                        print(text)
            except BaseException as err:
                if LogLevel.is_ocrtext(self.verbose):
                    print(f"Unexpected {err=}, {type(err)=}")
        return (cb, words)
#        return (cb, words)

    def find_mobs(self, img):
        # denoise before edge detection
        #blur = cv.GaussianBlur(img,(5,5),0)

        #  normalize color histogram for greater contrast
        #
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(img)

        median_pix = np.median(img)
#        self.cannyLower = int(max(0 ,0.7*median_pix))
#        self.canny_lower_signal.emit(self.cannyLower)

#        self.cannyUpper = int(min(255,1.3*median_pix))
#        self.canny_upper_signal.emit(self.cannyUpper)

#        print(median_pix, self.cannyLower, self.cannyUpper)

        edge = cv.Canny(contrast, self.cannyLower, self.cannyUpper)#, apertureSize=5)

        contours, hierarchy = cv.findContours(edge, 
            cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        mobs=[]
        for cnt in contours:
            (x,y,w,h)=cv.boundingRect(cnt)
            if y < 160 and x < 55:
                continue
            if y > 450 and x < 150:
                continue
            if x > 800 and y > 445:
                continue

            area = w * h
            if area > 200 and area < 500:
                mobs.append((x,y,w,h))
            if area > 900 and area < 1600:
                mobs.append((x,y,w,h))

        npcenter = np.array((490,310)) # spell menu x button location with 9 spells
#        dist = int(np.linalg.norm(npa - npb))
        mobs.sort(key = lambda e: int(np.linalg.norm(np.array((e[0],e[1])) - npcenter))) 
        return mobs

    def find_squares(self, img, thresh=None):
        if thresh is None:
            # denoise before edge detection
            blur = cv.GaussianBlur(img,(5,5),0)

            #  normalize color histogram for greater contrast
            #
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(img)

            median_pix = np.median(img)
            if (self.cannyLower is None):
                self.cannyLower = int(max(0 ,0.7*median_pix))
                self.canny_lower_signal.emit(self.cannyLower)

            if (self.cannyUpper is None):
                self.cannyUpper = int(min(255,1.3*median_pix))
                self.canny_upper_signal.emit(self.cannyUpper)

    #        print(median_pix, self.cannyLower, self.cannyUpper)

            edge = cv.Canny(contrast, self.cannyLower, self.cannyUpper)#, apertureSize=5)
        else:
            edge = thresh

        if (self.image_options == "Edge C"):
            im3 = img.copy()
            contours, hierarchy = cv.findContours(edge, 
                cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                cv.drawContours(im3, contours, i, (255, 0, 0), \
                                1, cv.LINE_8, hierarchy, 0)
            self.change_pixmap_signal.emit(im3)

        # dilate canny output to remove potential
        # holes between edge segments
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 8))
#               edge = cv.erode(edge, rect_kernel, iterations=1)
        edge = cv.dilate(edge, rect_kernel, iterations=1)
        rect_kernel = cv.getStructuringElement(cv.MORPH_RECT, (8, 1))
#               edge = cv.erode(edge, rect_kernel, iterations=1)
        edge = cv.dilate(edge, rect_kernel, iterations=1)

        #im2 = np.uint8(im2/2.)
        #im2[edge != 0] = (0,255,0)
        contours, hierarchy = cv.findContours(edge, 
            cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        if (self.image_options == "Sq Edge"):
            self.change_pixmap_signal.emit(edge)
        elif (self.image_options == "Sq Edge Contours"):
            im3 = img.copy()
            for i in range(len(contours)):
                cv.drawContours(im3, contours, i, (255, 0, 0), \
                                2, cv.LINE_8, hierarchy, 0)
            self.change_pixmap_signal.emit(im3)

        rect=[]
        for cnt in contours:
            approx = cv.approxPolyDP(cnt,0.02*cv.arcLength(cnt,True),True)
            if (len(approx)==4 and abs(cv.contourArea(cnt)) > (self.tileW * self.tileH)):
                #and abs(cv.contourArea(cnt)) <  ):
#                        print(abs(cv.contourArea(cnt)))
                cnt = approx.reshape(-1, 2)
                max_cos = np.max([self.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                if max_cos < 0.1:
                    rect.append(cnt)
        return rect


    def read_percent_bar(self,image, mask):
        width = image.shape[1] - 1
        height = image.shape[0] - 1 

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        masks = {}
        masks['blue'] = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([120,255,255]))
        red_low = cv2.inRange(hsv, np.array([0, 90, 20]), np.array([10,255,255]))
        red_high = cv2.inRange(hsv, np.array([175, 90, 20]), np.array([185,255,255]))
        masks['red'] = red_low + red_high
        masks['purple'] = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([140,255,255]))

#  test for non exist or 0%
#        masks['black']
        
        masked = cv2.bitwise_and(image,image, mask = masks[mask])

        return round(np.count_nonzero(np.sum(np.sum(masked,0),1) > 0) / masked.shape[1] * 100, 1)


    def run(self):
#if 'size' in params:
#    w, h = map(int, params['size'].split('x'))
#vid.set(cv.CAP_PROP_FRAME_WIDTH, 2400)
#vid.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
#        r = self.actions.startGame()
#        if "bail" in r:
#            print(r)
#            sys.exit()

        cap = cv.VideoCapture(self.devN)
        waited=0
        feedback = ""
        feedback2 = ""
        while self._run_flag:
            _ret, orig = cap.read()
            if (orig is None or self.reopenVideoFlag is True):
                if LogLevel.is_main(self.verbose):
                    print("VideoThread::run reopening video capture device")
                self.reopenVideoFlag = False
                time.sleep(0.1) 
                waited=waited+0.1
                if (waited > self.timeout):
                    cap = cv.VideoCapture(self.devN)
                    waited=0
                continue
            else:
                self.frame_count += 1



##################################
#  Main Event Loop
#
        #    if ((frame_count % 2) != 0):
        #        continue

            #  classically OCR uses white as text foreground and black for background
            #  
            #  MTG typically has black text on a lighter color bg
            #
        #    print('Original Dimensions : ',img.shape)
            self.prev_scale = self.scale_percent
#            scale_percent = cv.getTrackbarPos('scale', 'edge')
            self.width = orig.shape[1]
            self.height = orig.shape[0]
            width = int(orig.shape[1] * self.scale_percent / 100)
            height = int(orig.shape[0] * self.scale_percent / 100)
            dim = (width, height)

            t = self.clock()

            #  split frame into tiles so that we can have an area 
            #  for detected cards to occupy for reference
            #
            ih, iw, ic = orig.shape

            img = cv.resize(orig, dim, interpolation = cv.INTER_AREA)
            im2 = img.copy()

            #  with second plus framerates, show something on program start
            #
            if (self.frame_count == 1 or self.image_options == None or self.image_options == "V4L2"):
                self.change_pixmap_signal.emit(img)

            #try 10 by 10 grid
            rows, cols = 10,10
            tilesize = (int(ih//rows), int(iw//cols))
            self.tilesize = tilesize
            self.tileRows = 10
            self.tileCols = 10
            self.tileW = tilesize[1]
            self.tileH = tilesize[0]

        #    print(ih,iw,tilesize)
        #    print("rows %d, cols %d" % (orig.rows, orig.cols))

            #  first part is card shape detection
            #

        #    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
        #                	cells_per_block=(2, 2), visualize=True, multichannel=True)

        #    cv.imshow('hog', hog_image)

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # denoise before edge detection
            blur = cv.GaussianBlur(gray,(5,5),0)

            # 4. Dilate the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))

            # 5. Only take different areas that are different enough (>20 / 255)
            threshb_frame = cv.threshold(src=blur, thresh=self.threshLower, maxval=self.threshUpper, type=cv.THRESH_BINARY)[1]
            threshw_frame = cv.threshold(src=blur, thresh=self.threshLower, maxval=self.threshUpper, type=cv.THRESH_BINARY)[1]

            
            if (self.image_options in ["World"]) and self.battleEnd != None:
                
                r = self.find_mobs(blur)
                for mob in r:
                    (x,y,w,h) = mob
                    self.actions.tap(x+int(w/2), y + int(h/2))
                    QtTest.QTest.qWait(500)
                    area = w * h
                    if area > 200 and area < 500:
                       cv.rectangle(im2, (x,y), (x + w, y + h), (255,0,0), 2)
                    else:
                       cv.rectangle(im2, (x,y), (x + w, y + h), (0,255,0), 2)

            if (self.image_options == "Thresh"):
                self.change_pixmap_signal.emit(threshb_frame)

        #    cv.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        #    ret, thresh1 = cv.threshold(gray, 175, 255, cv.THRESH_BINARY)
            
            #  locate X button
            #
            w, h = self.red_x_btn_template.shape[::-1]
            res = cv.matchTemplate(gray, self.red_x_btn_template, cv.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)

            foundX = False
            for pt in zip(*loc[::-1]):
                cv.rectangle(im2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#distance from center of bounds box of match over 0.8 threshold to center of where button should be
                x = int(pt[0] + w / 2)
                y = int(pt[1] + h / 2)
                dist = int(np.linalg.norm(np.array((x,y)) - np.array((216,115))))
                if dist < 40: #battle button
                    cv.circle(im2, (x, y), 2 ,(0,0,255), 2)
                    print("\n\n!!!!found red X button!! pressing (%i, %i)" % (x, y))
                    self.actions.hold(x,y, 0.1)


            w, h = self.continuebtn_template.shape[::-1]
            res = cv.matchTemplate(gray, self.continuebtn_template, cv.TM_CCOEFF_NORMED)
            _minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res, None)
            loc = maxLoc
            if maxVal > 0.8:
                #print(loc, (w,h))
                cv.rectangle(im2, loc, (loc[0] + w, loc[1] + h), (0,0,255), 2)
                if (self.image_options in ["World"] ):
                    dist = int(np.linalg.norm(np.array(loc) - np.array((376,315))))
                    if dist < 25: #battle button
                         self.actions.tap(loc[0] + int(w / 2),loc[1] + int(h / 2))

                dist = int(np.linalg.norm(np.array(loc) - np.array((376,492))))
                if dist < 20: #continue button
                    if self.battleEnd == None:
                        self.battleEnd = datetime.now()
                        self.battle_signal.emit((self.battleStart, self.battleEnd))
                    if (self.image_options in ["World"] ):
                        self.actions.tap(loc[0] + int(w / 2),loc[1] + int(h / 2))
                        QtTest.QTest.qWait(1000)
                        print("tried to autoheal")
                        self.actions.hold(827,569, 2) #auto-heal (d-pad)
#                        self.actions.hold(567,558, 2) #auto-heal (tap to move mode)

            w, h = self.origintown_template.shape[::-1]
            res = cv.matchTemplate(gray, self.origintown_template, cv.TM_CCOEFF_NORMED)
            _minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res, None)
            loc = maxLoc
            if maxVal > 0.8:
                cv.rectangle(im2, loc, (loc[0] + w, loc[1] + h), (0,0,255), 2)

            w, h = self.balor_template.shape[::-1]
            res = cv.matchTemplate(gray, self.balor_template, cv.TM_CCOEFF_NORMED)
            _minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(res, None)
            loc = maxLoc
            if maxVal > 0.8:
                cv.rectangle(im2, loc, (loc[0] + w, loc[1] + h), (0,0,255), 2)
#                self.actions.hold((1140,1110), 2) #auto-heal

            w, h = self.warpgate_template.shape[::-1]
            res = cv.matchTemplate(gray, self.warpgate_template, cv.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)

            for pt in zip(*loc[::-1]):
                cv.rectangle(im2, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

            #hp_template = cv.imread('templates/hp_bar_empty_alpha.png', 0)
            #w, h = hp_template.shape[::-1]
            #res = cv.matchTemplate(gray, hp_template, cv.TM_CCOEFF_NORMED)
            #threshold = 0.8
            #loc = np.where( res >= threshold)
            #for pt in zip(*loc[::-1]):
            #    cv.rectangle(im2, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
            # locate world bosses


            if self.mode_options == "Canny" or self.mode_options == None:
                rect = self.find_squares(blur)
            if self.mode_options == 'Thresh':
                rect = self.find_squares(blur, threshb_frame)

#            print("width %s height %s" % (self.width, self.height)) 
            #hpbar = nmpy.array([165,317,216,328])

            stats = []
            for stat in self.bars_hpmpward:
                (loc, color) = stat

                barimg = orig[loc[1]:loc[1]+self.barwh[1],loc[0]:loc[0]+self.barwh[0]]
                stats.append( self.read_percent_bar(barimg,color) )

            (self.health_percent, mana, self.ward_percent) = stats
            if mana > 0:
                self.mana_percent = mana



            if (self.image_options in ["Rects", "CollectGold"]):
                cardnum=0
                for cnt in rect:
                    cardnum += 1
                    box=cv.boundingRect(cnt)
                    words=[]
                    cb=[]
                    cv.drawContours(im2,[cnt],0,(0,0,255),3)
                    if  (self.image_options in ["CollectGold", "Rects"] ):
                        unscaled = tuple(int(n / (self.scale_percent / 100)) for n in box)
        #                print(unscaled)
                        x, y, w, h = unscaled
                        cx = x + (w / 2)
                        cy = y + (h / 2)
                        boximage=orig[y: y + h, x : x + w]

                        (cb,words) = self.fast_east_tess(boximage)

                        if len(words):
                            if LogLevel.is_ocrtext(self.verbose):
                                print(" ".join(words))
                        bt = " ".join(words)


            if self.show_tile_grid:
                #  display tile grid
                #
                showts = tuple(int(n * self.scale_percent / 100) for n in tilesize)
                tilecount = 0
                for col in range(cols):
                    for row in range(rows):
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

            if (self.image_options in ["CollectGold"]):
                action = self.actions.action()
                if isinstance(action, dict):
                    if ('name' in action and action['name'] != 'noop'):
                        text = "%s - %s" % (action["name"], action["action"])
                        feedback = text
                cv.putText(im2, feedback, (20, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                r = self.actions.do()

            if len(feedback) > 0:
                cv.putText(im2, feedback, (20, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            #  show framerate (time between processed frames)
            #
            dt = self.clock() - t
            cv.putText(im2, 'time: %.1f ms (%.1f)fps' % (dt*1000, 1/dt), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            
#            cv.imshow(method, im2)

            #  escape key
#            if cv.waitKey(5) == 27:
#                break
#            if (self.image_options
            if (self.image_options in ["Rects", "CollectGold", "Raid", "Battle", "Stats", "World"]):
                self.change_pixmap_signal.emit(im2)
#            print("emitting change_table with:")
#            print(self.cardsByTile)

        # shut down capture system
        cap.release()

    def clock(self):
        return cv.getTickCount() / cv.getTickFrequency()

    def get_tile(self,x,y,w,h, tilesize):
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

        return x1+1, y1+1, tw, th

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    #  and checkbox is clicked, actions get's reconstructed with mode
    #
    def set_image_options(self, option):
        if LogLevel.is_main(self.verbose):
            print("VideoThread: image_options: ", option)
        self.image_options = option
        self.health_summons = []
        self.battleEnd = datetime.now()
        if (self.image_options == 'World'):
            self.actions.tap(80,522) #d-pad center
            QtTest.QTest.qWait(2000)
        if (self.image_options == 'Battle'):
            self.turns = 0
            self.battleStart = datetime.now()
            self.battleEnd = None
            self.action = ""
        else:
            self.battleEnd = datetime.now()
        if (self.image_options in ["CollectGold", "Raid"]):
            self.turns = 0
            self.battleStart = datetime.now()
            self.battleEnd = None
            p = self.actions.parent
            self.actions = MonkeyCity(mode=self.image_options, verbose=self.verbose, parent=p, modeOptions=self.actions.modeOptions)
        self.battle_signal.emit((self.battleStart, self.battleEnd))

    def set_mode_options(self, option):
        if LogLevel.is_main(self.verbose):
            print("VideoThread: mode_options: ", option)
        self.mode_options = option

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, uri, args, key):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = self.request
        self.uri = uri
        self.args = args
        self.key = key
        self.signals = WorkerSignals()

    def request(self):
        print("sending request to %s with %s" % (self.uri, self.args))
        response = requests.get(self.uri+self.args)
        json_response = json.loads(response.text)
        json_response["request"] = {"uri":self.uri, "args":self.args, "key":self.key}
        return json_response

    def print_output(self):
        print(self.response)

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.

        # Retrieve args/kwargs here; and fire processing using them
        '''
        try:
            result = self.request()
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class RequestPool(QObject):
    on_request_result = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()
        self.uri = "https://api.scryfall.com/cards/named?fuzzy="
        self.counter = 0


    def recurring_timer(self):
        self.counter +=1

        # Execute
#        self.threadpool.start(worker)

    def result(self, response):
        (key, args, name, image, price, link) = (None, None, None, None, None,None)
        print("request pool result for %s: " % response["request"]["args"])

        args = response["request"]["args"]
        key = response["request"]["key"]

#        {'object': 'error', 'code': 'not_found', 'status': 404, 'details': 'No cards found matching “ein Eldest Redorn”'}
        try:
            print(" name  : %s" % response["name"])
            name = response["name"]
        except:
            print(response)
            return

        try:
            print(" image : %s" % response["image_uris"]["small"])
            image = response["image_uris"]["small"]
        except:
            pass

        try:
            print(" price : %s" % response["prices"]["usd"])
            price = response["prices"]["usd"]
        except:
            pass
  
        try:
            link = response["related_uris"]["gatherer"]
        except:
            pass
  
        card = {"key":key, "args":args, "name": name, "image":image, "price":price, "link": link}
        print("emit on_request_result signal ", card)
        self.on_request_result.emit(card)

        

    def make_request(self,arg,key):
        print(f"RequestPool make_request:{arg}")
        # Pass the function to execute
        uri = self.uri + arg
        worker = Worker(self.uri,arg,key) # Any other args, kwargs are passed to the run function
#        worker.signals.finished.connect(self.thread_complete)
#        worker.signals.progress.connect(self.progress_fn)
        worker.signals.result.connect(self.result)
        # Execute
        self.threadpool.start(worker)


class TransparentOverlay(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self._updateParent(self.parentWidget())

    def setParent(self, parent, *args):
        prevParent = self.parentWidget()
        super().setParent(parent, *args)
        self._updateParent(parent, prevParent)

    def unsetParent(self, parent=None):
        if parent is None:
            parent = self.parentWidget()
        if parent is not None and hasattr(parent.resizeEvent, '_original'):
            parent.resizeEvent = parent.resizeEvent._original

    def _updateParent(self, parent, prevParent=None):
        if parent is not prevParent:
            self.unsetParent(prevParent)
            if parent is not None:
                original = parent.resizeEvent
                def resizeEventWrapper(event):
                    original(event)
                    self.resize(event.size())
                resizeEventWrapper._original = original
                parent.resizeEvent = resizeEventWrapper
                self.resize(parent.size())

proc = None
class ImageControl(QWidget):
    check_option_signal = pyqtSignal(str)
    check_mode_option_signal = pyqtSignal(str)

    def imageOptionClicked(self, state):
        isChecked = bool(state)
        if (isChecked):
            cb = self.checkBoxes.checkedButton()
            if (cb):
                if LogLevel.is_image(self.verbose):
                    print("imageOptionClicked: ",cb.text())
                self.check_option_signal.emit(cb.text())
                self.image_options = cb.text()
            else:
                print("unknown button checked")

    def modeOptionClicked(self, state):
        isChecked = bool(state)
        if (isChecked):
            cb = self.modeOptions.checkedButton()
            if (cb):
                if LogLevel.is_image(self.verbose):
                    print("modeOptionClicked: ",cb.text())
                self.check_mode_option_signal.emit(cb.text())
            else:
                print("unknown button checked")

    def setThreshLower(self):
        self.thread.threshLower = self.t3Slider.value()
    def setThreshUpper(self):
        self.thread.threshUpper = self.t4Slider.value()
    def updateThreshLower(self, value):
        self.t3Slider.setValue(value)
    def updateThreshUpper(self, value):
        self.t4Slider.setValue(value)
    def setCannyLower(self):
        self.thread.cannyLower = self.t1Slider.value()

    def updateBattle(self,startend):
        self.battleStart = startend[0]
        self.battleEnd = startend[1]

    def updateCannyLower(self, value):
        self.t1Slider.setValue(value)

    def updateForceModeActions(self, state):
        self.forceModeActionsChk.setChecked(state)

    #  the signals exist so VideoCapure thread can emit a signal to interact
    #  with ImageControl gui
    #  ImageControl can just operate on thread object directly
    #
    def forceModeActions(self, state):
        isChecked = bool(state)
        self.updateForceModeActions(isChecked)
        self.thread.actions.forceModeActionsFlag = isChecked
        self.thread.actions.reset()
        print("ImageControl forceModeActions get/set, value %r" % self.thread.actions.forceModeActionsFlag )

    def setCannyUpper(self):
        self.thread.cannyUpper = self.t2Slider.value()

    def updateCannyUpper(self,value):
        self.t2Slider.setValue(value)

    def setSliderLabels(self):
        self.t1Label.setText("%d" % self.t1Slider.value())
        self.t2Label.setText("%d" % self.t2Slider.value())
        self.t3Label.setText("%d" % self.t3Slider.value())
        self.t4Label.setText("%d" % self.t4Slider.value())


    def _on_resized(self, event):
        '''
        print(ppretty(event, indent='    ', depth=2, width=30, seq_length=6,
              show_protected=True, show_private=False, show_static=True,
              show_properties=True, show_address=True))
        '''
        ow = event.oldSize().width()
        oh = event.oldSize().height()

        w = event.size().width()
        h = event.size().height()

        wd = w - ow
        hd = h - oh

        print(event.oldSize(), event.size())
        print((ow,oh), (w, h), (wd, hd))

#        self.display_width += wd
#        self.display_height += hd

#        self.image_label.resize(self.display_width, self.display_height)

    #def __init__(self,devN=0,lower=None, upper=None, verbose=False):
    def __init__(self, actions=None):
        super().__init__()
        if (not actions):
            actions = Actions()
            print("not ??")
        else:
            print(type(actions))
        self.actions = actions
        #(self.device_width,self.device_height,self.X11_width,self.X11_height) = self.actions.getResolutions()
        self.scale = 0
#self.actions.scale()

        #  get this from video thread
        self.display_width = 960
        self.display_height = 600
        self.image_options = 'CollectGold'

        self.requestPool = RequestPool()

        self.resizeEvent = (lambda old_method: (lambda event: (self._on_resized(event), old_method(event))[-1]))(self.resizeEvent)
        #serial = "FAMVRW9D9HDEHEWS" #BLU Pure

        parser = optparse.OptionParser()
        parser.set_defaults(verbose=0)
        parser.add_option('-a', '--watchad', action='store_true', dest='watchad')
        
        parser.add_option('-v', '--verbose', type="int", dest='verbose')
        parser.add_option('--device', type="int", dest='device',
                         help="video device # to read from", default="0")
        parser.add_option('-l', '--lower', type="int", dest='lower', \
                            help="lower canny threshold for rectangle detection processing")
        parser.add_option('-u', '--upper', type="int", dest='upper', \
                            help="upper canny threshold")
        (o, args) = parser.parse_args()
        print(o)

        self.verbose = 0
        if o.verbose:
            self.verbose = LogLevel(o.verbose)

        self.devN = 0
        if o.device:
            self.devN = int(o.device)

#        self.watchad = False
        #if 'watchad' in o:
        if o.watchad != None:
            print('watch ad option present')
            print("modeOptions.WatchAd.disabled = %r" % self.actions.modeOptions['WatchAd']['disabled'])
            self.actions.modeOptions['WatchAd']['disabled'] = not bool(o.watchad)
            print("set modeOptions WatchAd disabled to %r" % self.actions.modeOptions['WatchAd']['disabled'])

        self.cannyLower = None
##        self.cannyLower = 50
        self.cannyLower = 40
        if o.lower:
            self.cannyLower = int(o.lower)

        self.cannyUpper = 237
        if o.upper:
            self.cannyUpper = int(o.upper)

        self.threshLower = 10 
        self.threshLower = 52 #great for high contrast mode balor
        self.threshUpper = 255

        self.replayFramesMax = 25 
        self.replayFrames = []
        self.replayFreq = 0.1
        self.replayLast = None
        self.replaying = False
        self.battleStart = None
        self.battleEnd = None

        self.setWindowTitle("Image Processing Control")
        self.timeout=5
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

#        self.image_label.enterEvent = lambda e: 
#        self.image_label.leaveEvent = lambda e: print('goodbye')
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = lambda e: self.lblXY.setText('X, Y ( %d, %d ) x2 ( %d, %d )' % (e.x(), e.y(), e.x() * 2, e.y() * 2))

        # create a text label
        self.lblVdev = QLabel('/dev/video%d' % self.devN)
        self.lblXY = QLabel('')
        self.forceModeActionsChk = QCheckBox('Force Actions')
        self.forceModeActionsChk.stateChanged.connect(self.forceModeActions)

        self.snapBtn = QPushButton('Snapshot')
        self.takeSnapshot = False
        self.snapBtn.clicked.connect(self.snapshot)

        self.t1Slider = QSlider(Qt.Horizontal)
        self.t2Slider = QSlider(Qt.Horizontal)
        self.t3Slider = QSlider(Qt.Horizontal)
        self.t4Slider = QSlider(Qt.Horizontal)
        self.t1Slider.setMinimum(0)
        self.t2Slider.setMinimum(0)
        self.t3Slider.setMinimum(0)
        self.t4Slider.setMinimum(0)
        self.t1Slider.setMaximum(255)
        self.t2Slider.setMaximum(255)
        self.t3Slider.setMaximum(255)
        self.t4Slider.setMaximum(255)
        try:
            self.t1Slider.setValue(self.cannyLower)
        except:
            self.t1Slider.setValue(0)

        try:
            self.t2Slider.setValue(self.cannyUpper)
        except:
            self.t2Slider.setValue(255)

        try:
            self.t3Slider.setValue(self.threshLower)
        except:
            self.t3Slider.setValue(0)

        try:
            self.t4Slider.setValue(self.threshUpper)
        except:
            self.t4Slider.setValue(0)


        self.t1Label = QLabel("%d" % self.t1Slider.value())
        self.t2Label = QLabel("%d" % self.t2Slider.value())
        self.t3Label = QLabel("%d" % self.t3Slider.value())
        self.t4Label = QLabel("%d" % self.t4Slider.value())
        self.t1Slider.setTickPosition(QSlider.TicksBelow)
        self.t2Slider.setTickPosition(QSlider.TicksBelow)
        self.t3Slider.setTickPosition(QSlider.TicksBelow)
        self.t4Slider.setTickPosition(QSlider.TicksBelow)
        self.t1Slider.setTickInterval(5)
        self.t2Slider.setTickInterval(5)
        self.t3Slider.setTickInterval(5)
        self.t4Slider.setTickInterval(5)
        self.t1Slider.valueChanged.connect(self.setSliderLabels)
        self.t2Slider.valueChanged.connect(self.setSliderLabels)
        self.t3Slider.valueChanged.connect(self.setSliderLabels)
        self.t4Slider.valueChanged.connect(self.setSliderLabels)
        self.t1Slider.sliderReleased.connect(self.setCannyLower)
        self.t2Slider.sliderReleased.connect(self.setCannyUpper)
        self.t3Slider.sliderReleased.connect(self.setThreshLower)
        self.t4Slider.sliderReleased.connect(self.setThreshUpper)


        self.checkBoxes = QButtonGroup()
        self.modeOptions = QButtonGroup()
        #'image options')

        replayBtn = QPushButton('Replay')
        replayBtn.clicked.connect(self.replay_clicked)

        hbox = QHBoxLayout()
        hbox.addWidget(self.forceModeActionsChk)

        hbox.addWidget(self.snapBtn)
        hbox.addWidget(replayBtn)
        options = [QCheckBox("CollectGold"),QCheckBox("V4L2"), QCheckBox("Rects"), QCheckBox("Thresh"),  QCheckBox("Edge C"), QCheckBox("Sq Edge"), QCheckBox("Sq Edge Contours")]

        options[0].setChecked(True)
        for i in range(len(options)):
            hbox.addWidget(options[i])
            self.checkBoxes.addButton(options[i], i)
            options[i].stateChanged.connect(self.imageOptionClicked)

        opt2 = [QCheckBox('Thresh'), QCheckBox('Canny')]
        hboxmo = QHBoxLayout()

        opt2[1].setChecked(True)
        for i in range(len(opt2)):
            hboxmo.addWidget(opt2[i])
            self.modeOptions.addButton(opt2[i], i)
            opt2[i].stateChanged.connect(self.modeOptionClicked)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addLayout(hbox)
        vbox.addLayout(hboxmo)
        hbt1 = QHBoxLayout()
        hbt2 = QHBoxLayout()
        hbt3 = QHBoxLayout()
        hbt4 = QHBoxLayout()
        hbt1.addWidget(self.t1Label)
        hbt1.addWidget(self.t1Slider)
        hbt2.addWidget(self.t2Label)
        hbt2.addWidget(self.t2Slider)
        hbt3.addWidget(self.t3Label)
        hbt3.addWidget(self.t3Slider)
        hbt4.addWidget(self.t4Label)
        hbt4.addWidget(self.t4Slider)
        vbox.addLayout(hbt1)
        vbox.addLayout(hbt2)
        vbox.addLayout(hbt3)
        vbox.addLayout(hbt4)
        vbox.addWidget(self.t2Slider)
        hhbox = QHBoxLayout()
        hhbox.addWidget(self.lblVdev)
        hhbox.addWidget(self.lblXY)
        vbox.addLayout(hhbox)
        
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        print("capture /dev/video%d\n" % self.devN)

    @pyqtSlot(dict)
    def validate_text(self, card):

        #camecase, join with space
        phrase = ""
        for word in card["text"]:
            if (word.istitle()):
                phrase = "+".join([phrase, urllib.parse.quote(word)])
            else:
                phrase = "".join([phrase, urllib.parse.quote(word)])

        #even though this is likely the correct construction of the phrase,
        # there is a chance tesseract had a better OCR on one word over the other

        print("validate_text: make_request phrase %s -> %s" % ("".join(card["text"]), phrase))
        if (float(card["mconf"]) > -1):
            print(card['mconf'])
            self.requestPool.make_request(phrase, card["key"])
        else:
            print(card['mconf'])


    def run(self):
        #  will block for either timeout or time until success string outputs
        #  proc used in handler
        (proc, q) = self.v4l2sink()

#        self.isRunningOrStart() #open camera on android device


        # create the video capture thread
        self.thread = VideoThread(devN=self.devN,lower=self.cannyLower,upper=self.cannyUpper, lowerT=self.threshLower, upperT=self.threshUpper, verbose=self.verbose, actions=self.actions)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.check_option_signal.connect(self.thread.set_image_options)
        self.check_mode_option_signal.connect(self.thread.set_mode_options)

        #unused from VideoThread as a setter
        self.thread.force_mode_actions_signal.connect(self.forceModeActions)

        self.thread.canny_lower_signal.connect(self.updateCannyLower)
        self.thread.canny_upper_signal.connect(self.updateCannyUpper)
        self.thread.thresh_lower_signal.connect(self.updateThreshLower)
        self.thread.thresh_upper_signal.connect(self.updateThreshUpper)
        self.thread.battle_signal.connect(self.updateBattle)

        self.thread.change_text_signal.connect(self.validate_text)

        #  wait for signals to be detected so initial state is set
        #
        # start the thread
        self.thread.start()

        # !!! checked by default now!!!
#meh just call the function the signal triggers
#        self.forceModeActions(state=True)
#        self.forceModeActionsChk.setChecked(True)
        #self.forceModeActions(True)

    #  check for existing scrcpy using v42l-sink
    #  if non, start one
    #    TODO: handle errors from scrcpy
    #
    def v4l2sink(self,devN=0):

        #  check for process being run externally
        #
        try:
            fuser = subprocess.run(['fuser', "/dev/video%d" % devN], check=True, capture_output=True, text=True)
            if (len(fuser.stdout) > 0):
                m = re.match(r".*?:?\s*(?P<pid>\d+).*", fuser.stdout)
                ps = subprocess.run(['ps','-q', str(m.group('pid')), '-o', 'user=,pid=,state=,tname=,time=,command='], check=True, capture_output=True, text=True)
                if (len(ps.stdout) > 0):
                    print("v4l2 device video%d in use by the following:\n" % devN, ps.stdout)
                    return (False,False)
        except subprocess.CalledProcessError as e:
            if (e.returncode == 1 and len(e.output) == 0):
                print("v4l2 device has no current user")
            else:
                print(e.returncode)
                print(e.output)
        except BaseException as err:
            previous_frame = inspect.currentframe().f_back
            (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
            print(f"Unexpected {err=}, {type(err)=}", line_number, lines)
            pass

        #  begin v4l2_sink with scrcpy as Popen thread with threaded output readers
        #
        print("v4l2 device currently free, starting scrcpy")
    #    cmd = "scrcpy -s %s --lock-video-orientation=3 --v4l2-sink=/dev/video0 --stay-awake --power-off-on-close --no-display --show-touches" % serial
#        cmd = "scrcpy -s %s --lock-video-orientation=3 --v4l2-sink=/dev/video0 --stay-awake --power-off-on-close --no-display --show-touches" % self.serial
        # TODO: get laptop's screensize

        wR = 0
        hR = 0
        s = self.actions.serial()
        m = self.actions.mOption()
#  lock orientation in landscape mode or v4l2loopback bugs out, would need to re-open scrcpy then reopen /dev/videoN
        cmd = "scrcpy -s %s -m%s --v4l2-sink=/dev/video%d --show-touches --no-playback --stay-awake --capture-orientation=@270" % (self.actions.serial(), self.actions.mOption(), devN)
        print(cmd)
        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        pq = queue.Queue()
        t1 = threading.Thread(target=stdout_reader, args=(proc, pq))
        t1.start()
        t2 = threading.Thread(target=stderr_reader, args=(proc, pq))
        t2.start()

        wait = isinstance(proc, subprocess.Popen)
        waited = 0
        self.timeout = 5
        while wait and waited < self.timeout:
            try:
                line = pq.get(block=False)

                if (re.match(r"(WARN|INFO|ERROR|\[server).*", line)):
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

    def checkCurrentFocus(self):
        cmd = "adb -s %s shell dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp|mHoldScreenWindow'"
        print(cmd)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        return result.stdout

    #  for future reference,
    #   setting up open camera
    #  video mode, hide all gui elements
    #  set resolution to that of phone screen size
    #  enable digital video stabilization
    #  preference_auto_stabilise = true (auto level in gui)
    #    images mat be smaller resolution due to rotating and cropping
    #
    def isRunningOrStart(self,package="net.sourceforge.opencamera"):
        focus=self.checkCurrentFocus()
        m = re.match(r".*?%s.*" % package,focus)
        if (m):
            return
        else:
            cmd = "adb -s %s shell monkey -p %s -c android.intent.category.LAUNCHER 1" % (SERIAL,package)
            print(cmd)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)



    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)

        if self.replaying == False and self.battleStart != None and self.battleEnd == None:
            if self.replayLast == None or ((datetime.now() - self.replayLast) > timedelta(seconds=self.replayFreq)):
                self.replayFrames.append(qt_img)
                self.replayLast = datetime.now()
                if (len(self.replayFrames) > self.replayFramesMax):
                    self.replayFrames.pop(0)

        if self.replaying == False:
            self.image_label.setPixmap(qt_img)
    
    def snapshot(self):
        self.takeSnapshot = True

    def replay_clicked(self):
        self.replaying = True
        #self.replayFrame = 0
        #self.replay = True
        count = 0
        for frame in self.replayFrames:
            count += 1
            self.image_label.setPixmap(frame)
            QtTest.QTest.qWait(int(1000 * self.replayFreq * 3))
        self.replaying = False


    def convert_cv_qt(self, cv_img):
        if (self.takeSnapshot == True):
            self.takeSnapshot = False
            cv.imwrite("snapshot.png", cv_img)
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    signal.signal(signal.SIGINT, handler)
    app = QApplication(sys.argv)
    main = ImageControl(actions=MonkeyCity())

#its dirty and cant get messy but the has-a object and check persistent data store of options now
    main.actions.parent = main
    main.run()
    cb = main.checkBoxes.checkedButton()
    if (cb):
        print("imageOptionClicked: ",cb.text())
        main.check_option_signal.emit(cb.text())
    main.show()
    #  cleanup after Popen
    frame = inspect.currentframe()
    handler(signal.SIGINT, frame)

