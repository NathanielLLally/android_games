#!/usr/bin/python

import sys
import os
import numpy as np

class Test():
    def __init__(self, Parray=[], *args, **kwargs):
        print("Test:__init__")
        print("self: ")
        print(self)
        if hasattr(Parray, "__len__"):
            if isinstance(Parray, dict):
                print("its a dict, not array")
            else:
                if len(Parray):
                    self.array = Parray
        self.mode = "child"
        self.time = 5
        print("end Test:__init__")

    def printMode(self):
        print(self.mode)

    def nTime(self, n):
        self.time *= n

    def printTime(self):
        print(self.time)

    def getSet(self,Parray=[]):
        if hasattr(Parray, "__len__"):
            if isinstance(Parray, dict):
                print("its a dict, not array")
            if len(Parray):
                self.array = Parray
        return self.array

class MyTest(Test):
    def __init__(self, *args, **kwargs):
        print("MyTest:__init__")
        super(MyTest, self).__init__(*args, **kwargs)
        self.Raid = ["step1", "step2", "step3"]
        if "mode" in kwargs:
            m = kwargs["mode"]
            print("constructor argument: mode [%s]" % (m))
            if hasattr(self, m):
                self.mode = getattr(self, m)
            else:
                raise Error("no mode defined for argument [%s]" % (m))
        print("end MyTest:__init__")


t = Test([1,2,3,4])

refOrCopy = t.getSet()
el = refOrCopy.pop(0)

print("element 1 [%s]" % el)
print("instance array")
print(t.getSet())

print("apparently pythyon is a pass by reference model")

mt = MyTest([{"one":1,"two":2,"three":3}], mode="Raid")
print(mt.getSet())
mt.printMode()
mt.printTime()
mt.nTime(3)
mt.printTime()


pt = (50, 50)
wh = (10, 100)
print(pt)
#print(list(pt) / [2,2])

box = [10,30,20,100]
print(box)

print(pt, tuple(np.array(pt) + np.array(wh)))
br = np.array(pt) + np.array(wh)
brt = tuple(br)

x, y = br
print(x, y)
x, y = brt
print(x, y)

masked = np.array([
[
np.array([1,47,142]),
np.array([0,123,152]),
np.array([5,71,146]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0])
],[
np.array([0,10,142]),
np.array([7,34,152]),
np.array([0,78,146]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0]),
np.array([0,0,0])
]
])

print(masked)
print(masked.shape)
print(masked.size)
print(masked.itemsize)
print(np.count_nonzero(np.sum(masked,0) > 0))
print(np.count_nonzero(masked > 0, 1))
#print(np.count_nonzero(
#np.count_nonzero(masked > 0, 1)

print(np.sum(masked,0))
print(np.sum(np.sum(masked,0),1))
print(np.count_nonzero(np.sum(np.sum(masked,0),1) > 0))
