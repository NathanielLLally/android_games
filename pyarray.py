#!/usr/bin/python

import sys
import os
import time
from datetime import datetime 
from datetime import timedelta 

mylist = [1,2,3,4,5]
l = [ 1, [2,3], [1], 5]

for el in l:
    print(el)
    if hasattr(el, "__len__"):
        print('is array')
    else:
        print('is scalar')

then = datetime.now()

time.sleep(1)
print("%s" % str(datetime.now() - then).split('.')[0])
