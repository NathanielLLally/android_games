#!/usr/bin/python

# Source - https://stackoverflow.com/a/35071359
# Posted by Dimitris Fasarakis Hilliard, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-24, License - CC BY-SA 3.0

from sysconfig import get_paths
from pprint import pprint

info = get_paths()  # a dictionary of key-paths

# pretty print it for now
pprint(info)
{'data': '/usr/local',
 'include': '/usr/local/include/python2.7',
 'platinclude': '/usr/local/include/python2.7',
 'platlib': '/usr/local/lib/python2.7/dist-packages',
 'platstdlib': '/usr/lib/python2.7',
 'purelib': '/usr/local/lib/python2.7/dist-packages',
 'scripts': '/usr/local/bin',
 'stdlib': '/usr/lib/python2.7'}

