# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:37:44 2015

@author: Brutsev
"""

import csv
import string
import matplotlib.pyplot as plt

data = [event for event in csv.reader(open('tags_good.txt','r'))]
events = [[x for x in csv.reader(event, delimiter=':')] for event in data]
tags = [{e[0][1]:e[1][1]} for e in events if int(e[2][1])>0]

keys = []
for k in tags:
    if k.keys() not in keys:
        keys.append(k.keys())

keyMap = {}
i=0
for k in keys:
    keyMap[k[0]]=i
    i+=1

data = [[int(t.values()[0]),keyMap[t.keys()[0]]] for t in tags]
data = zip(*data)

plt.plot(data[0],data[1],'s', ms=15, mew=0, alpha=0.3)
plt.show()