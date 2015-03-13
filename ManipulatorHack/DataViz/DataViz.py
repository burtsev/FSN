# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 16:37:44 2015

@author: Burtsev
"""

import csv
import matplotlib.pyplot as plt

# reading data
data = [event for event in csv.reader(open('tags_good.txt', 'r'))]
events = [dict([x for x in csv.reader(event, delimiter=':')]) for event in data]

# extracting names of variables
keys = []
for k in events:
    if k['tagName'] not in keys:
        keys.append(k['tagName'])
keyMap = {}
i = 0
for k in keys:
    keyMap[k] = i
    i += 1

# extracting input from the system {write:0} and commands {write:1}
in_events = [[int(t['time']), keyMap[t['tagName']]]
             for t in events if t['write'] == '1' and t['value'] != '0']
out_events = [[int(t['time']), keyMap[t['tagName']]]
              for t in events if t['write'] == '0' and t['value'] != '0']

time = [int(t['time']) for t in events]
time = sorted(time)

dt = []
i = 0
for t in time:
    if i > 0:
        d = t - time[i - 1]
        if d > 0:
            dt.append(d)
    i += 1

in_events = zip(*in_events)
out_events = zip(*out_events)

plt.figure()
plt.plot(in_events[0], in_events[1], '|', color='blue', ms=15, alpha=0.7)
plt.plot(out_events[0], out_events[1], '|', color='red', ms=15, alpha=0.7)

plt.figure()
plt.hist(dt, bins=200)

plt.show()