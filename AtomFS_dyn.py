# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:12:25 2013

@author: Brutsev
"""

import FSNpy as FSN
import matplotlib.pyplot as plt

# create net
FSNet = FSN.FSNetwork()
# inputs
inputs = []
hidden = []
nIn = 2
nHid = 2
for fs in range(nIn):
    inputs.append(FSN.AtomFS())
    FSNet.add(inputs[fs])
# FS which is tested
for fs in range(nHid):
    hidden.append(FSN.AtomFS())
    FSNet.add(hidden[fs])
# add links
links = []
for fs in range(nIn):
    for hid in range(nHid):
        links.append([fs,(nIn+hid),1.])
FSNet.addActionLinks(links)
links = []
for hid in range(nHid):
    for hid2 in range(nHid):
        if (hid != hid2):
            links.append([(nIn+hid),(nIn+hid2),1.])            
FSNet.addInhibitionLinks(links)
links = []
for fs in range(nIn):
    for hid in range(nHid):
        links.append([fs,(nIn+hid),0.])
FSNet.addPredictionLinks(links)

plotData = [] # storage for results of FS simulation
plots = []
res = 10
period = 200
for hid in range(nHid):
    hidden[hid].tau = period+1
# set inputs
inAct = {}
for fs in range(nIn):
    inAct[fs] = 1.

for initFS1 in range(res):
    print ((initFS1+1.)/res)
    for hid in range(nHid):
        hidden[hid].oldActivity = (initFS1+1.)/res
        hidden[hid].activity = (initFS1+1.)/res
        hidden[hid].onTime = 0
        hidden[hid].isActive = False
    #print initFS1, initFS2, FS1.activity, FS2.activity
    plotData = []     
    FSNet.update(inAct.copy())
    for t in range(period):           
        if t > 4:
            FSNet.update(inAct.copy())
            #print inAct
        
        plotData.append([hidden[hid2].activity for hid2 in range(nHid)])
#        pd = zip(*plotData)
#        plt.plot(pd[0],pd[1])
    plots.extend(plotData)
            
#print plotData 
#pd = zip(*plotData) #transposing array
#plt.plot(plots)#plotData[1])
FSNet.drawNet()
plt.figure()
plt.plot(plots)
plt.show()