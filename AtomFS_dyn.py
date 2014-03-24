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
nIn = 10
nHid = 5
res = 20
period = 100
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
#hidden[0].k = 20
#hidden[0].x0 = 0.122
#hidden[1].k = 15
#hidden[1].x0 = 0.5

for hid in range(nHid):
    hidden[hid].tau = (res+5)*period+1
# set inputs
inAct = {}
for fs in range(nIn):
    inputs[fs].tau = (res+5)*period+1
    inAct[fs] = 1.

for initFS1 in range(res):
    print ((initFS1+1.)/res)
    for hid in range(nHid):
        hidden[hid].oldActivity = 0 #(initFS1+1.)/res
        hidden[hid].activity = 0 #(initFS1+1.)/res
        hidden[hid].onTime = 0
        hidden[hid].isActive = False
    #print initFS1, initFS2, FS1.activity, FS2.activity
    plotData = []
    #for fs in range(nIn):
        #inAct[fs] = (initFS1+0.)/res
    inAct[0] = (initFS1+1.)/res
    FSNet.update(inAct.copy())
    plotData.append([hidden[hid2].activity for hid2 in range(nHid)])
#    for fs in range(nIn):
#        inAct[fs] = 0.
    for t in range(period):
        if t > 4:
            FSNet.update(inAct.copy())
            #print inAct

        plotData.append([hidden[hid2].activity for hid2 in range(nHid)])
    plots.extend(plotData)

for fs in range(nIn):
    inAct[fs] = 0.
plotData = []
FSNet.update(inAct.copy())
plotData.append([hidden[hid2].activity for hid2 in range(nHid)])
for t in range(period):
    if t > 4:
        FSNet.update(inAct.copy())
        #print inAct
    plotData.append([hidden[hid2].activity for hid2 in range(nHid)])

#        pd = zip(*plotData)
#        plt.plot(pd[0],pd[1])
plots.extend(plotData)

# testing inhibition
for fs in range(nIn):
    inAct[fs] = 0.31
    for hid in range(nHid):
        hidden[hid].activationWeights[fs] = 0.
        hidden[hid].inhibitionWeights[fs] = 1.

plotData = []
FSNet.update(inAct.copy())
plotData.append([hidden[hid2].activity for hid2 in range(nHid)])
for t in range(period):
    if t > 4:
        FSNet.update(inAct.copy())
    plotData.append([hidden[hid2].activity for hid2 in range(nHid)])


plots.extend(plotData)

#print plotData
#pd = zip(*plotData) #transposing array
#plt.plot(plots)#plotData[1])
FSNet.drawNet()
plt.figure()
plt.plot(plots)
plt.show()