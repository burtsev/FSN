# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 17:05:35 2013

@author: Burtsev
"""

import FSNpy as FSN
import matplotlib.pyplot as plt

# weights to FS's inputs
pw = {0: 1., 1: 0.}
gw = {0: 0., 1: 1.}
# inputs
input1 = FSN.AtomFS()
input2 = FSN.AtomFS()
# FS which is tested
FS1 = FSN.AtomFS()
FS1.set_params(pw, gw, 15, 0.5, 0)
FSNet = FSN.FSNetwork()
FSNet.add(input1)
FSNet.add(input2)
FSNet.add(FS1)
FS1.activity = FS1.mismatch = 0
plotData = [] # storage for results of FS simulation
worldState = {0:0,1:0}
for i in range(10):
    FSNet.update(worldState)
    #input1.activity += 0.2
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
#input1.activity = 1
worldState = {0:1,1:0}                    
for i in range(10):
    FSNet.update(worldState)
    #input2.activity += 0.01
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    worldState = {0:0,1:0}
worldState = {0:0,1:1}
for i in range(10):
    FSNet.update(worldState)
    #input1.activity += 0.01
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    worldState = {0:0,1:0}
worldState = {0:1,1:0}  
#print FS1.onTime
for i in range(20):
    FSNet.update(worldState)
    #print FS1.onTime, ' failed?', FS1.failed
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    worldState = {0:0,1:0}
worldState = {0:1,1:0}                  
for i in range(10):
    FSNet.update(worldState)   
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    worldState = {0:0,1:0}
worldState = {0:0,1:1}
for i in range(5):
    FSNet.update(worldState)   
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])  
    worldState = {0:0,1:0}
pd = zip(*plotData) #transposing array
#fig, ((in1_plt, in2_plt), (fsa_plt, fsm_plt), (fsab_plt, fsaf_plt)) = plt.subplots(nrows=3, sharex=True)
in1_plt = plt.bar(range(-1,(len(pd[0])-1)), pd[0], label = 'input 1', color = 'pink')
in2_plt = plt.bar(range(-1,(len(pd[1])-1)), pd[1], label = 'input 2',  color = 'lightblue')
fsa_plt = plt.plot(pd[2], label = 'FS activity', linewidth=3)
fsm_plt = plt.plot(pd[3], label = 'FS goal match', linewidth=2)
fsab_plt = plt.plot(pd[4], label = 'FS activated', linewidth=2)
fsaf_plt = plt.plot(pd[5], label = 'FS failed', linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()