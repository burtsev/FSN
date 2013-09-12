# -*- coding: utf-8 -*-
"""This code is intendent for the tesing of dynamics of the elementary FS

Created on Mon Sep 02 13:49:23 2013

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
input1.ID = 0
input2.ID = 1
FS1.ID = 2
FSNet = {0: input1,
         1: input2,
         2: FS1}
input1.activity = input2.activity = FS1.activity = FS1.mismatch = 0
plotData = [] # storage for results of FS simulation
for i in range(10):
    FS1.update(FSNet)
    #input1.activity += 0.2
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
input1.activity = 1
for i in range(10):
    FS1.update(FSNet)
    #input2.activity += 0.01
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    input1.activity = 0
input2.activity = 1
for i in range(10):
    FS1.update(FSNet)
    #input1.activity += 0.01
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    input2.activity = 0 
input1.activity = 1   
#print FS1.onTime
for i in range(20):
    FS1.update(FSNet)
    print FS1.onTime, ' failed?', FS1.failed    
    print FS1.goalState
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    input1.activity = 0
input1.activity =  1                   
for i in range(10):
    FS1.update(FSNet)   
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])
    input1.activity = 0
input2.activity = 1
for i in range(5):
    FS1.update(FSNet)   
    plotData.append([input1.activity, input2.activity,
                     FS1.activity, FS1.mismatch,
                     FS1.isActive, FS1.failed])  
    input2.activity = 0
pd = zip(*plotData) #transposing array
#in1_plt, in2_plt, fsa_plt, fsm_plt, fsab_plt, fsf_plt = plt.subplots(nrows=6)
in1_plt = plt.bar(range(-1,(len(pd[0])-1)), pd[0], label = 'input 1', color = 'pink')
in2_plt = plt.bar(range(-1,(len(pd[1])-1)), pd[1], label = 'input 2',  color = 'lightblue')
fsa_plt = plt.plot(pd[2], label = 'FS activity', linewidth=3)
fsm_plt = plt.plot(pd[3], label = 'FS goal match', linewidth=2)
fsab_plt = plt.plot(pd[4], label = 'FS activated', linewidth=2)
fsf_plt = plt.plot(pd[5], label = 'FS failed', linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()