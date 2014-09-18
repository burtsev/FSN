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
# FS which is tested
FS1 = FSN.AtomFS()
FS1.set_params(pw, gw, 10, 0.5, 0)
print 'pw:', FS1.problemWeights,'gw:',FS1.goalWeights
FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
plotData = [] # storage for results of FS simulation
for i in range(5):
    FS1.update()
    print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
    #input1.oldActivity += 0.2
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
#FS1.tau = 5

FS1.problemState = FS1.goalState = {0: 1., 1: 0.}
FS1.update()
print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
FS1.update()
print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
FS1.problemState = FS1.goalState = {0: 0., 1: 1.}
FS1.update()
print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
FS1.problemState = FS1.goalState = {0: 1., 1: 0.}
FS1.update()
print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
for i in range(5):
    FS1.update()
    print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
    FS1.problemState = FS1.goalState = {0: 0., 1: 0.}

FS1.problemState = FS1.goalState = {0: 0., 1: 1.}
FS1.update()
print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
FS1.tau = 7
for i in range(5):
    FS1.update()
    print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
    #input2.oldActivity += 0.01
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])

#FS1.problemState = FS1.goalState = {0: 0., 1: 1.}
#for i in range(5):
#    FS1.update()
#    print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
#    #input1.oldActivity += 0.01
#    plotData.append([FS1.problemState[0], FS1.problemState[1],
#                     FS1.activity,( FS1.mismatch-0.01),
#                     FS1.isActive, (FS1.failed+0.01)])
#    FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
FS1.problemState = FS1.goalState = {0: 1., 1: 0.}
#print FS1.onTime
for i in range(10):
    FS1.update()
    print FS1.problemState,'act:',FS1.activity,'wIn:',FS1.calcProblemActivation(),'mismatch:',FS1.mismatch
    print FS1.onTime, ' failed?', FS1.failed
    print FS1.goalState
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
    #input1.oldActivity = 0

for i in range(3):
    FS1.update()
    print FS1.onTime, ' failed?', FS1.failed
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
    FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
FS1.problemState = FS1.goalState = {0: 0., 1: 1.}
for i in range(3):
    FS1.update()
    print FS1.onTime, ' failed?', FS1.failed
    plotData.append([FS1.problemState[0], FS1.problemState[1],
                     FS1.activity,( FS1.mismatch-0.01),
                     FS1.isActive, (FS1.failed+0.01)])
    FS1.problemState = FS1.goalState = {0: 0., 1: 0.}
pd = zip(*plotData) #transposing array
plt.axes([0.05,0.05,0.7,0.9])
#in1_plt, in2_plt, fsa_plt, fsm_plt, fsab_plt, fsf_plt = plt.subplots(nrows=6)
in1_plt = plt.bar(range(-1,(len(pd[0])-1)), pd[0], label = 'action input', color = 'pink')
in2_plt = plt.bar(range(-1,(len(pd[1])-1)), pd[1], label = 'goal input',  color = 'lightgreen')
fsab_plt = plt.plot(pd[4], label = 'FS activated', linewidth=6, color = 'yellow')
fsa_plt = plt.plot(pd[2], label = 'FS activity', linewidth=2, color = 'red')
fsm_plt = plt.plot(pd[3], label = 'FS goal match', linewidth=2, color = 'green')
fsf_plt = plt.plot(pd[5], label = 'FS failed', linewidth=2,  color = 'blue')

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
plt.show()