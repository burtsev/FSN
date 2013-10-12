# -*- coding: utf-8 -*-
"""Learning a route from 000...000 to 111...111 on a hypercube with 
a Functional Systems Network

Created on Wed Sep 11 09:01:40 2013

@author: Burtsev
"""
import FSNpy as FSN
import matplotlib.pyplot as plt
# import operator

"""inputs for binary string associated with a hypercube nodes
                       [0] 000... [dim-1]
                     [dim] 111... [2*dim-1]
    outputs "1->0" [2*dim] ...... [3*dim-1]
            "0->1" [3*dim] ...... [4*dim-1]
"""
    
def inputMap(state=[]):
    """converts binary description of the current state into activations
       of the input layer"""
    inputs = []
    for bit in range(len(state)):
        if state[bit] == 0:
            inputs.append(1)
        else:
            inputs.append(0)
    for bit in range(len(state)):
        if state[bit] == 1:
            inputs.append(1)
        else:
            inputs.append(0)
    return dict(zip(range(2*dim),inputs))
    
def outputMap(state = [], outFSActivity = {}):
    """calculates change of the environmental state caused by activities of FSs """
#    winFS = max(outFSActivity.iteritems(), key=operator.itemgetter(1))[0]
    
    for fs in FSNet.net.keys():
        if FSNet.net[fs].isActive and (fs in range(2*dim,4*dim)):
            winFS = fs
            state[winFS % dim] = int(winFS/dim)-2    
    return state    

dim = 3 # dimension of a hypercube
FSNet = FSN.FSNetwork()
for i in range(2*dim+2*dim+1): # initial FSs: inputs + effectors + goal
    FSNet.add(FSN.AtomFS())

for i in range(dim): # create links of the initial network
    # outputs "0->1"
    FSNet.addActionLinks([[i, i+3*dim, 1.]])
    FSNet.addPredictionLinks([[i+dim, i+3*dim, 1.]])
    # outputs "1->0"
    FSNet.addActionLinks([[i+dim, i+2*dim, 1.]])
    FSNet.addPredictionLinks([[i, i+2*dim, 1.]])
    # goal FS
    FSNet.addActionLinks([[i, 4*dim, 1.]])
    FSNet.addActionLinks([[4*dim, i+2*dim, 1.]])
    FSNet.addActionLinks([[4*dim, i+3*dim, 1.]])
    FSNet.addPredictionLinks([[i+dim, 4*dim, 1.]])
    # lateral inhibition
    for j in range (2*dim,4*dim):
        if j != (i+2*dim):
            FSNet.addInhibitionLinks([[i+2*dim, j, 1.]])
        if j != (i+3*dim):            
            FSNet.addInhibitionLinks([[i+3*dim, j, 1.]])      
#for j in range (2*dim,4*dim):
#    # self-referent input
#    FSNet.addActionLinks([[j, j, 1.9]])


start = [0 for i in range(dim)] # start state
goal = [1 for i in range(dim)] # goal state
#FSNet.drawNet()
currState = start
data = []
goalFS = []
#FSNet.activateFS(dict(zip(range(2*dim),inputMap(currState)))) 
for t in range(1500):
    output = FSNet.update(inputMap(currState))
    oldState = currState
    currState = outputMap(currState, dict((x, output[x]) for x in range(2*dim, 4*dim)))
    print t, currState
    print 'activations:', FSNet.activation
    print 'mismatches:', FSNet.mismatch
    print 'active:', FSNet.activatedFS
    print 'failed:', FSNet.failedFS
    print '-'
    data += [[output[6],output[7],output[8],output[9],output[10],output[11]]]
    goalFS.append([FSNet.activation[12],FSNet.mismatch[12],
               FSNet.net[12].isActive,FSNet.net[12].failed])
        
gFSdata = zip(*goalFS)    
plt.plot(data)
plt.figure()
plt.plot(gFSdata[0], color='red')
plt.plot(gFSdata[1], color='blue')
plt.bar(range(-1,(len(gFSdata[2])-1)),gFSdata[2],width=0.5,color='pink')
plt.bar(range(-1,(len(gFSdata[3])-1)),gFSdata[3],width=0.8,color='lightBlue')
plt.figure()
FSNet.drawNet()
plt.show()