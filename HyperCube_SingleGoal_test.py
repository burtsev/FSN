# -*- coding: utf-8 -*-
"""Learning a route from 000...000 to 111...111 on a hypercube with 
a Functional Systems Network

Created on Wed Sep 11 09:01:40 2013

@author: Burtsev
"""
import FSNpy as FSN
import operator

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
    winFS = max(outFSActivity.iteritems(), key=operator.itemgetter(1))[0]
    state[winFS % dim] = int(winFS/dim)-2    
    return state
    

dim = 3 # dimension of a hypercube
FSNet = FSN.FSNetwork()

for i in range(2*dim+2*dim+1): # initial FSs: inputs + effectors + goal
    FSNet.add(FSN.AtomFS())

for i in range(dim):
    # outputs "0->1"
    FSNet.addActionLinks([[i, i+3*dim, 1.]])
    FSNet.addPredictionLinks([[i+dim, i+3*dim, 1.]])
    # outputs "1->0"
    FSNet.addActionLinks([[i+dim, i+2*dim, 1.]])
    FSNet.addPredictionLinks([[i, i+2*dim, 1.]])
    # goal FS
    FSNet.addActionLinks([[i, 4*dim, 1.]])
    FSNet.addPredictionLinks([[i+dim, 4*dim, 1.]])    

start = [0 for i in range(dim)]
goal = [1 for i in range(dim)] 

currState = start
#FSNet.activateFS(dict(zip(range(2*dim),inputMap(currState)))) 
for t in range(10) :
    output = FSNet.update(inputMap(currState))
    oldState = currState
    currState = outputMap(currState, dict((x, output[x]) for x in range(2*dim, 4*dim)))
    FSNet.drawNet()
    print output
    print currState
#for fs in FSNet.net.keys():
#    print FSNet.net[fs].ID