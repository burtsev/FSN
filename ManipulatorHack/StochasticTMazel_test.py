# -*- coding: utf-8 -*-
"""Learning a route from 000...000 to 111...111 in stochastic T-maze with
a Functional Systems Network

Created on Fri Mar 27 11:01:40 2013

@author: Burtsev
"""
import random
import FSNpy as FSN
import AtomFS as fs
import matplotlib.pyplot as plt
import scipy as np
import VizFSN as viz
# import operator

"""inputs for binary string associated with a hypercube nodes
                       [0] 000... [dim-1]
                     [dim] 111... [2*dim-1]
    outputs "1->0" [2*dim] ...... [3*dim-1]
            "0->1" [3*dim] ...... [4*dim-1]
    interlayer FS
            "1->0" [4*dim] ...... [5*dim-1]
            "0->1" [5*dim] ...... [6*dim-1]
"""


def probSel(weights):
    rnd = random.random() * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def st2Ind(st):
    """Converts list of bits to the decimal index"""
    return int(''.join(map(str, st)), 2)


def inputMap(state):
    """converts binary description of the current state into activations
       of the input layer"""
    inputs = []
    for bit in range(len(state)):
        if state[bit] == 0:
            inputs.append(1)
        else:
            inputs.append(0)

    return inputs  # dict(zip(range(dim), inputs))


def outputMap(state, outFS, trans):  # TODO
    """calculates change of the environmental state caused by activities of FSs """

    winFS = probSel([out.activity for out in outFS.values()])
    newState = state[:]
    newState[winFS % dim] = int(winFS / dim) - 2
    if trans[st2Ind(state)][st2Ind(newState)]:
        state = newState[:]
    print 'act:', winFS, ' ->', state
    return state


def setTransitions(dimension):
    """ setting transitions in the state space """
    space_size = np.power(2, dimension)
    transition = np.ndarray(shape=(space_size, space_size), dtype=bool)
    transition.fill(False)
    state1 = [0 for i in range(dimension)]
    state2 = state1[:]
    for i in range(dimension):
        state2[i] = 1
        transition[st2Ind(state1)][st2Ind(state2)] = True  # forward transition
        transition[st2Ind(state2)][st2Ind(state1)] = True  # backward transition
        state1 = state2[:]
    for i in range(space_size):
        for j in range(space_size):
            if transition[i][j]:
                print str(bin(i))[2:], '->', str(bin(j))[2:]
    return transition


dim = 2  # a dimension of a hypercube
drawFSNet = False  # draw FSNet for every FS addition
stateTr = setTransitions(dim)
start = [0 for i in range(dim)]  # start state
goal = [1 for i in range(dim)]  # goal state

FSNet = FSN.FSNetwork()
FSNet.initCtrlNet(dim, 2*dim, 1)
FSNet.addActionLinks([[l, FSNet.goalFS.keys()[0], start[l]] for l in range(dim)])
FSNet.addPredictionLinks([[l, FSNet.goalFS.keys()[0], goal[l]] for l in range(dim)])

# -------------------------
convergenceLoops = 1  # a number of FS network updates per world's state update
period = 50  # a period of simulation
# ------------------------
# FSNet.drawNet()
currState = start[:]
data = []
goalFS = []
goalsReached = 0
goalsDyn = []
NFSDyn = []

# FSNet.activateFS(dict(zip(range(2*dim),inputMap(currState))))
for t in range(period):

    FSNet.update(t, inputMap(currState))
    oldState = currState[:]
    if (t % convergenceLoops) == 0:
        currState = outputMap(currState, FSNet.outFS, stateTr)

    print 't', t
    print 'goals:', goalsReached
    print 'activations:', {k: round(v, 2) for k, v in FSNet.activation.iteritems()}
    print 'mismatches:', {k: round(v, 2) for k, v in FSNet.mismatch.iteritems()}
    #    tau = {}
    #    for fs in FSNet.net.keys():
    #        tau[fs]=FSNet.net[fs].onTime
    #    print 'on time', tau
    #    isact = {}
    #    for fs in FSNet.net.keys():
    #        isact[fs]=FSNet.net[fs].isActive
    #    print 'active', isact
    print 'active:', FSNet.activatedFS
    print 'failed:', FSNet.failedFS
    print 'learning:', FSNet.learningFS
    print 'mem trace:', FSNet.memoryTrace.keys()
    print 'matched:', FSNet.matchedFS
    #    print 'net:', FSNet.net.keys()
    print currState
    print '-'
    # data += [[output[6],output[7],output[8],output[9],output[10],output[11]]]
    fs_dyn = []
    for j in sorted(FSNet.activation.iterkeys()):
        if j > (dim - 1):
            fs_dyn += [FSNet.activation[j]]
    data += [fs_dyn]
    #goalFS.append([FSNet.activation[12],FSNet.mismatch[12],FSNet.net[12].isActive,FSNet.net[12].failed])
    goalsDyn.append(goalsReached)
    NFSDyn.append(len(FSNet.net.keys()))
    if (currState == goal):
        if (oldState != goal):
            goalsReached += 1
            # break
        #        if (len(FSNet.failedFS)==0 and (np.rand() < 0.2)):# and (len(FSNet.activatedFS)==dim):
        if (np.rand() < 0.2):
            currState = start[:]
            FSNet.resetActivity()
            print currState, start
    if len(FSNet.matchedFS) > 0 and drawFSNet:
        plt.figure(num=('t:' + str(t)))
        plt.subplots_adjust(left=0.02, right=0.98, top=1., bottom=0.0)
        viz.drawNet(FSNet.net)

plt.figure()
plt.subplot(3, 1, 1)
plt.pcolor(np.asarray(zip(*data)))
plt.title('out FS dynamics')
#plt.figure()
plt.subplot(3, 1, 2)
plt.plot(goalsDyn)
plt.title('goals reached')
plt.subplot(3, 1, 3)
plt.plot(NFSDyn)
plt.title('number of FS')
#gFSdata = zip(*goalFS)
#plt.plot(gFSdata[0], color='red')
#plt.plot(gFSdata[1], color='blue')
#plt.bar(range(-1,(len(gFSdata[2])-1)),gFSdata[2],width=0.5,color='pink')
#plt.bar(range(-1,(len(gFSdata[3])-1)),gFSdata[3],width=0.8,color='lightBlue')

plt.figure()
plt.subplots_adjust(left=0.02, right=0.98, top=1., bottom=0.0)
viz.drawNet(FSNet.net)

plt.show()