# -*- coding: utf-8 -*-
"""This library implements classes and functions
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

import scipy as np
from copy import deepcopy

""" Some general functions.
"""
def sigmoid(x, k, x0): # sigmoid activation function
    return 1/(1+np.exp(-k*(x-x0)))

class AtomFS:
    """Class for the elementary functional system (FS).

        This class implements basic FS functionality:
        1) activation in the problem state;
        2) deactivation in the goal state;
        3) tracking time of transition from the problem to the goal
    """
    # FS attributes
    # - metadata
    ID = int # FS identificator
    parentID = int # identificator of parent FS
    # - structural parameters
    activationWeights = {} # wieghts for the problemState input
    predictionWeights = {} # weights for the goal input
    lateralWeights = {} # weights for the lateral inhibition
    controlWeights = {}    # weights for the top-down control
    # - dynamical parameters
    tau = float # expected time for transition from the problem to the goal state
    threshold = float # for the activation
    noise = float # random value to be added to the FS activation
    k = float
    x0 = float
    pr_threshold = float # for the prediction
    pr_k = float
    pr_x0 = float
    rateOfWeightLearning = 0.1
    # - state variables
    problemState = [] # input for the features of the problem to be solved by FS
    goalState = [] # input for the features of the required solution
    inhibitionState = [] # input for the lateral inhibition
    controlState = [] # input for the top-down control
    activity = float # current value of FS activity
    activityOld = float # value of FS activiy on the previous time step
    wasActive = [] # history of FS activity
    onTime = int # the period of current FS's activity
    mismatch = float # current value of mismatch between goal and current state
    isActive = bool # presence of FS activity
    isLearning = bool # learning state
    failed = bool # FS was unable to achive goal state
    isInput = bool # is true if value is set externally
    isOutput = bool # is true if the value is not predicted

    def __init__(self):
        """"Create and initialize FS."""
        self.ID = 0
        self.activity = 0.
        self.oldActivity = 0.
        self.mismatch = 0.
        self.isActive = False
        self.isInput = False
        self.isOutput = False
        self.wasActive = []
        for i in range(2): # depth of the activation memory is 2
            self.wasActive.append(False)
        self.failed = False
        self.isLearning = False
        self.onTime = 0
        self.tau = 50
        self.k = 30      # k and x0 are choosen to have output 0.5 for normalized weighted
        self.x0 = 0.5    # input of 0.5 and high activation for input = 1
        self.threshold = 0.95
        self.pr_k = self.k
        self.pr_x0 = self.x0
        self.pr_threshold = self.threshold
        self.noise = 0.01
        self.activationWeights = {}
        self.predictionWeights = {}
        self.lateralWeights = {}
        self.controlWeights = {}

    def set_params(self, pw, gw, t, th, n, cw={0:0}):
        """"set parameters of FS."""
        self.activationWeights = pw
        self.predictionWeights = gw
        self.controlWeights = cw
        self.threshold = th
        self.tau = t
        self.noise = n

    def calcProblemActivation(self):
        """Returns a value of activation for the weighted input."""
        # calculation of sum of weighted inputs (positive weights)
        wInSum = np.array([[self.problemState[fs],self.activationWeights[fs]]
                    for fs in self.activationWeights.iterkeys()
                    if fs in self.problemState]
                        ).prod(1).sum()
        if (wInSum > 0): # weighted sum is normalised
             wInSum = wInSum/sum(w for w in self.activationWeights.itervalues())
        # adding noise
        wInSum += 2*(0.5-np.rand())*self.noise

    def activate(self):
        """Returns a value of activation for the weighted input."""
        # calculation of sum of weighted inputs (positive weights)
        wInSum = np.array([[self.problemState[fs],self.activationWeights[fs]]
                    for fs in self.activationWeights.iterkeys()
                    if fs in self.problemState]
                        ).prod(1).sum()
        if (wInSum > 0): # weighted sum is normalised
             wInSum = wInSum/sum(w for w in self.activationWeights.itervalues())
        # positive feedback from own activity to stay active
        # when the activated (problem) input is gone
        wInSum = wInSum*0.36+self.oldActivity
        # adding noise
        wInSum += 2*(0.5-np.rand())*self.noise
        # calculation of sum of weighted inputs (negative weights)
        # TODO : незабыть, что когда будет введена возможность латерального
        #        возбуждения надо будет убрать знак минус перед взвешенной
        #        суммой при ее добавлении к аргументу.
        sumInhW = sum(w for w in self.lateralWeights.itervalues())
        if (sumInhW>0):
            wInhIn = [[self.problemState[fs],self.lateralWeights[fs]]
                    for fs in self.lateralWeights.iterkeys()
                    if fs in self.problemState]
            if (len(wInhIn)>0): # weighted sum is normalised
                wInSum -= np.array(wInhIn).prod(1).sum()/sumInhW
        return sigmoid(wInSum, self.k, self.x0)

проводить расчет для каждого из входов отдельно, а затем собирать в ФС элементе

    def predict(self, weightedIn, k, x0, sigma,
                   weightedNegIn = [[0,0]], norm = True):
        """Returns a value of prediction for the weighted input.

        weightedIn is a list of pairs [input_activation, weight]"""
        wInSum = np.array(weightedIn).prod(1).sum()
        if (norm and wInSum>0):# the argument is normalised
            wInSum = wInSum/np.array(weightedIn)[:,1].sum()
        return sigmoid(wInSum, k, x0)

    def weightsUpdate(self, fsnet = {}):
        """Updates current weights of FS to exclude unimportant connections"""
        for fs in self.activationWeights.keys():
            if fsnet[fs].isActive == False:
                if self.activationWeights[fs] > self.rateOfWeightLearning:
                    self.activationWeights[fs] -= self.rateOfWeightLearning
                else: self.activationWeights[fs] = 0

    def updateActivation(self):
        """Updates activation of FS"""
        self.activity = self.activate()
        # FS is activated if it's problem state is present
        if (self.activity >= self.threshold):
            self.isActive = True
            #self.activity = 1.
        else:
            self.isActive = False
        return self.activity

    def updatePrediction(self, fsnet = {}):
        """Updates prediction of FS"""
        self.goalState = [] # updating the goal input
        for offFS in self.predictionWeights.keys():
        # calculates weighted inputs for the FS deactivation
            if not fsnet[offFS].isLearning:
                self.goalState += [[fsnet[offFS].oldActivity,
                                self.predictionWeights[offFS]]]

        # goal recognition should be more definite than activation, hence, k=10 and noise=0
        if len(self.predictionWeights.keys())>0:
            self.mismatch = self.predict(self.goalState,self.pr_k,
                                            self.pr_x0, 0)
        if (self.mismatch >= self.pr_threshold): # the goal state has been obtained
            self.failed = False
            self.isActive = False # FS is not needed any more
            self.mismatch = 0
            self.onTime = 0
            self.activity = 0
#            if self.isLearning:
#                self.tau = self.onTime
#                self.isLearning = False
        else: # the goal is not achived in expected timeframe
            if ((self.onTime == self.tau)): #and not self.isLearning):
                self.failed = True
                #self.isActive = False
                self.activity = 0
        return self.mismatch

    def update(self, fsnet = {}): # net is a dictionary {FSID: AtomFS}
        """Updates current state of FS."""
        self.wasActive.pop(0)
        self.wasActive.append(self.isActive)
        self.updateActivation()
        if self.isActive and self.wasActive[-2] == False:
            self.onTime = 0 # internal time is reset
        if ((self.wasActive[-2] or self.isActive or self.failed)
            and not self.isLearning and not self.isOutput):
            self.updatePrediction(fsnet)
        # update FS lifetime
        if (self.isActive or self.failed or self.isLearning or not self.isOutput):
            self.onTime += 1
        return self.activity, self.mismatch

    def resetActivity(self):
        """Resets FS activity"""
        self.failed = False
        self.isActive = False
        self.wasActive = []
        for i in range(2): # depth of the activation memory is 2
            self.wasActive.append(False)
        self.mismatch = 0
        self.onTime = 0
        self.activity = 0
        self.activityOld = 0
# end of AtomFS class

class FSNetwork:
    """Implements a network of functional systems"""
    net = {} # net is a dictionary {FSID: AtomFS}
    memoryTrace = {} # is a dictionary {FSID: AtomFS}
    idCounter = int # counter for FS id's
    failedFS = [] # a list of FSs that failed at the current time
    matchedFS = [] # a list of FSs that were failed and now have prediction satisfied
    activatedFS = []  # a list of FSs that activated at the current time
    activation = {} # dict with {fsID, activation}
    mismatch = {}
    learningFS = []

    def __init__(self):
        self.net = {}
        self.idCounter = 0
        self.memoryDepth = 1 # how long a FS is retained in the memory trace
        self.failedFS = [] # list of FSs that failed at the current time
        self.activatedFS = [] # a list of FSs that activated at the current time
        self.matchedFS = [] # a list of FSs that were failed and now have prediction satisfied

    def add(self, fs):
        """adds FS to the network"""
        fs.ID = self.idCounter
        self.net[fs.ID] = fs
        self.idCounter +=1

    def duplicate(self, ID, outLnkDup = False): # outLnkDup is optional parameter
        """duplicates FS and returns offspring"""
        offspring = deepcopy(self.net[ID])
        offspring.parentID = ID
        self.add(offspring)
        if outLnkDup:
            for fs in self.net.keys():
                if ID in self.net[fs].activationWeights.keys():
                  self.net[fs].activationWeights[offspring.ID] = \
                  self.net[fs].activationWeights[ID]
                if ID in self.net[fs].lateralWeights.keys():
                    self.net[fs].lateralWeights[offspring.ID] = \
                    self.net[fs].lateralWeights[ID]
#                if ID in self.net[fs].predictionWeights.keys():
#                    self.net[fs].predictionWeights[offspring.ID] = \
#                    self.net[fs].predictionWeights[ID]
        return offspring

    def removeFS(self, ID):
        """removes FS from the network with cleaning up all outgiong links"""
        del self.net[ID]
        for fs in self.net.keys():
            if ID in self.net[fs].activationWeights.keys():
                del self.net[fs].activationWeights[ID]
            if ID in self.net[fs].lateralWeights.keys():
                del self.net[fs].lateralWeights[ID]
            if ID in self.net[fs].predictionWeights.keys():
                del self.net[fs].predictionWeights[ID]

    def createFS(self, problemFS):
        newFS = self.duplicate(problemFS.ID, outLnkDup = True) # duplication
        # update the problem state of the new FS with the current state
        newFS.activationWeights.clear()
        newFS.predictionWeights.clear()
        newFS.lateralWeights.clear()
        newFS.isActive = False
        newFS.activity = 1.
        newFS.oldActivity = 1.
        newFS.mismatch = 0.
        newFS.failed = False
        newFS.onTime = 0
        newFS.isLearning = True
        # FS is relevant to the problem of the parent FS
        newFS.activationWeights[problemFS.ID] = 1.
        newFS.lateralWeights[problemFS.ID] = 1.
        # problemFS.predictionWeights[newFS.ID] = 1.
        # alernative behavior inhibits failed behavior
        problemFS.lateralWeights[newFS.ID] = 1.

        for fs in self.net.keys():

            # new FS should be activated in the state
            # that created the problem for the parent FS
            if self.net[fs].isInput: # TODO проверить на необходимость учета только входов от среды
               if self.net[fs].wasActive[-2]:
                   newFS.activationWeights[fs] = 1.
#            else:
#                if self.net[fs].wasActive[-2]:
                    #newFS.activationWeights[fs] = 1.

            # new FS should activate other FSs (i.e. 'motor' FS)
            # that contribute to the memorising state transition
            if (self.net[fs].wasActive[-2] and
                #problemFS.ID in self.matchedFS and
                not (self.net[fs].isLearning or self.net[fs].isInput
                    or self.net[fs].failed or fs==newFS.parentID)):
                    #or self.net[fs].wasActive[-1])):
                self.net[fs].activationWeights[newFS.ID] = 1.
                #newFS.predictionWeights[fs] = 1. # alt to the next line
                #newFS.lateralWeights[self.net[fs].ID] = 1.

            # results of actions (i.e. neurons activated after actions)
            # should be predicted by new FS
            if (self.net[fs].isActive and self.net[fs].onTime==1
                and self.net[fs].isInput):
                newFS.predictionWeights[fs] = 1.

        #newFS.update(self)
        return newFS

    def update(self, inputStates = {}):
        """updates the network given values of activations for input elements"""
        self.activation = {} # dict with {fsID, activation}
        self.mismatch = {}

        # activate elements (FSs) correspondent to inputs
        self.activateFS(inputStates)
        # update activations and predictions of hidden and effector FSs
#        for cycle in range(3): # convergence loop
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
#            self.problemState = {} # updating the problem input
            self.net[fs].problemState = {k: self.net[k].oldActivity
                for k in self.net[fs].activationWeights.iterkeys()
                    if not self.net[k].isLearning}
            self.net[fs].problemState.update({k: self.net[k].oldActivity
                for k in self.net[fs].lateralWeights.iterkeys()
                    if self.net[k].isActive and not self.net[k].isLearning})
            self.net[fs].problemState.update({k: self.net[k].oldActivity
                for k in self.net[fs].predictionWeights.iterkeys()
                    if not self.net[k].isLearning})
            self.net[fs].problemState.update({k: self.net[k].oldActivity
                for k in self.net[fs].controlWeights.iterkeys()
                    if not self.net[k].isLearning})

            self.activation[fs], self.mismatch[fs] = self.net[fs].update(self.net)
        # update history of activity
        self.logActivity(inputStates)

        # learning

        # prune ineffective connections
#        for fs in (set(self.net.keys()) - set(inputStates.keys())):
#            if self.net[fs].isActive and not self.net[fs].isOutput:
#                self.net[fs].weightsUpdate(self.net)

# TODO :
        """    implement addition of weights from active FSs
            to currently learning FSs """

        # remove tentative FSs older than memoryDepth
        for fs in self.memoryTrace.keys():
                if self.net[fs].onTime > (self.memoryDepth-1):
                    del self.memoryTrace[fs]
                    self.removeFS(fs)

# TODO:        # create alternatives for the failed FSs
        for i in range(len(self.failedFS)):
            #if self.net[self.failedFS[i]].isActive:
            print 'creating 2nd fs for failedFS:', self.failedFS[i]
            newFS = self.createFS(self.net[self.failedFS[i]])
            self.memoryTrace[newFS.ID] = newFS
            print 'new fs ID:', newFS.ID

        # create alternatives for the matched FSs
        for i in range(len(self.matchedFS)):
           # if self.net[self.matchedFS[i]].wasActive[-1]:
            print 'creating 2nd fs for matchedFS:', self.matchedFS[i]
            newFS = self.createFS(self.net[self.matchedFS[i]])
            self.memoryTrace[newFS.ID] = newFS
            print 'new fs ID:', newFS.ID

        # connect to the known trajectory
#        for i in range(len(self.activatedFS)):
#            if (self.net[self.failedFS[i]].wasActive[-1]
#                and self.net[self.failedFS[i]].onTime>=self.net[self.failedFS[i]].tau):
#                for fs in self.memoryTrace.keys():
#                    if self.net[fs].parentID == self.failedFS[i]:
#                        del self.memoryTrace[self.net[fs].ID]
#                        self.net[fs].isLearning = False
#                        self.net[fs].onTime = 0

        # activate new memory trace for the matched FSs
        for i in range(len(self.matchedFS)):
            for fs in self.memoryTrace.keys():
                if self.net[fs].parentID == self.matchedFS[i]:
                    del self.memoryTrace[fs]
                    self.net[fs].isLearning = False
                    self.net[fs].onTime = 0

#        self.memoryTrace = {}
        return self.activation

    def addActionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].activationWeights[links[lnk][0]] = links[lnk][2]

    def addInhibitionLinks(self, links=[]):
        """creates  inhibition links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].lateralWeights[links[lnk][0]] = links[lnk][2]

    def addPredictionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].predictionWeights[links[lnk][0]] = links[lnk][2]

    def activateFS(self, fs_list={}):
        """sets activations for listed FSs"""

        for inFS in fs_list.keys():
            self.net[inFS].isInput = True # TODO: in the case of variable inputs this flag should be resetted
            # remove oldest record from activation history
            self.net[inFS].wasActive.pop(0)
            # add activation from the last time step
            self.net[inFS].wasActive.append(self.net[inFS].isActive)
            # set current value of activity
            #fs_list[inFS] = fs_list[inFS] #*self.net[inFS].threshold
            self.net[inFS].oldActivity = fs_list[inFS]
            self.net[inFS].activity = fs_list[inFS]
            self.activation[inFS] = fs_list[inFS]
            self.net[inFS].isActive = True # TODO: remove ???
            if (self.net[inFS].oldActivity >= self.net[inFS].threshold):
                #self.net[inFS].isActive = True
                self.net[inFS].onTime += 1
            else:
                #self.net[inFS].isActive = False
                self.net[inFS].onTime = 0

    def setOutFS(self, fs_list):
        """marks listed FSs as outputs"""
        for outFS in range(len(fs_list)):
            self.net[fs_list[outFS]].isOutput = True

    def logActivity(self,inputStates):
        self.activatedFS = []
        wasFailed = self.failedFS[:]
        self.failedFS = []
        self.learningFS = []
        for inFS in inputStates.keys():
            if self.net[inFS].isActive: # if FS is active
               self.activatedFS.append(self.net[inFS].ID)
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            self.net[fs].oldActivity = self.activation[fs]
            # if FS has failed to reach predicted state
            if (self.net[fs].failed and not self.net[fs].isLearning):
               self.failedFS.append(self.net[fs].ID) # add FS to the failers list
               if self.net[fs].ID in wasFailed:
                   wasFailed.remove(self.net[fs].ID)
            # if FS is active
            if self.net[fs].isActive:
               self.activatedFS.append(self.net[fs].ID)
            if self.net[fs].isLearning:
                self.learningFS.append(fs)
        self.matchedFS = wasFailed[:]
    # end of logActivity

    def resetActivity(self):
        """resets activity for all FS in the net"""
        for fs in self.net.keys():
            self.net[fs].resetActivity()
        self.failedFS = [] # list of FSs that failed at the current time
        self.activatedFS = [] # a list of FSs that activated at the current time
        self.matchedFS = [] # a list of FSs that were failed and now have prediction satisfied


    def drawNet(self):
        """draws the FS network"""
        import networkx as nx
        import matplotlib.pyplot as plot

        G=nx.MultiDiGraph()
        G.add_nodes_from(self.net.keys())
        actionEdges = []
        actionWeights = []
        inhibitionEdges = []
        lateralWeights = []
        predictionEdges = []
        predictionWeights = []
        net_activity = []
        for fs in self.net.keys():
            net_activity.append(self.net[fs].activity)
            for synapse in self.net[fs].activationWeights.keys():
                G.add_edge(synapse, fs, key=0,
                           weight=self.net[fs].activationWeights[synapse])
                actionEdges.append((synapse, fs))
                actionWeights.append(self.net[fs].activationWeights[synapse])
            for synapse in self.net[fs].predictionWeights.keys():
                G.add_edge(synapse, fs, key=1,
                           weight=self.net[fs].predictionWeights[synapse])
                predictionEdges.append((synapse, fs))
                predictionWeights.append(self.net[fs].predictionWeights[synapse])
            for synapse in self.net[fs].lateralWeights.keys():
                G.add_edge(synapse, fs, key=2,
                           weight=self.net[fs].lateralWeights[synapse])
                inhibitionEdges.append((synapse, fs))
                lateralWeights.append(self.net[fs].lateralWeights[synapse])
        node_layout = nx.circular_layout(G)  #nx.graphviz_layout(G,prog="neato")
        plot.cla()
        nx.draw_networkx_nodes(G, pos=node_layout, node_size=1200,
                               node_color = net_activity, cmap = plot.cm.Reds)
        nx.draw_networkx_labels(G, pos=node_layout)
        ar = plot.axes()
        actArrStyle=dict(arrowstyle='simple',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="red",ec="none", alpha=0.6,
                      connectionstyle="arc3,rad=-0.1",)
        inhibitionArrStyle=dict(arrowstyle='simple',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="blue",ec="none", alpha=0.6,
                      connectionstyle="arc3,rad=-0.8",)
        predArrStyle=dict(arrowstyle='simple',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="green",ec="none", alpha=0.6,
                      connectionstyle="arc3,rad=0.7",)
        for vertex in G.edges(keys=True,data=True): # drawing links
            #print vertex
            if vertex[3]['weight'] > 0:
                coords = [node_layout[vertex[1]][0],
                          node_layout[vertex[1]][1],
                         node_layout[vertex[0]][0],
                         node_layout[vertex[0]][1]]
                if (vertex[2] == 0) :
                    actArrStyle['fc'] = plot.cm.YlOrRd(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = actArrStyle)
                if (vertex[2] == 1) :
                    predArrStyle['fc'] = plot.cm.Greens(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = predArrStyle)
                if (vertex[2] == 2) :
                    predArrStyle['fc'] = plot.cm.Blues(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = inhibitionArrStyle)

        ar.xaxis.set_visible(False)
        ar.yaxis.set_visible(False)
        plot.subplots_adjust(left=0.0, right=1., top=1., bottom=0.0)
        plot.show()
        # todo - handle self-links
