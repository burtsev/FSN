# -*- coding: utf-8 -*-
"""This library implements classes and functions
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

import scipy as np
from copy import deepcopy

""" Some general functions."""
def sigmoid(x, k, x0): # sigmoid activation function
    return 1/(1+np.exp(-k*(x-x0)))

def weightedSum(inputs,weights): # calculation of weighted sum, arguments are lists
    if len(inputs) > 0:
        return np.array([[inputs[i],weights[i]]
                    for i in weights.iterkeys() if i in inputs]).prod(1).sum()
    else:
        return 0

""" Main classes """
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
    problemWeights = {} # wieghts for the problemState input
    goalWeights = {} # weights for the goal input
    lateralWeights = {} # weights for the lateral inhibition
    controlWeights = {}    # weights for the top-down control
    plasticWeights = {} # temporary weights for predictive features of env.
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
    problemState = {} # input for the features of the problem to be solved by FS
    goalState = {} # input for the features of the required solution
    lateralState = {} # input for the lateral inhibition (activation)
    controlState = {} # input for the top-down control
    activity = float # current value of FS activity
    wasActive = [] # history of FS activity
    onTime = int # the period of current FS's activity
    mismatch = float # current value of mismatch between goal and current state
    # - flags
    isActive = bool # presence of FS activity
    isLearning = bool # learning state
    failed = bool # FS was unable to achive goal state
    isInput = bool # is true if value is set externally
    isOutput = bool # is true if the value is not predicted
    exactInputMatch = bool # is true if the FS should be (de)activated only
                        # in the case when the input exactly matches the weights

    def __init__(self):
        """"Create and initialize FS."""
        self.ID = 0
        self.problemWeights = {}
        self.goalWeights = {}
        self.lateralWeights = {}
        self.controlWeights = {}
        self.plasticWeights = {}
        self.tau = 6
        self.threshold = 0.95
        self.noise = 0.1
        self.k = 10      # k and x0 are choosen to have output 0.5 for normalized weighted
        self.x0 = 0.5    # input of 0.5 and high activation for input = 1
        self.pr_threshold = self.threshold
        self.pr_k = self.k
        self.pr_x0 = self.x0
        self.activity = 0.
        self.oldActivity = 0.
        self.wasActive = []
        for i in range(2): # depth of the activation memory is 2
            self.wasActive.append(False)
        self.onTime = 0
        self.mismatch = 0.
        self.isActive = False
        self.isLearning = False
        self.failed = False
        self.isInput = False
        self.isOutput = False
        self.exactInputMatch = True

    def set_params(self, pw, gw, t, th, n, cw={0:0}):
        """"set parameters of FS."""
        self.problemWeights = pw
        self.goalWeights = gw
        self.controlWeights = cw
        self.threshold = th
        self.tau = t
        self.noise = n

    def calcProblemActivation(self):
        """Returns a value of activation for the weighted problem input."""
        # calculation of sum of weighted inputs (positive weights)
        if self.exactInputMatch:
            if (len(self.problemWeights)>0):
                return int(np.array_equal(self.problemState, self.problemWeights))
            return 0
        wInSum = weightedSum(self.problemState, self.problemWeights)
        if (wInSum != 0): # weighted sum is normalised
             wInSum = wInSum/sum(w for w in self.problemWeights.itervalues())
        return wInSum

    def calcGoalMismatch(self):
        """Returns a value of goal state mismatch."""
        if (len(self.goalState)==0):
            return 0
        if self.exactInputMatch:
            self.mismatch = float(np.array_equal(self.goalState,self.goalWeights))
            return self.mismatch
        self.mismatch = weightedSum(self.goalState,self.goalWeights)
        if (self.mismatch != 0): # weighted sum is normalised
             self.mismatch = self.mismatch/sum(w for w in self.goalWeights.itervalues())
        return self.mismatch

    def calcLateralActivation(self):
        """Returns a value of activation for the weighted lateral input."""
        if (len(self.lateralState)==0):
            return 0
        wInSum = weightedSum(self.lateralState,self.lateralWeights)
#        if (wInSum != 0): # weighted sum is normalised
#                wInSum = wInSum/sum(w for w in self.lateralWeights.itervalues())
        return wInSum

    def calcControlActivation(self):
        """Returns a value of activation for the weighted control input."""
        # calculation of sum of weighted inputs (positive weights)
        if (len(self.controlState)==0):
            return 0
        wInSum = weightedSum(self.controlState,self.controlWeights)
        return wInSum

    def calcCore(self):
        """Returns a value of current activation of the FS."""
        if (self.isActive and self.onTime >= self.tau):
            self.failed = True
            self.isActive = False
            self.activity = 0
            self.onTime = 0
            if (self.calcGoalMismatch()>= self.pr_threshold):
                self.failed = False
            else:
                self.problemWeights.update(self.plasticWeights)
                print '%#%# learned fs',self.ID
                print ' wpr:',self.problemWeights
                print ' pl w:',self.plasticWeights
        else:
            wInSum = self.oldActivity
            wInSum += self.calcProblemActivation()
            wInSum -= self.calcLateralActivation()
            wInSum += self.calcControlActivation()
            wInSum += 2*(0.5-np.rand())*self.noise
            #if self.wasActive[-1] and not self.isOutput:
            if not self.isOutput:
                wInSum -= self.calcGoalMismatch()
            self.activity = sigmoid(wInSum, self.k, self.x0)
            if (self.activity >= self.threshold):
                self.isActive = True
            else:
                self.isActive = False
            if (self.mismatch >= self.pr_threshold): #the goal state has been obtained
                self.failed = False
                self.plasticWeights = {}
                #self.mismatch = 0
                self.onTime = 0
            if (self.isActive and not self.isOutput):
                self.onTime += 1
        # TODO : незабыть, что когда будет введена возможность латерального
        #        возбуждения надо будет убрать знак минус перед взвешенной
        #        суммой при ее добавлении к аргументу.
        return self.activity

    def update(self): # net is a dictionary {FSID: AtomFS}
        """Updates current state of FS."""
        self.wasActive.pop(0)
        self.wasActive.append(self.isActive)
        self.oldActivity = self.activity
        self.calcCore()
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

    def weightsUpdate(self, fsnet = {}):
        """Updates current weights of FS to exclude unimportant connections"""
        for fs in self.problemWeights.keys():
            if fsnet[fs].isActive == False:
                if self.problemWeights[fs] > self.rateOfWeightLearning:
                    self.problemWeights[fs] -= self.rateOfWeightLearning
                else: self.problemWeights[fs] = 0
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
                if ID in self.net[fs].problemWeights.keys():
                  self.net[fs].problemWeights[offspring.ID] = \
                  self.net[fs].problemWeights[ID]
                if ID in self.net[fs].lateralWeights.keys():
                    self.net[fs].lateralWeights[offspring.ID] = \
                    self.net[fs].lateralWeights[ID]
#                if ID in self.net[fs].goalWeights.keys():
#                    self.net[fs].goalWeights[offspring.ID] = \
#                    self.net[fs].goalWeights[ID]
        return offspring

    def removeFS(self, ID):
        """removes FS from the network with cleaning up all outgiong links"""
        del self.net[ID]
        for fs in self.net.keys():
            if ID in self.net[fs].problemWeights.keys():
                del self.net[fs].problemWeights[ID]
            if ID in self.net[fs].lateralWeights.keys():
                del self.net[fs].lateralWeights[ID]
            if ID in self.net[fs].goalWeights.keys():
                del self.net[fs].goalWeights[ID]

    def createFS(self, problemFS):
        if problemFS.failed: # duplication
            newFS = self.duplicate(problemFS.ID, outLnkDup = False)
        else:
            newFS = self.duplicate(problemFS.ID, outLnkDup = True)
        # update the problem state of the new FS with the current state
        newFS.problemWeights.clear()
        newFS.goalWeights.clear()
        newFS.lateralWeights.clear()
        newFS.isActive = False
        newFS.activity = 1.
        newFS.oldActivity = 1.
        newFS.mismatch = 0.
        newFS.failed = False
        newFS.onTime = 0
        newFS.isLearning = True
        # FS is relevant to the problem of the parent FS
        # newFS.problemWeights[problemFS.ID] = 1.
        # newFS.lateralWeights[problemFS.ID] = 1.
        # problemFS.goalWeights[newFS.ID] = 1.
        # alernative behavior inhibits failed behavior
        problemFS.lateralWeights[newFS.ID] = 1.
# TODO реализовать два типа обучения - 
        """ 
1) ошибка(failed) - формируется ФС торомозящая ошибочное действие в ситуации
ВХОД - проблемная ситуация в которой ФС не смогла реализоваться
ВЕСА = запоминают чем отличается проблемная ситуация в которой ФС активировалась,
от стандартной => для этого надо временно запоминать для каждой активировавшейся ФС,
чем текущая ситуация отличается от стандартной. Например, создавать ФС, которая
будет неактивна пока старая ФС ждет результат. Если результат получен, то 
резервная ФС удаляется, если нет, то от нее формируется тормозный вес на 
старую ФС.
 Или просто добавлять тормозные связи в саму ФС? ТОгда в момент активации ФС 
 "запоминает" отличающиеся входы с тормозными синапсами, но они актуализируются
 только в случае отсутствия результата.

2) успех(matched) - формируется ФС возбуждающая правильное действие
# ? Два случая различаются только знаком (типом - торм или возб) веса от 
новой ФС к моторной ФС?
        """
        for fs in self.net.keys():

            # new FS should be activated in the state
            # that created the problem for the parent FS
            if self.net[fs].isInput: # TODO проверить на необходимость учета только входов от среды
               if self.net[fs].wasActive[-1]:
                   newFS.problemWeights[fs] = 1.

            # new FS should activate other FSs (i.e. 'motor' FS)
            # that contribute to the memorising state transition
            if ( (problemFS.ID in self.matchedFS)and 
                 self.net[fs].wasActive[-1]      and
                 not (self.net[fs].isLearning or self.net[fs].isInput
                    or self.net[fs].failed or fs == newFS.parentID)):
                    self.net[fs].problemWeights[newFS.ID] = 1.

            # results of actions (i.e. neurons activated after actions)
            # should be predicted by new FS
            if (self.net[fs].isActive and self.net[fs].onTime == 1
                and self.net[fs].isInput):
                newFS.goalWeights[fs] = 1.

        #newFS.update(self)
        return newFS
        
    def updateFSInputs(self, fs, inputStates):
        """updates inputs of the given FS"""
        self.net[fs].problemState = {k: self.net[k].oldActivity
            for k in self.net[fs].problemWeights.iterkeys()}
                #if not self.net[k].isLearning}
        self.net[fs].goalState = {k: self.net[k].oldActivity
            for k in self.net[fs].goalWeights.iterkeys()}
                #if not self.net[k].isLearning}
        self.net[fs].lateralState = {k: self.net[k].oldActivity
            for k in self.net[fs].lateralWeights.iterkeys()}
                #if self.net[k].isActive and not self.net[k].isLearning}
        self.net[fs].controlState = {k: self.net[k].oldActivity
            for k in self.net[fs].controlWeights.iterkeys()}
                #if not self.net[k].isLearning}
                        
    def setPlasticWeights(self, fs, inputStates):
        """calculates mismatch between (inputs of) problem weights and
        current active inputs in the previous layer"""
        for inFS in inputStates.keys():
             if (self.net[inFS].isActive and 
                inFS not in self.net[fs].problemWeights):
                 self.net[fs].plasticWeights[inFS] = -1.
                 print 'failed weight',inFS,'->',fs
        
        
    def update(self, inputStates = {}) :
        """updates the network given values of activations for input elements"""
        self.activation = {} # dict with {fsID, activation}
        self.mismatch = {}
        # activate elements (FSs) corresponding to the inputs
        self.activateFS(inputStates)
        # update activations and predictions of hidden and effector FSs
#        for cycle in range(3): # convergence loop
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
           # updating FS inputs
            self.updateFSInputs(fs,inputStates)
            self.activation[fs], self.mismatch[fs] = self.net[fs].update()
            if self.net[fs].isActive and self.net[fs].onTime == 1:
                self.setPlasticWeights(fs,inputStates)
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
#        for i in range(len(self.failedFS)):
#            if self.net[self.failedFS[i]].isLearning:
#            print 'creating 2nd fs for failedFS:', self.failedFS[i]
#            newFS = self.createFS(self.net[self.failedFS[i]])
#            self.memoryTrace[newFS.ID] = newFS
#            print 'new fs ID:', newFS.ID

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
            self.net[links[lnk][1]].problemWeights[links[lnk][0]] = links[lnk][2]

    def addLateralLinks(self, links=[]):
        """creates  inhibition links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].lateralWeights[links[lnk][0]] = links[lnk][2]

    def addPredictionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].goalWeights[links[lnk][0]] = links[lnk][2]

    def addControlLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].controlWeights[links[lnk][0]] = links[lnk][2]

    def activateFS(self, fs_list={}):
        """sets activations for listed FSs"""

        for inFS in fs_list.keys():
            self.net[inFS].isInput = True # TODO: in the case of variable inputs this flag should be resetted
            self.net[inFS].exactInputMatch = True # HINT: remove in the case of input recognition
            # remove oldest record from activation history
            self.net[inFS].wasActive.pop(0)
            # add activation from the last time step
            self.net[inFS].wasActive.append(self.net[inFS].isActive)
            # set current value of activity
            #fs_list[inFS] = fs_list[inFS] #*self.net[inFS].threshold
            self.net[inFS].oldActivity = fs_list[inFS]
            self.net[inFS].activity = fs_list[inFS]
            self.activation[inFS] = fs_list[inFS]
#            self.net[inFS].isActive = True # TODO: remove ???
            if (self.net[inFS].oldActivity >= self.net[inFS].threshold):
                self.net[inFS].isActive = True
                self.net[inFS].onTime += 1
            else:
                self.net[inFS].isActive = False
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
        net_activity = []
        for fs in self.net.keys():
            net_activity.append(self.net[fs].activity)
            for synapse in self.net[fs].lateralWeights.keys():
                G.add_edge(synapse, fs, key=2,
                           weight=abs(self.net[fs].lateralWeights[synapse]))
            for synapse in self.net[fs].goalWeights.keys():
                G.add_edge(synapse, fs, key=1,
                           weight=self.net[fs].goalWeights[synapse])
            for synapse in self.net[fs].problemWeights.keys():
                G.add_edge(synapse, fs, key=0,
                           weight=self.net[fs].problemWeights[synapse])
            for synapse in self.net[fs].controlWeights.keys():
                G.add_edge(synapse, fs, key=3,
                           weight=abs(self.net[fs].controlWeights[synapse]))

        node_layout = nx.circular_layout(G)  #nx.graphviz_layout(G,prog="neato")
        plot.cla()
        nx.draw_networkx_nodes(G, pos=node_layout, node_size=800,
                               node_color = net_activity, cmap = plot.cm.Reds)
        nx.draw_networkx_labels(G, pos=node_layout)
        ar = plot.axes()
        actArrStyle=dict(arrowstyle='fancy',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="red",ec="none", alpha=0.85, lw=0,
                      connectionstyle="arc3,rad=-0.1",)
        inhibitionArrStyle=dict(arrowstyle='fancy',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="blue",ec="none", alpha=0.6,
                      connectionstyle="arc3,rad=-0.13",)
        predArrStyle=dict(arrowstyle='fancy',
                      shrinkA=20, shrinkB=20, aa=True,
                      fc="green",ec="none", alpha=0.7,
                      connectionstyle="arc3,rad=0.2",)
        for vertex in G.edges(keys=True,data=True): # drawing links
            #print vertex
            if vertex[3]['weight'] != 0:
                coords = [node_layout[vertex[1]][0],
                          node_layout[vertex[1]][1],
                         node_layout[vertex[0]][0],
                         node_layout[vertex[0]][1]]
                if (vertex[2] == 0) :
                    if vertex[3]['weight'] > 0:
                        actArrStyle['fc'] = plot.cm.YlOrRd(vertex[3]['weight'])
                        ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                    arrowprops = actArrStyle)
                    else:
                        actArrStyle['fc'] = plot.cm.Greys(abs(vertex[3]['weight']))
                        print '&&& plot:', vertex
                        ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                    arrowprops = actArrStyle)
                if (vertex[2] == 1) :
                    predArrStyle['fc'] = plot.cm.Greens(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = predArrStyle)
                if (vertex[2] == 2) :
                    inhibitionArrStyle['fc'] = plot.cm.Blues(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = inhibitionArrStyle)
                if (vertex[2] == 3) :
                    actArrStyle['fc'] = plot.cm.RdPu(vertex[3]['weight'])
                    ar.annotate('',(coords[0], coords[1]),(coords[2], coords[3]),
                                arrowprops = actArrStyle)



        ar.xaxis.set_visible(False)
        ar.yaxis.set_visible(False)
        plot.subplots_adjust(left=0.0, right=1., top=1., bottom=0.0)
        plot.show()
        # todo - handle self-links
