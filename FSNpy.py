# -*- coding: utf-8 -*-
"""This library implements classes and functions 
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

import scipy as np
from copy import deepcopy

class AtomFS:
    """Class for the elementary functional system (FS).
    
        This class implements basic FS functionality:
        1) activation in the problem state;
        2) deactivation in the goal state;
        3) tracking time of transition from the problem to the goal
    """
    # FS attributes
    ID = int # FS identificator
    parentID = int # identificator of parent FS 
    onTime = int # the period of current FS's activity
    problemState = [] # description (vector of values) 
                 # of the problem to be solved by FS
    goalState = [] # description (vector of values) of the required solution
    activity = float # current value of FS activity
    activityOld = float # value of FS activiy on the previous time step
    mismatch = float # current value of mismatch between goal and current state
    isActive = bool # presence of FS activity
    wasActive = bool # history of FS activity
    learning = bool # learning state
    failed = bool # FS was unable to achive goal state  
    
    # FS parameters
    activationWeights = {} # wieghts for the problemState input
    predictionWeights = {} # weights for the goal input
    inhibitionWeights = {} # weights for the lateral inhibition
    rateOfWeightLearning = 0.001 
    tau = float # expected time required for transition from the problem to the goal state
    threshold = float # for the activation
    noise = float # random value to be added to the FS activation
    k = float
    x0 = float  
        
    def __init__(self):
        """"Creates and initilize FS."""
        self.ID = 0
        self.activity = 0.
        self.oldActivity = 0.
        self.mismatch = 0.
        self.isActive = False
        self.wasActive = False
        self.failed = False
        self.learning = False
        self.threshold = 0.9
        self.onTime = 0
        self.tau = 1
        self.k = 3      # k and x0 are choosen to have output 0.5 for normalized weighted 
        self.x0 = 0.5    # input of 0.5 and high activation for input = 1
        self.noise = 0.2
        self.activationWeights = {}
        self.predictionWeights = {}
        self.inhibitionWeights = {}
    
    def set_params(self, pw, gw, t, th, n):
        """"set parameters of FS."""
        self.activationWeights = pw
        self.predictionWeights = gw
        self.threshold = th
        self.tau = t
        self.noise = n
        
    def activation(self, weightedIn, k, x0, sigma,
                   weightedNegIn = [[0,0]], prediction = False): 
        """Returns value of activation for the weighted input.
        
        weightedIn is a list of pairs [activation, weight]"""
        
        def sigmoid(x): # sigmoid activation function
            return 1/(1+np.exp(-k*(x-x0)))
        wInSum = (np.array(weightedIn).prod(1).sum()/ 
                np.array(weightedIn)[:,1].sum()- # the argument is normalised      
                +2*(0.5-np.rand())*sigma)
        neg = np.array(weightedNegIn).prod(1).sum()
        if (neg > 0):
            wInSum -= 4*neg/np.array(weightedNegIn)[:,1].sum()    
        if not prediction: # self amplification is not required for the prediction
            wInSum += self.oldActivity
        return (sigmoid(wInSum)) 
                
    def weightsUpdate(self, fsnet = {}):
        """Updates current weights of FS to exclude unimportant connections"""
        for fs in self.activationWeights.keys():
            if fsnet[fs].isActive == False:
                if self.activationWeights[fs] > self.rateOfWeightLearning:
                    self.activationWeights[fs] -= self.rateOfWeightLearning
                else: self.activationWeights[fs] = 0
    
    def updateActivation(self, fsnet = {}):
        """Updates activation of FS"""
        self.problemState = [] # updating the problem input
        for onFS in self.activationWeights.keys(): # calculates weighted inputs for the FS activation
            self.problemState += [[fsnet[onFS].oldActivity, 
                                   self.activationWeights[onFS]]]
        competition = [[0,0]] # updating the inhibition
        for onFS in self.inhibitionWeights.keys(): # calculates weighted inputs for the FS activation
            competition += [[fsnet[onFS].oldActivity, 
                                   self.inhibitionWeights[onFS]]]
        self.activity = self.activation (self.problemState, self.k, 
                                         self.x0, self.noise, competition)
        # FS is activated if it's problem state is present
        if (self.activity >= self.threshold): 
            self.isActive = True
       
        return self.activity
    
    def updatePrediction(self, fsnet = {}):
        """Updates prediction of FS"""
        self.goalState = [] # updating the goal input
        for offFS in self.predictionWeights.keys(): # calculates weighted inputs for the FS deactivation
            self.goalState += [[fsnet[offFS].oldActivity, 
                                self.predictionWeights[offFS]]]
                                
        # goal recognition should be more definite than activation, hence, k=10 and noise=0
        self.mismatch = self.activation(self.goalState, 10, 
                                            self.x0, 0, 
                                            prediction = True)
        if (self.mismatch >= self.threshold): # the goal state has been obtained
            self.failed = False
            self.isActive = False # FS is not needed any more
            self.mismatch = 0
            if self.learning:
                self.tau = self.onTime
                self.learning = False
        else: # the goal is not achived in expected timeframe
            if ((self.onTime >= self.tau)): #and not self.learning): 
                self.failed = True
                self.isActive = False
                #self.activity = 0  
        return self.mismatch
    
    def update(self, fsnet = {}): # net is a dictionary {FSID: AtomFS}
        """Updates current state of FS."""   
        self.wasActive = self.isActive
        self.updateActivation(fsnet)
        if (self.isActive and self.wasActive == False and not self.failed): 
            self.onTime = 0 # internal time is reset 
        if (self.isActive or self.failed): 
            self.updatePrediction(fsnet)
        # if FS is inactive or not failed            
#        else: 
#            self.updateActivation(fsnet)
                 
        
        if (self.isActive or self.failed or self.learning): # update FS lifetime
            self.onTime += 1        
        return self.activity, self.mismatch # end of AtomFS.update

class FSNetwork:
    """Implements a network of functional systems
    
todo 
    1) при обновлении сети вести списки ФС, находящихся в том или ином состоянии    
    """    
    net = {} # net is a dictionary {FSID: AtomFS}
    workingMemory = {} # is a dictionary {FSID: AtomFS}
    idCounter = int # counter for FS id's
    failedFS = [] # a list of FSs that failed at the current time
    activatedFS = []  # a list of FSs that activated at the current time   
    activation = {} # dict with {fsID, activation}
    mismatch = {}
        
    def __init__(self):
        self.net = {}
        self.idCounter = 0
        self.failedFS = [] # list of FSs that failed at the current time
        self.activatedFS = [] # a list of FSs that activated at the current time
    
    def add(self, fs):
        """adds FS to the network"""
        fs.ID = self.idCounter
        self.net [fs.ID] = fs
        self.idCounter +=1

    def duplicate(self, ID):
        """duplicates FS and returns offspring"""
        offspring = deepcopy(self.net[ID])
        offspring.parentID = ID
        #self.add(offspring)
        for fs in self.net.keys():
            if ID in self.net[fs].activationWeights:
                self.net[fs].activationWeights[offspring.ID] = \
                self.net[fs].activationWeights[ID]
            if ID in self.net[fs].predictionWeights:
                self.net[fs].predictionWeights[offspring.ID] = \
                self.net[fs].predictionWeights[ID]
        return offspring
    
    def addActionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):    
            self.net[links[lnk][1]].activationWeights[links[lnk][0]] = links[lnk][2]
            
    def addInhibitionLinks(self, links=[]):
        """creates  inhibition links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):    
            self.net[links[lnk][1]].inhibitionWeights[links[lnk][0]] = links[lnk][2]
        
    def addPredictionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):    
            self.net[links[lnk][1]].predictionWeights[links[lnk][0]] = links[lnk][2]
      
    def activateFS(self, fs_list={}):
        """sets activations for listed FSs"""
        for fs in fs_list:
            self.net[fs].isActive = True
            self.net[fs].activity = fs_list[fs] 
            
    def createFS(self, problemFS):
        newFS = self.duplicate(problemFS.ID)
        # no need to worry the problem should be solved by new FS
        # problemFS.failed = False 
        # update the problem state of the new FS with the current state
        newFS.activationWeights.clear()
        newFS.isActive = False
        newFS.activity = 0
        newFS.mismatch = 0
        newFS.isActive = False
        newFS.failed = False
        newFS.onTime = 0
        newFS.learning = True
        newFS.activationWeights[problemFS.ID] = 1.
        for fs in self.net.keys():
            if self.net[fs].wasActive: newFS.activationWeights[fs] = 1.
            if self.net[fs].isActive:
                self.net[fs].activationWeights[newFS.ID] = 1.
                newFS.predictionWeights[fs] = 1.
                
        #newFS.update(self)
        return newFS
        ### todo
    
    def update(self, inputStates = {}):
        """updates the network given values of activations for input elements"""
        self.activation = {} # dict with {fsID, activation}
        self.mismatch = {}
        
        # activate elements (FSs) correspondent to inputs
        for inFS in inputStates.keys():
            self.net[inFS].wasActive = self.net[inFS].isActive
            self.net[inFS].oldActivity = inputStates[inFS]
            self.activation[inFS] = inputStates[inFS]
            if (self.net[inFS].oldActivity >= self.net[inFS].threshold):
                self.net[inFS].isActive = True
            else:
                self.net[inFS].isActive = False
                
        # update activations and predictions of hidden and effector FSs
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            self.activation[fs], self.mismatch[fs] = self.net[fs].update(self.net)
            
        # learning
        # prune ineffective connections
#        for fs in (set(self.net.keys()) - set(inputStates.keys())):
#            if self.net[fs].isActive: self.net[fs].weightsUpdate(self.net)
            """ todo: 
            implement addition of weights from active FSs 
            to currently learning FSs """
        # create alternatives for the failed FSs
        for inFS in inputStates.keys(): 
            self.net[inFS].oldActivity = self.net[inFS].activity
#        for i in range(len(self.failedFS)):
#            newFS = self.createFS(self.net[self.failedFS[i]])
#            self.workingMemory[newFS.ID] = newFS
            #if self.net[failedFS[i]].failed == False:
                # if the goal has been achived create new FS connetcting  
                # previous state and action with prediction of the goal
                       
        # update history of activity
        self.activatedFS = []
        self.failedFS = []    
        for inFS in inputStates.keys(): 
            self.net[inFS].activity = inputStates[inFS]
            if self.net[inFS].isActive: # if FS is active
               self.activatedFS.append(self.net[inFS].ID)    
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            self.net[fs].oldActivity = self.activation[fs]
            if self.net[fs].failed and (self.net[fs].ID not in self.failedFS): # if FS has failed to reach predicted state
               self.failedFS.append(self.net[fs].ID) # add FS to the failers list 
            if self.net[fs].isActive: # if FS is active
               self.activatedFS.append(self.net[fs].ID)      
#        for fs in self.workingMemory.keys():
#                self.add(self.)
        map(self.add, self.workingMemory.itervalues())
        self.workingMemory = {}
        return self.activation
            
    def drawNet(self):
        """draws the FS network"""
        import networkx as nx
        import matplotlib.pyplot as plot

        G=nx.MultiDiGraph()
        G.add_nodes_from(self.net.keys())
        actionEdges = []
        actionWeights = []
        inhibitionEdges = []
        inhibitionWeights = []
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
            for synapse in self.net[fs].inhibitionWeights.keys():
                G.add_edge(synapse, fs, key=2, 
                           weight=self.net[fs].inhibitionWeights[synapse])
                inhibitionEdges.append((synapse, fs))
                inhibitionWeights.append(self.net[fs].inhibitionWeights[synapse])   
        node_layout = nx.circular_layout(G)  #nx.graphviz_layout(G,prog="neato")
        plot.cla()
        nx.draw_networkx_nodes(G, pos=node_layout, node_color = net_activity,
                               cmap = plot.cm.Reds)
        nx.draw_networkx_labels(G, pos=node_layout)        
        ar = plot.axes() 
        actArrStyle=dict(arrowstyle='simple',                                   
                      shrinkA=10,
                      shrinkB=10,
                      fc="red",ec="none",
                      connectionstyle="arc3,rad=-0.1",)
        inhibitionArrStyle=dict(arrowstyle='simple',                                   
                      shrinkA=10,
                      shrinkB=10,
                      fc="blue",ec="none",
                      connectionstyle="arc3,rad=-0.2",)
        predArrStyle=dict(arrowstyle='simple',                                   
                      shrinkA=10,
                      shrinkB=10,
                      fc="green",ec="none",
                      connectionstyle="arc3,rad=0.1",)
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
        plot.show()        
        # todo - handle self-links
