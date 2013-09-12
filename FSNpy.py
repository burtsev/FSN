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
    mismatch = float # current value of mismatch between goal and current state
    isActive = bool # presence of FS activity
    failed = bool # FS was unable to achive goal state  
    
    # FS parameters
    activationWeights = {} # wieghts for the problemState input
    predictionWeights = {} # weights for the goal input
    tau = float # expected time required for transition from the problem to the goal state
    threshold = float # for the activation
    noise = float # random value to be added to the FS activation
    k = float
    x0 = float  
        
    def __init__(self):
        """"Creates and initilize FS."""
        self.ID = 0        
        self.activity = 0
        self.mismatch = 0
        self.isActive = False
        self.failed = False
        self.tau = 1
        self.k = 5       # k and x0 are choosen to have output 0.5 for normalized weighted 
        self.x0 = 0.5    # input of 0.5 and high activation for input = 1
        self.onTime = 0 
        self.noise = 0.01 
        self.activationWeights = {}
        self.predictionWeights = {}
    
    def set_params(self,  pw, gw, t, th, n):
        """"set parameters of FS."""
        self.activationWeights = pw
        self.predictionWeights = gw
        self.threshold = th
        self.tau = t
        self.noise = n
        
    def activation(self, weightedIn, k, x0, sigma): # activation function
        """Returns value of activation for the weighted input."""
        def sigmoid(x): # sigmoid activation function
            return 1/(1+np.exp(-k*(x-x0)))
        return (sigmoid(np.array(weightedIn).prod(1).sum()/ 
                np.array(weightedIn)[:,1].sum())+np.rand()*sigma)# the argument is normalised 
     
    def weightsUpdate(self):
        """Updates current weights of FS to exclude unimportant connections."""
        pass
    
    def update(self, fsnet = {}): # net is a dictionary {FSID: AtomFS}
        """Updates current state of FS."""
        
        if (self.isActive or self.failed): 
            self.goalState = [] # updating the goal input
            for offFS in self.predictionWeights.keys(): # calculates weighted inputs for the FS deactivation
                self.goalState += [[fsnet[offFS].activity, 
                                    self.predictionWeights[offFS]]]
            self.mismatch = self.activation(self.goalState, self.k, 
                                                self.x0, self.noise)
            if (self.mismatch >= self.threshold): # the goal state has been obtained
                self.failed = False
                self.isActive = False # FS is not needed any more
                self.mismatch = 0                
            else:
                if (self.onTime > self.tau): # the goal is not achived in expected timeframe
                    self.failed = True
                    self.isActive = False
                    self.activity = 0  
                    
        else: # if FS is inactive or not failed
            self.problemState = [] # updating the problem input
            for onFS in self.activationWeights.keys(): # calculates weighted inputs for the FS activation
                self.problemState += [[fsnet[onFS].activity, 
                                       self.activationWeights[onFS]]]
            self.activity = self.activation (self.problemState, self.k, 
                                             self.x0, self.noise)
            if (self.activity >= self.threshold): # FS is activated if it's problem state is present
                self.isActive = True
                self.onTime = 0 # internal time is reset   
                
        if (self.isActive or self.failed): # update FS lifetime
            self.onTime += 1
        return self.activity

class FSNetwork:
    """Implements a network of functional systems
    
todo 
    1) при обновлении сети вести списки ФС, находящихся в том или ином состоянии    
    2) исследовать необходимость добавления специального тормозного входа
    """
    
    net = {} # net is a dictionary {FSID: AtomFS}
    idCounter = int # counter for FS id's    
    
    def __init__(self):
        self.net = {}
        self.idCounter = 0
    
    def add(self, fs):
        """adds FS to the network"""
        fs.ID = self.idCounter
        self.net [fs.ID] = fs
        self.idCounter +=1

    def duplicate(self, ID):
        """duplicates FS and returns offspring"""
        offspring = deepcopy(self.net[ID])
        offspring.parentID = ID
        self.add(offspring)
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
#        if len(links) >= 1 :
        for lnk in range(len(links)):    
            self.net[links[lnk][1]].activationWeights[links[lnk][0]] = links[lnk][2]
#        else:
#            self.net[links[1]].activationWeights[links[0]] = links[2]
        
    def addPredictionLinks(self, links=[]):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):    
            self.net[links[lnk][1]].predictionWeights[links[lnk][0]] = links[lnk][2]
      
    def activateFS(self, fs_list={}):
        """sets activations for listed FSs"""
        for fs in fs_list:
            self.net[fs].isActive = True
            self.net[fs].activity = fs_list[fs]         
    
    def update(self, inputStates = {}):
        """updates the network given values of activations for the input elements"""
        activation = {}
        for inFS in inputStates.keys():
            self.net[inFS].activity = inputStates[inFS]
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            activation[fs] = self.net[fs].update(self.net)
        return activation
            
    def drawNet(self):
        """draws the FS network"""
        import networkx as nx
        import matplotlib.pyplot as plot

        G=nx.MultiDiGraph()
        G.add_nodes_from(self.net.keys())
        actionEdges = []
        actionWeights = []
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
        node_layout = nx.circular_layout(G)  #nx.graphviz_layout(G,prog="neato")
        plot.cla()
        nx.draw_networkx_nodes(G, pos=node_layout, node_color = net_activity, cmap = plot.cm.Reds)
        nx.draw_networkx_labels(G, pos=node_layout)        
        ar = plot.axes() 
        actArrStyle=dict(arrowstyle='simple',                                   
                      shrinkA=10,
                      shrinkB=10,
                      fc="red",ec="none",
                      connectionstyle="arc3,rad=-0.1",)
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
        
        ar.xaxis.set_visible(False)
        ar.yaxis.set_visible(False)        
        plot.draw()
        plot.show()        
        # todo - handle self-links
        
        
            
        
        