# -*- coding: utf-8 -*-
"""This library implements classes and functions
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

from copy import deepcopy
import AtomFS as FS


class FSNetwork:
    """Implements a network of functional systems (FS)"""
    net = {}  # net is a dictionary {FSID: AtomFS}
    inFS = {}  # a list of input FS
    goalFS = {}  # a list of FS for the representation of goals
    hiddenFS = {}  # a list of FS for experience storage
    outFS = {}  # a list of output FS
    memoryTrace = {}  # is a dictionary {FSID: AtomFS}
    idCounter = int  # counter for FS id's
    failedFS = []  # a list of FSs that failed at the current time
    matchedFS = []  # a list of FSs that were failed and now have prediction satisfied
    activatedFS = []  # a list of FSs that activated at the current time
    activation = {}  # dict with {fsID, activation}
    mismatch = {}
    learningFS = []

    def __init__(self):
        self.inFS = {}  # a list of input FS
        self.goalFS = {}  # a list of FS for the representation of goals
        self.hiddenFS = {}  # a list of FS for experience storage
        self.outFS = {}  # a list of output FS
        self.memoryTrace = {}  # is a dictionary {FSID: AtomFS}
        self.net = {}
        self.idCounter = 0
        #  self.memoryDepth = 1  # how long a FS is retained in the memory trace
        self.failedFS = []  # list of FSs that failed at the current time
        self.activatedFS = []  # a list of FSs that activated at the current time
        self.matchedFS = []  # a list of FSs that were failed and now have prediction satisfied

    def initPredNet(self, nIn, nOut):
        """ creates FS network for the prediction (no goal FS)
        :param nIn: a number of inputs of FS network
        :param nOut: a number of outputs of FS network
        :return: FS net
        """

        for i in range(nIn):
            fs = self.add(FS.AtomFS())
            fs.isInput = True
            self.inFS[fs.ID] = fs

        for i in range(nOut):
            fs = self.add(FS.AtomFS())
            fs.isOutput = True
            self.outFS[fs.ID] = fs

        return self.net

    def initCtrlNet(self, nIn, nOut, nGoal):
        """ creates FS network for the prediction (no goal FS)
        :param nIn: a number of inputs of FS network
        :param nOut: a number of outputs of FS network
        :param nGoal: a number of goals of FS network
        :return: FS net
        """

        self.initPredNet(nIn, nOut)

        for i in range(nGoal):
            fs = self.add(FS.AtomFS())
            self.goalFS[fs.ID] = fs

        return self.net

    # noinspection PyUnusedLocal
    def updateFSInputs(self, fs):
        """updates input values of the given FS"""

        self.net[fs].problemState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].problemWeights.iterkeys()}
        # if not self.net[k].isLearning}
        self.net[fs].goalState = {k: self.net[k].oldActivity
                                  for k in self.net[fs].goalWeights.iterkeys()}
        # if not self.net[k].isLearning}
        self.net[fs].lateralState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].lateralWeights.iterkeys()}
        # if self.net[k].isActive and not self.net[k].isLearning}
        self.net[fs].controlState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].controlWeights.iterkeys()}
        # if not self.net[k].isLearning}

    def update(self, time, inputStates):
        """updates the network given values of activations for input elements"""

        self.activation = {}  # dict with {fsID, activation}
        self.mismatch = {}

        # activate elements (FSs) corresponding to the inputs
        self.activateFS(inputStates)

        # updating goal FSs
        for fs in self.goalFS.values():
            self.updateFSInputs(fs.ID)
            self.activation[fs.ID], self.mismatch[fs.ID] = fs.update(time)

        # updating hidden FSs
        for fs in self.hiddenFS.values():
            # updating FS inputs
            self.updateFSInputs(fs.ID)
            self.activation[fs.ID], self.mismatch[fs.ID] = fs.update(time)
            # if self.net[fs].isActive and self.net[fs].onTime == 1:
            #     self.setPlasticWeights(fs, inputStates)

        # updating action FSs
        for fs in self.outFS.values():
            self.updateFSInputs(fs.ID)
            self.activation[fs.ID], self.mismatch[fs.ID] = fs.update(time)

        self.logActivity()

        self.learn(time)

        return self.activation

    def learn(self, time):
        """ modifies network structure to save new experience
        :return:
        """
        activeHiddenFS = set(self.activatedFS).intersection(self.hiddenFS)

        # checking if existing tentative FSs were effective

        for fs in self.memoryTrace:
            # remove tentative FSs that expired
            if time - fs.startTime > fs.tau:  # TODO: make the removal probabilistic
                del self.memoryTrace[fs.ID]
                self.removeFS(fs.ID)

            # integrate effective FS in the network
            if fs.parentID in self.matchedFS or len(activeHiddenFS) > 0:
                fs.tau = time - fs.startTime
                # adding links to predict current state of environment
                for inFS in self.inFS:
                    if inFS.isActive:
                        fs.goalValues[inFS.ID] = inFS.activity
                        fs.goalWeights[inFS.ID] = 1
                self.hiddenFS[fs.ID] = fs
                self.net[fs.ID] = fs
                del self.memoryTrace[fs.ID]

        # generating tentative FSs for unexpected outcomes
        if len(activeHiddenFS) == 0:
            for gFS in self.goalFS.values():
                if gFS.isActive:
                    newFS = self.createFS(gFS)
                    self.memoryTrace[newFS.ID] = newFS

        # TODO: пластичность весов от мотивационной ФС
        # prune ineffective connections
        # for fs in (set(self.net.keys()) - set(inputStates.keys())):
        # if self.net[fs].isActive and not self.net[fs].isOutput:
        # self.net[fs].weightsUpdate(self.net)
    # end learn

    def activateFS(self, values):
        """sets activations for the input FSs"""
        for i, fs in enumerate(self.inFS.itervalues()):
            self.activation[fs.ID] = fs.setFSActivation(values[i])

    def resetActivity(self):
        """resets activity for all FS in the net"""
        for fs in self.net.keys():
            self.net[fs].resetActivity()
        self.failedFS = []  # list of FSs that failed at the current time
        self.activatedFS = []  # a list of FSs that activated at the current time

    def setOutFS(self, fs_list):
        """marks listed FSs as outputs"""
        for outFS in range(len(fs_list)):
            self.net[fs_list[outFS]].isOutput = True

    def setPlasticWeights(self, fs, inputStates):
        """calculates mismatch between (inputs of) problem weights and
        current active inputs in the previous layer"""
        for inFS in inputStates.keys():
            if self.net[inFS].isActive and inFS not in self.net[fs].problemWeights:
                self.net[fs].plasticWeights[inFS] = -1.
                print 'failed weight', inFS, '->', fs

    def add(self, fs):
        """adds FS to the network"""

        fs.ID = self.idCounter
        self.net[fs.ID] = fs
        self.idCounter += 1

        return fs

    def duplicate(self, ID, outLnkDup=False):  # outLnkDup is optional parameter
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
                    # if ID in self.net[fs].goalWeights.keys():
                    # self.net[fs].goalWeights[offspring.ID] = \
                    # self.net[fs].goalWeights[ID]
        return offspring

    def removeFS(self, ID):
        """removes FS from the network with cleaning up all outgoing links"""
        del self.net[ID]
        for fs in self.net.keys():
            if ID in self.net[fs].problemWeights.keys():
                del self.net[fs].problemWeights[ID]
            if ID in self.net[fs].lateralWeights.keys():
                del self.net[fs].lateralWeights[ID]
            if ID in self.net[fs].goalWeights.keys():
                del self.net[fs].goalWeights[ID]

    def createFS(self, problemFS):

        newFS = self.add(FS.AtomFS())

        newFS.controlWeights[problemFS.ID] = 1

        # adding links to recognize current state of environment
        for fs in self.inFS:
            if fs.isActive:
                newFS.problemValues[fs.ID] = fs.activity
                newFS.problemWeights[fs.ID] = 1

        # TODO:  adding links to recognize current state of the self activation
        # for fs in self.hiddenFS:
        #     if fs.isActive:
        #         newFS.problemWeights[fs.ID] = 1

        # adding links to actions
        for fs in self.outFS:
            if fs.isActive:
                fs.controlWeights[newFS.ID] = 1

        return newFS

    def addActionLinks(self, links):
        """creates links between FSs. Input format [[start, end, weight]]"""
        if not links:
            links = []
        for lnk in range(len(links)):
            self.net[links[lnk][1]].problemValues[links[lnk][0]] = links[lnk][2]
            self.net[links[lnk][1]].problemWeights[links[lnk][0]] = 1

    def addLateralLinks(self, links):
        """creates  inhibition links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].lateralWeights[links[lnk][0]] = links[lnk][2]

    def addPredictionLinks(self, links):
        """creates links between FSs. Input format [[start, end, value]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].goalValues[links[lnk][0]] = links[lnk][2]
            self.net[links[lnk][1]].goalWeights[links[lnk][0]] = 1

    def addControlLinks(self, links):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].controlWeights[links[lnk][0]] = links[lnk][2]

    def logActivity(self):
        self.activatedFS = []
        wasFailed = self.failedFS[:]
        self.failedFS = []
        self.learningFS = []
        for inFS in self.inFS.keys():
            if self.net[inFS].isActive:  # if FS is active
                self.activatedFS.append(self.net[inFS].ID)
        for fs in (set(self.net.keys()) - set(self.inFS.keys())):
            # self.net[fs].oldActivity = self.activation[fs]
            # if FS has failed to reach predicted state
            if self.net[fs].failed and not self.net[fs].isLearning:
                self.failedFS.append(self.net[fs].ID)  # add FS to the failers list
                if self.net[fs].ID in wasFailed:
                    wasFailed.remove(self.net[fs].ID)
            # if FS is active
            if self.net[fs].isActive:
                self.activatedFS.append(self.net[fs].ID)
            if self.net[fs].isLearning:
                self.learningFS.append(fs)
        self.matchedFS = wasFailed[:]

    # end of logActivity