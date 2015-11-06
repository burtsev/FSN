# -*- coding: utf-8 -*-
"""This library implements classes and functions
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

from copy import deepcopy
import random
import AtomFS as FS


def probSel(out_fs):
    rnd = random.random() * sum([ofs.activity for ofs in out_fs])
    for ofs in out_fs:
        rnd -= ofs.activity
        if rnd < 0:
            return ofs.ID


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
    activatedFS = []  # a list of FSs activated at the current time
    usedFS = []  # a list of FSs used at the current trial
    activation = {}  # dict with {fsID, activation}
    activationHist = {}
    mismatch = {}
    mismatchHist = {}
    learningFS = []
    prnLg = False
    reentry = 2

    def __init__(self):
        self.inFS = {}  # a list of input FS
        self.goalFS = {}  # a list of FS for the representation of goals
        self.hiddenFS = {}  # a list of FS for experience storage
        self.outFS = {}  # a list of output FS
        self.memoryTrace = {}  # is a dictionary {FSID: AtomFS}
        self.net = {}
        self.idCounter = 0
        # self.memoryDepth = 1  # how long a FS is retained in the memory trace
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

    def updateFSInputs(self, fs):
        """updates input values of the given FS"""

        self.net[fs].problemState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].problemWeights.iterkeys()
                                     if not self.net[k].wasUsed and self.net[k].isActive}
        self.net[fs].goalState = {k: self.net[k].oldActivity
                                  for k in self.net[fs].goalWeights.iterkeys()
                                  if not self.net[k].wasUsed and self.net[k].isActive}
        self.net[fs].lateralState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].lateralWeights.iterkeys()
                                     if not self.net[k].wasUsed and self.net[k].isActive}
        self.net[fs].controlState = {k: self.net[k].oldActivity
                                     for k in self.net[fs].controlWeights.iterkeys()
                                     if not self.net[k].wasUsed and self.net[k].isActive}

    def update(self, time, inputStates, t):
        """feedforward update of the network given values of activations for input elements"""

        self.activation = {}  # dict with {fsID, activation}
        self.mismatch = {}

        # activate elements (FSs) corresponding to the inputs with input values
        self.activateFS(inputStates)

        # updating goal FSs
        for fs in self.goalFS.values():
            self.updateFSInputs(fs.ID)
            self.activation[fs.ID], self.mismatch[fs.ID] = fs.update(time)
            fs.wasUsed = False

        # updating hidden FSs
        fs_s = sorted(self.hiddenFS.keys())
        for fs in fs_s:
            # updating FS inputs
            self.updateFSInputs(fs)
            self.activation[fs], self.mismatch[fs] = self.hiddenFS[fs].update(time)

        # updating action FSs
        self.updOut(time)

        # re-checking goal FSs
        for fs in self.goalFS.values():
            if fs.mismatch >= fs.pr_threshold:
                self.resetUsedFS(fs)

        for fs in self.goalFS.values():
            fs.oldActivity = fs.activity
        for fs in self.hiddenFS.values():
            fs.oldActivity = fs.activity

        self.logActivity(time, t)


    def step(self, time, inputStates):

        self.updateWorkingMemory(time)
        self.matchedFS = []

        for t in range(self.reentry):
            self.update(time, inputStates, t)
            if self.prnLg:
                print '----- loop:', t
                self.printLog()

        self.learn(time)

        return self.activation


    def learn(self, time):
        """ modifies network structure to save new experience
        :return:
        """

        activeHiddenFS = []
        activeHiddenUsedFS = []
        for fs in self.hiddenFS.values():
            if fs.isActive:
                if fs.wasUsed:
                    activeHiddenUsedFS.append(fs)
                else:
                    activeHiddenFS.append(fs)
        # set(self.activatedFS).intersection(self.hiddenFS.keys())

        # checking if existing tentative FSs were effective
        for fs in self.memoryTrace.values():
            # integrate effective FS in the network
            # print fs.ID, " learn/ par.:", fs.parentID, "# hidFS:", len(activeHiddenFS)
            if len(set(fs.goalID).intersection(self.matchedFS)) > 0 \
                    or len(activeHiddenFS) > 0: #fs.parentID in self.matchedFS or
                fs.tau = time - fs.startTime
                # adding links to predict current state of environment
                for inFS in self.inFS.values():
                    if inFS.isActive:
                        fs.goalValues[inFS.ID] = inFS.activity
                        fs.goalWeights[inFS.ID] = 1
                for activeHFS in activeHiddenFS:
                    activeHFS.lateralWeights[fs.ID] = 0.1  # the weight for the sequence
                self.hiddenFS[fs.ID] = fs
                # self.net[fs.ID] = fs
                del self.memoryTrace[fs.ID]

                print "fs:", fs.ID, "is activated!  <<<<< <<< <<  <  <"
                print "fs.prob:", fs.problemValues
                print "fs.goal:", fs.goalValues
                print "fs.ctrl:", fs.controlWeights
                print "fs.lat:", fs.lateralWeights

        # generating tentative FSs for unexpected outcomes
        if len(activeHiddenFS) == 0:
            newFS = self.createFS(time)
            self.memoryTrace[newFS.ID] = newFS
            # print newFS.ID, "created / parent:", newFS.parentID

            # TODO: пластичность весов от мотивационной ФС
            # prune ineffective connections
            # for fs in (set(self.net.keys()) - set(inputStates.keys())):
            # if self.net[fs].isActive and not self.net[fs].isOutput:
            # self.net[fs].weightsUpdate(self.net)

    # end learn

    def createFS(self, time):

        newFS = self.add(FS.AtomFS())
        newFS.startTime = time

        # adding links to recognize current state of environment
        for fs in self.inFS.values():
            if fs.isActive:
                newFS.problemValues[fs.ID] = fs.activity
                newFS.problemWeights[fs.ID] = 1

        # adding links to lateral FS
        for fs in self.hiddenFS.values():
            if fs.isActive and fs.wasUsed:
                fs.lateralWeights[newFS.ID] = -1
                newFS.lateralWeights[fs.ID] = -1

        # adding links to actions
        for fs in self.outFS.values():
            if fs.isActive:
                fs.controlWeights[newFS.ID] = 2

        for gFS in self.goalFS.values():
            if gFS.isActive or gFS.failed:
                newFS.goalID.append(gFS.ID)
                newFS.controlWeights[gFS.ID] = 1
                gFS.controlWeights[newFS.ID] = -1

        # TODO:  adding links to recognize current state of the self activation
        # for fs in self.hiddenFS:
        # if fs.isActive:
        # newFS.problemWeights[fs.ID] = 1

        return newFS

    def updateWorkingMemory(self, time):

        for fs in self.memoryTrace.values():
            # remove tentative FSs that expired
            if (time - fs.startTime) > fs.tau:  # TODO: make the removal probabilistic
                del self.memoryTrace[fs.ID]
                # print "fs:", fs.ID, "is deleted. tau =", fs.tau
                self.removeFS(fs.ID)

    def updOut(self, time):
        noActiveOut = True
        maxOut = (0, 0)
        for fs in self.outFS.values():
            self.updateFSInputs(fs.ID)
            self.activation[fs.ID], self.mismatch[fs.ID] = fs.update(time)
            fs.wasUsed = False
            if fs.activity > maxOut[1]:
                maxOut = (fs.ID, fs.activity)
                if fs.isActive:
                    noActiveOut = False
        if noActiveOut:
            if maxOut[1] == 0:
                fs = random.sample(self.outFS.keys(), 1)
                self.outFS[fs[0]].isActive = True
            else:
                self.outFS[probSel(self.outFS.values())].isActive = True
        else:
            for fs in self.outFS.values():
                if fs.isActive and fs.ID != maxOut[0]:
                    fs.isActive = False

    def resetUsedFS(self, gFS):
        """ reactivates hidden FS that were used for the completion of the goal represented by gFS
        :param gFS: FS with a goal completed
        :return:        """
        for fs_id in gFS.controlWeights.keys():
            if gFS.controlWeights[fs_id] == -1 and self.net[fs_id].wasUsed:
                self.net[fs_id].wasUsed = False
                #  print '# # # reset activity for FS:', fs_id

    def activateFS(self, values):
        """sets activations for the input FSs"""
        for fs in self.inFS.itervalues():
            self.activation[fs.ID] = fs.setFSActivation(values[fs.ID])
            fs.wasUsed = False

    def resetActivity(self):
        """resets activity for all FS in the net"""
        for fs in self.net.keys():
            self.net[fs].resetActivity()
        self.failedFS = []  # list of FSs that failed at the current time
        self.activatedFS = []  # a list of FSs that activated at the current time

        for fs in self.usedFS:
            self.net[fs].wasUsed = False
        self.usedFS = []

        for fs in self.memoryTrace.values():
            del self.memoryTrace[fs.ID]
            self.removeFS(fs.ID)

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
            if ID in self.net[fs].controlWeights.keys():
                del self.net[fs].controlWeights[ID]

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

    def logActivity(self, time, t):

        self.activatedFS = []
        wasFailed = self.failedFS[:]
        self.failedFS = []
        self.learningFS = []

        for inFS in self.inFS.keys():
            if self.net[inFS].isActive:  # if FS is active
                self.activatedFS.append(self.net[inFS].ID)

        for fs in (set(self.net.keys()) - set(self.inFS.keys())):

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

        for fs in wasFailed:
            self.matchedFS.append(fs)

        self.usedFS = [fs.ID for fs in self.net.values() if fs.wasUsed]

        self.activationHist[(time+float(t)/float(self.reentry))] = self.activation
        self.mismatchHist[(time+float(t)/float(self.reentry))] = self.mismatch

        # end of logActivity

    def printLog(self):

        print 'activations:', {k: round(v, 2) for k, v in self.activation.iteritems()}
        # print 'mismatches:', {k: round(v, 2) for k, v in FSNet.mismatch.iteritems()}
        print 'active:', self.activatedFS
        print 'usedFS:', self.usedFS
        print 'hidden:', self.hiddenFS.keys(), len(self.hiddenFS)
        print 'failed:', self.failedFS
        print 'learning:', self.learningFS
        print 'mem trace:', self.memoryTrace.keys()
        print 'matched:', self.matchedFS
        #    print 'net:', FSNet