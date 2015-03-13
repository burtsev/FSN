# -*- coding: utf-8 -*-
"""This library implements classes and functions
for the Functional Systems Networks (FSN)

Created on Sun Sep 01 14:25:22 2013

@author: Burtsev
"""

from copy import deepcopy


class FSNetwork:
    """Implements a network of functional systems"""
    net = {}  # net is a dictionary {FSID: AtomFS}
    memoryTrace = {}  # is a dictionary {FSID: AtomFS}
    idCounter = int  # counter for FS id's
    failedFS = []  # a list of FSs that failed at the current time
    matchedFS = []  # a list of FSs that were failed and now have prediction satisfied
    activatedFS = []  # a list of FSs that activated at the current time
    activation = {}  # dict with {fsID, activation}
    mismatch = {}
    learningFS = []

    def __init__(self):
        self.net = {}
        self.idCounter = 0
        self.memoryDepth = 1  # how long a FS is retained in the memory trace
        self.failedFS = []  # list of FSs that failed at the current time
        self.activatedFS = []  # a list of FSs that activated at the current time
        self.matchedFS = []  # a list of FSs that were failed and now have prediction satisfied

    def add(self, fs):
        """adds FS to the network"""
        fs.ID = self.idCounter
        self.net[fs.ID] = fs
        self.idCounter += 1

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
        if problemFS.failed:  # duplication
            newFS = self.duplicate(problemFS.ID, outLnkDup=False)
        else:
            newFS = self.duplicate(problemFS.ID, outLnkDup=True)
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
            if self.net[fs].isInput:  # TODO проверить на необходимость учета только входов от среды
                if self.net[fs].wasActive[-1]:
                    newFS.problemWeights[fs] = 1.

            # new FS should activate other FSs (i.e. 'motor' FS)
            # that contribute to the memorising state transition
            if ((problemFS.ID in self.matchedFS) and
                    self.net[fs].wasActive[-1] and
                    not (self.net[fs].isLearning or self.net[fs].isInput
                         or self.net[fs].failed or fs == newFS.parentID)):
                self.net[fs].problemWeights[newFS.ID] = 1.

            # results of actions (i.e. neurons activated after actions)
            # should be predicted by new FS
            if (self.net[fs].isActive and self.net[fs].onTime == 1
                    and self.net[fs].isInput):
                newFS.goalWeights[fs] = 1.

        # newFS.update(self)
        return newFS

    # noinspection PyUnusedLocal
    def updateFSInputs(self, fs, inputStates):
        """updates inputs of the given FS"""
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

    def setPlasticWeights(self, fs, inputStates):
        """calculates mismatch between (inputs of) problem weights and
        current active inputs in the previous layer"""
        for inFS in inputStates.keys():
            if self.net[inFS].isActive and inFS not in self.net[fs].problemWeights:
                self.net[fs].plasticWeights[inFS] = -1.
                print 'failed weight', inFS, '->', fs

    def update(self, inputStates):
        """updates the network given values of activations for input elements"""
        self.activation = {}  # dict with {fsID, activation}
        self.mismatch = {}
        # activate elements (FSs) corresponding to the inputs
        self.activateFS(inputStates)
        # update activations and predictions of hidden and effector FSs
        # for cycle in range(3): # convergence loop
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            # updating FS inputs
            self.updateFSInputs(fs, inputStates)
            self.activation[fs], self.mismatch[fs] = self.net[fs].update()
            # if self.net[fs].isActive and self.net[fs].onTime == 1:
            #     self.setPlasticWeights(fs, inputStates)
        # update history of activity
        self.logActivity(inputStates)

        # learning

        # prune ineffective connections
        # for fs in (set(self.net.keys()) - set(inputStates.keys())):
        # if self.net[fs].isActive and not self.net[fs].isOutput:
        # self.net[fs].weightsUpdate(self.net)

        # TODO :
        """    implement addition of weights from active FSs
            to currently learning FSs """

        # remove tentative FSs older than memoryDepth
        for fs in self.memoryTrace.keys():
            if self.net[fs].onTime > (self.memoryDepth - 1):
                del self.memoryTrace[fs]
                self.removeFS(fs)

                # TODO:        # create alternatives for the failed FSs
                # for i in range(len(self.failedFS)):
                # if self.net[self.failedFS[i]].isLearning:
                # print 'creating 2nd fs for failedFS:', self.failedFS[i]
                # newFS = self.createFS(self.net[self.failedFS[i]])
                # self.memoryTrace[newFS.ID] = newFS
                # print 'new fs ID:', newFS.ID

        # create alternatives for the matched FSs
        for i in range(len(self.matchedFS)):
            # if self.net[self.matchedFS[i]].wasActive[-1]:
            print 'creating 2nd fs for matchedFS:', self.matchedFS[i]
            newFS = self.createFS(self.net[self.matchedFS[i]])
            self.memoryTrace[newFS.ID] = newFS
            print 'new fs ID:', newFS.ID

            # connect to the known trajectory
        # for i in range(len(self.activatedFS)):
        # if (self.net[self.failedFS[i]].wasActive[-1]
        # and self.net[self.failedFS[i]].onTime>=self.net[self.failedFS[i]].tau):
        # for fs in self.memoryTrace.keys():
        # if self.net[fs].parentID == self.failedFS[i]:
        # del self.memoryTrace[self.net[fs].ID]
        # self.net[fs].isLearning = False
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

    def activateFS(self, fs_list):
        """sets activations for listed FSs"""

        for inFS in fs_list.keys():
            self.net[inFS].isInput = True  # TODO: in the case of variable inputs this flag should be resetted
            self.net[inFS].exactInputMatch = True  # HINT: remove in the case of input recognition
            # remove oldest record from activation history
            self.net[inFS].wasActive.pop(0)
            # add activation from the last time step
            self.net[inFS].wasActive.append(self.net[inFS].isActive)
            # set current value of activity
            # fs_list[inFS] = fs_list[inFS] #*self.net[inFS].threshold
            self.net[inFS].oldActivity = fs_list[inFS]
            self.net[inFS].activity = fs_list[inFS]
            self.activation[inFS] = fs_list[inFS]
            # self.net[inFS].isActive = True # TODO: remove ???
            if self.net[inFS].oldActivity >= self.net[inFS].threshold:
                self.net[inFS].isActive = True
                self.net[inFS].onTime += 1
            else:
                self.net[inFS].isActive = False
                self.net[inFS].onTime = 0

    def setOutFS(self, fs_list):
        """marks listed FSs as outputs"""
        for outFS in range(len(fs_list)):
            self.net[fs_list[outFS]].isOutput = True

    def resetActivity(self):
        """resets activity for all FS in the net"""
        for fs in self.net.keys():
            self.net[fs].resetActivity()
        self.failedFS = []  # list of FSs that failed at the current time
        self.activatedFS = []  # a list of FSs that activated at the current time
        self.matchedFS = []  # a list of FSs that were failed and now have prediction satisfied

    def addActionLinks(self, links):
        """creates links between FSs. Input format [[start, end, weight]]"""
        if not links:
            links = []
        for lnk in range(len(links)):
            self.net[links[lnk][1]].problemWeights[links[lnk][0]] = links[lnk][2]

    def addLateralLinks(self, links):
        """creates  inhibition links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].lateralWeights[links[lnk][0]] = links[lnk][2]

    def addPredictionLinks(self, links):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].goalWeights[links[lnk][0]] = links[lnk][2]

    def addControlLinks(self, links):
        """creates links between FSs. Input format [[start, end, weight]]"""
        for lnk in range(len(links)):
            self.net[links[lnk][1]].controlWeights[links[lnk][0]] = links[lnk][2]

    def logActivity(self, inputStates):
        self.activatedFS = []
        wasFailed = self.failedFS[:]
        self.failedFS = []
        self.learningFS = []
        for inFS in inputStates.keys():
            if self.net[inFS].isActive:  # if FS is active
                self.activatedFS.append(self.net[inFS].ID)
        for fs in (set(self.net.keys()) - set(inputStates.keys())):
            self.net[fs].oldActivity = self.activation[fs]
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