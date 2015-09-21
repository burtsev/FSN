# -*- coding: utf-8 -*-
__author__ = 'Burtsev'

import scipy as np

""" Some general functions."""


def sigmoid(x, k, x0):  # sigmoid activation function
    return 1 / (1 + np.exp(-k * (x - x0)))


def weightedSum(inputs, weights, norm=False):  # calculation of weighted sum, arguments are lists
    if len(inputs) > 0:
        wsum = np.array([[inputs[i], weights[i]]
                         for i in weights.iterkeys() if i in inputs]).prod(1).sum()
        if norm and wsum != 0:
            wsum = wsum / sum(abs(weights))
        return wsum
    else:
        return 0


def rbf(inputs, centroids, weights):
    if len(inputs) > 0:
        icw = np.array([[inputs[i], centroids[i], weights[i]]
                        for i in inputs.keys()])
        sw = np.absolute(np.subtract(icw[:, 0], icw[:, 1]))
        return np.exp(-10 * np.multiply(sw, icw[:, 2]).sum())  # /len(inputs))
    else:
        return 0


class AtomFS:
    """Class for the elementary functional system (FS).

        This class implements basic FS functionality:
        1) activation in the problem state;
        2) deactivation in the goal state;
        3) tracking time of transition from the problem to the goal
    """
    # FS attributes
    # - metadata
    ID = int  # FS id
    goalID = []  # ids of goal FS's
    # - structural parameters
    problemWeights = {}  # weights for the problemState input
    problemValues = {}  # centroids for the problemState input
    goalWeights = {}  # weights for the goal input
    goalValues = {}
    lateralWeights = {}  # weights for the lateral inhibition
    controlWeights = {}  # weights for the top-down control
    plasticWeights = {}  # temporary weights for predictive features of env.
    # - dynamical parameters
    tau = float  # expected time for transition from the problem to the goal state
    threshold = float  # for the activation
    noise = float  # random value to be added to the FS activation
    k = float
    x0 = float
    pr_threshold = float  # for the prediction
    pr_k = float
    pr_x0 = float
    rateOfWeightLearning = 0.1
    # - state variables
    problemState = {}  # input for the features of the problem to be solved by FS
    goalState = {}  # input for the features of the required solution
    lateralState = {}  # input for the lateral inhibition (activation)
    controlState = {}  # input for the top-down control
    activity = float  # current value of FS activity
    wasActive = []  # history of FS activity
    onTime = float  # the period of current FS's activity
    startTime = float  # time of the activation for the current FS's activity
    mismatch = float  # current value of mismatch between goal and current state
    # - flags
    isActive = bool  # presence of FS activity
    isLearning = bool  # learning state
    failed = bool  # FS was unable to achieve the goal state
    wasUsed = bool  # FS was already activated during current goal-directed behavior
    isInput = bool  # is true if value is set externally
    isOutput = bool  # is true if the value is not predicted
    exactInputMatch = bool  # is true if the FS should be (de)activated only
    # in the case when the input exactly matches the weights

    def __init__(self):
        """"Create and initialize FS."""
        self.ID = 0
        self.problemWeights = {}
        self.problemValues = {}
        self.goalWeights = {}
        self.goalValues = {}
        self.lateralWeights = {}
        self.controlWeights = {}
        self.plasticWeights = {}
        self.tau = 1
        self.threshold = 0.95
        self.noise = 0.001
        self.k = 10  # k and x0 are chosen to have output 0.5 for normalized weighted
        self.x0 = 0.5  # input of 0.5 and high activation for input = 1
        self.pr_threshold = self.threshold
        self.pr_k = self.k
        self.pr_x0 = self.x0
        self.activity = 0.
        self.oldActivity = 0.
        self.wasActive = []
        for i in range(2):  # depth of the activation memory is 2
            self.wasActive.append(False)
        self.onTime = 0.
        self.startTime = 0.
        self.mismatch = 0.
        self.isActive = False
        self.isLearning = False
        self.failed = False
        self.wasUsed = True
        self.isInput = False
        self.isOutput = False
        self.exactInputMatch = False

    def set_params(self, pw, gw, t, th, n, cw):
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
            if len(self.problemWeights) > 0:
                return int(np.array_equal(self.problemState, self.problemWeights))
            return 0

        return rbf(self.problemState, self.problemValues, self.problemWeights)

    def calcGoalMismatch(self):
        """Returns a value of goal state mismatch."""

        if len(self.goalState) == 0:
            return 0

        if self.exactInputMatch:
            self.mismatch = float(np.array_equal(self.goalState, self.goalWeights))
            return self.mismatch

        self.mismatch = rbf(self.goalState, self.goalValues, self.goalWeights)

        return self.mismatch

    def calcLateralActivation(self):
        """Returns a value of activation for the weighted lateral input."""

        if len(self.lateralState) == 0:
            return 0

        return weightedSum(self.lateralState, self.lateralWeights)

    def calcControlActivation(self):
        """Returns a value of activation for the weighted control input."""

        if len(self.controlState) == 0:
            return 0

        return weightedSum(self.controlState, self.controlWeights)

    def calcCore(self, time):
        """Returns a value of current activation of the FS."""

        if self.isActive:
            self.onTime = time - self.startTime

        if self.isActive and self.onTime >= self.tau and \
                not self.wasUsed and not self.isOutput:  # expected time of activation is over
            self.failed = True
            self.wasUsed = True
            self.activity = 0
            self.isActive = False

            if self.calcGoalMismatch() >= self.pr_threshold:  # the goal has been obtained
                self.failed = False
                self.onTime = 0
                # else:
                # # if the goal hasn't been obtained then
                # # correct problem weights to make FS activation more specific
                #     self.problemWeights.update(self.plasticWeights)
                #     print '%#%# learned fs', self.ID
                #     print ' wpr:', self.problemWeights
                #     print ' pl w:', self.plasticWeights
        else:
            wInSum = 0.2*self.oldActivity
            wInSum += 0.8*self.calcProblemActivation()
            wInSum += self.calcLateralActivation()
            wInSum += 0.5*self.calcControlActivation()
            wInSum += (1 - 2 * np.rand()) * self.noise

            if not self.isOutput:
                wInSum -= self.calcGoalMismatch()
            self.activity = sigmoid(wInSum, self.k, self.x0)

            if self.activity >= self.threshold:
                self.isActive = True
            else:
                self.isActive = False

            if self.isActive and self.onTime == 0:  # start of FS activity
                self.startTime = time

            if self.mismatch >= self.pr_threshold:  # the goal has been obtained
                self.failed = False
                self.onTime = 0
                # self.plasticWeights = {}
                # self.mismatch = 0

        return self.activity, self.mismatch

    def update(self, time):  # net is a dictionary {FSID: AtomFS}
        """Updates current state of FS."""

        self.wasActive.pop(0)
        self.wasActive.append(self.isActive)
        # self.oldActivity = self.activity

        return self.calcCore(time)

    def weightsUpdate(self, fsnet):
        """Updates current weights of FS to exclude unimportant connections"""

        for fs in self.problemWeights.keys():
            if not fsnet[fs].isActive:
                if self.problemWeights[fs] > self.rateOfWeightLearning:
                    self.problemWeights[fs] -= self.rateOfWeightLearning
                else:
                    self.problemWeights[fs] = 0

    def setFSActivation(self, outValue):

        self.wasActive.pop(0)
        self.wasActive.append(self.isActive)
        self.oldActivity = outValue
        self.activity = outValue
        self.isActive = True

        return self.activity

    def resetActivity(self):
        """Resets FS activity"""

        self.failed = False
        self.isActive = False
        self.wasActive = [False, False]  # depth of the activation memory is 2
        self.mismatch = 0
        self.onTime = 0
        self.activity = 0
        self.oldActivity = 0

        # end of AtomFS class