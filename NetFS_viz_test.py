# -*- coding: utf-8 -*-
"""
Created on Sat Sep 07 01:55:30 2013

@author: Burtsev
"""

import FSNpy as FSN
#import matplotlib.pyplot as plt

# weights to FS's inputs
pw = [[0, 2, 1.], [1, 2, 0.]]
gw = [[0, 2, 0.], [1, 2, 1.]]
# inputs
input1 = FSN.AtomFS()
input2 = FSN.AtomFS()
# FS which is tested
FS_2 = FSN.AtomFS()
FS_2.set_params({}, {}, 15, 0.5, 0)
FSNet = FSN.FSNetwork()
FSNet.add(input1)
FSNet.add(input2)
FSNet.add(FS_2)
FSNet.addActionLinks(pw)
FSNet.addPredictionLinks(gw)
FS_3 = FSNet.duplicate(FS_2.ID)
FS_4 = FSNet.duplicate(FS_2.ID)
FS_2.activationWeights[FS_3.ID] = 0.5
FS_5 = FSNet.duplicate(FS_3.ID)
FS_6 = FSNet.duplicate(FS_2.ID)
FS_7 = FSNet.duplicate(FS_3.ID)

FSNet.drawNet()
