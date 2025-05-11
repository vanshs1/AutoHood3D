#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"AutoHood3D: A Multi-Modal Benchmark for Automotive Hood Design and Fluid–Structure Interaction".

This is a demonstration intended to provide a working example with data provided in the repo.
For application on other datasets, the requirement is to configure the settings. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python <this_file.py>" will run the demo code. 
    NOTE - It is important to check README prior to running this code.

Contact: 
    - Vansh Sharma at vanshs@umich.edu
    - Venkat Raman at ramanvr@umich.edu

Affiliation: 
    - APCL Group 
    - Dept. of Aerospace Engg., University of Michigan, Ann Arbor
"""

import numpy as np
import os
import madcad as mc
from madcad import *

#%% USER INPUT
local = os.getcwd()  ## Get local dir
os.chdir(local)      ## shift the work dir to local dir
# print('\nWork Directory: {}'.format(local))

dir_path = 'PATH_TO_SKINS_FROM_Wollstadt_et_al_CarHoods10k'
sknum = 5
geomname = 'skin_%d/geometry_4.stl'%(sknum)

#%%
domain_stl = mc.read(dir_path + geomname)
domain_stl.mergeclose() ## Very important
flatThis = convexhull(domain_stl)
flatThis = flatThis.transform(scale(vec3(0.001, 0.001, 0.001)) ) ### For creating CFD Shells

## trasnform the hood - orthogonal to flow direction
flatThis = flatThis.transform(translate(vec3(-1*flatThis.barycenter()[0], -1*flatThis.barycenter()[1], -1*flatThis.barycenter()[2])) )
flatThis = flatThis.transform(rotatearound(8.4*pi/180, (O, Y)) ) ## determined manually
pointsArr = np.array(list(flatThis.points))
max_point = pointsArr[pointsArr[:, 0:1].argmax()]
print(max_point)

## Add shell thickness to the hood
thicknessShell = 0.01
flatThis = thicken(flatThis, thicknessShell, 0.5, method='face')
mc.write( flatThis, "./Base_Skin/convHull_sk%03d_geo4.stl"%(sknum))




