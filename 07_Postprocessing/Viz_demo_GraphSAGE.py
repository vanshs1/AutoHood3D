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

import torch
import numpy as np
import pyvista as pv
import torch_geometric.nn as nng
import torch.nn as nn
import time, os


use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')
#%% Settings

## For GraphSAGE
hparams = {
        "encoder": [10, 128, 128, 64],
        "decoder": [64, 128, 128, 7],
        "nb_hidden_layers": 4,
        "size_hidden_layers": 64,
        "batch_size": 128,
        "nb_epochs": 352,
        "lr": 1e-06,
        "max_neighbors": 4,
        "bn_bool": True,
        "r": 0.05
    }


model_dir ='output_model/GSAGE/'
modelname = "GraphSAGE.pt"

PATH_TO_TEST_DATA = "./dataset_graph/id_testing/batch_03.pt"
PATH_TO_SAVED_DATA = "OUTPUT_FOLDER"

#%%
from models.MLP import MLP
encoder = MLP(hparams['encoder'], batch_norm = False)
decoder = MLP(hparams['decoder'], batch_norm = False)
from models.GraphSAGE import GraphSAGE
model = GraphSAGE(hparams, encoder, decoder)


model_dir=model_dir+modelname
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(state_dict)

dirName_forOutput = modelname.split('.')[0]
PATH_TO_SAVED_DATA = PATH_TO_SAVED_DATA+dirName_forOutput

if not os.path.exists(PATH_TO_SAVED_DATA):
    os.makedirs(PATH_TO_SAVED_DATA)
    print(f"Directory '{dirName_forOutput}' created.")
else:
    print(f"Error: Directory '{PATH_TO_SAVED_DATA}' already exists.")
    # sys.exit(1)

if PATH_TO_SAVED_DATA is not None:
    val_dataset = torch.load(PATH_TO_TEST_DATA)

model.eval()

counter=1
for data in val_dataset:        
    data_clone = data.clone()
    
    ### Only used for GraphSAGE
    data_clone.edge_index = nng.radius_graph(x = data_clone.pos.to(device), 
                                              r = hparams['r'], loop = True, 
                                              max_num_neighbors = int(hparams['max_neighbors'])).cpu()
    
    data_clone = data_clone.to(device)
    out = model(data_clone)       

    targets = data_clone.y

    data_clone = data_clone.x.detach().numpy()
    out = out.detach().numpy()
    targets = targets.detach().numpy()
    points = data_clone[:, 0:3]
    counter+=1
    for field in ['mock', 'p', 'U',  'D']:
        
        if field == 'mock':
            pdata_trg = pv.vector_poly_data(points, targets[:,0:3])
            pdata_pred = pv.vector_poly_data(points, out[:,0:3])
            
            p = pv.Plotter(shape=(2, 2), border=False, off_screen=True)

            p.subplot(0, 0)
            p.add_mesh(pdata_trg, cmap='gist_rainbow', scalar_bar_args={'title': '%s-trg'%(field) }, )
            p.add_text("Target", font_size=24)
            p.camera_position = 'yx'
            
            p.subplot(1, 0)
            p.add_mesh(pdata_trg,  cmap='gist_rainbow', scalar_bar_args={'title': '%s-trg'%(field) }, show_edges=False,)
            p.add_text("Target", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90

            p.subplot(0, 1)
            actor2 = p.add_mesh(pdata_pred, cmap='gist_rainbow', scalar_bar_args={'title': '%s-pred'%(field) },)
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'yx'

            p.subplot(1, 1)
            actor3 = p.add_mesh(pdata_pred,  cmap='gist_rainbow', scalar_bar_args={'title': '%s-pred'%(field) }, show_edges=False, )
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            
            fname = f'out_idTest_{field}_%02d.png'%( counter)
            outPath_pics = os.path.join(PATH_TO_SAVED_DATA, fname)
            p.screenshot(filename = outPath_pics,
                          scale = 1
                      )
            p.close()
            time.sleep(1)
         
        if field == 'U':
            pdata_trg = pv.vector_poly_data(points, targets[:,0:3])
            pdata_pred = pv.vector_poly_data(points, out[:,0:3])
            
            
            p = pv.Plotter(shape=(2, 2), border=False, off_screen=True)
            p.subplot(0, 0)
            p.add_mesh(pdata_trg, cmap='gist_rainbow', scalar_bar_args={'title': '%s-trg'%(field) }, )
            p.add_text("Target", font_size=24)
            p.camera_position = 'yx'
            
            p.subplot(1, 0)
            p.add_mesh(pdata_trg,   cmap='gist_rainbow', scalar_bar_args={'title': '%s-trg'%(field) }, show_edges=False,)
            p.add_text("Target", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90

            p.subplot(0, 1)
            actor2 = p.add_mesh(pdata_pred,cmap='gist_rainbow', scalar_bar_args={'title': '%s-pred'%(field) },)
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'yx'

            
            p.subplot(1, 1)
            actor3 = p.add_mesh(pdata_pred, cmap='gist_rainbow', scalar_bar_args={'title': '%s-pred'%(field) }, show_edges=False, )
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            
            fname = f'out_idTest_{field}_%02d.png'%( counter)
            outPath_pics = os.path.join(PATH_TO_SAVED_DATA, fname)
            p.screenshot(filename = outPath_pics,
                          scale = 2
                      )
            p.close()
            time.sleep(5)
            
        elif field == 'p':
            vectorsT = np.hstack([targets[:, 3:4], np.zeros((targets.shape[0], 2))])
            vectorsP = np.hstack([out[:, 3:4], np.zeros((out.shape[0], 2))])
            
            pdata_trg = pv.vector_poly_data(points, vectorsT)
            pdata_pred = pv.vector_poly_data(points, vectorsP)
            
            p = pv.Plotter(shape=(2, 2), border=False, off_screen=True)
            p.subplot(0, 0)
            p.add_mesh(pdata_trg,  cmap='jet', scalar_bar_args={'title': '%s-trg'%(field) }, )
            p.add_text("Target", font_size=24)
            p.camera_position = 'yx'
            
            p.subplot(1, 0)
            p.add_mesh(pdata_trg,   cmap='jet', scalar_bar_args={'title': '%s-trg'%(field) }, show_edges=False,)
            p.add_text("Target", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            

            p.subplot(0, 1)
            actor2 = p.add_mesh(pdata_pred, cmap='jet', scalar_bar_args={'title': '%s-pred'%(field) },)
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'yx'

            p.subplot(1, 1)
            actor3 = p.add_mesh(pdata_pred, cmap='jet', scalar_bar_args={'title': '%s-pred'%(field) }, show_edges=False, )
            # p.enable_eye_dome_lighting()
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            
            fname = f'out_idTest_{field}_%02d.png'%( counter)
            outPath_pics = os.path.join(PATH_TO_SAVED_DATA, fname)
            p.screenshot(filename = outPath_pics,
                          scale = 2
                      )
            p.close()
            time.sleep(1)
            
            
        elif field == 'D':
            
            pdata_trg = pv.vector_poly_data(points, targets[:,4:])
            pdata_pred = pv.vector_poly_data(points, out[:,4:])
        
            p = pv.Plotter(shape=(2, 2), border=False, off_screen=True)
            p.subplot(0, 0)
            p.add_mesh(pdata_trg, cmap='gnuplot2', scalar_bar_args={'title': '%s-trg'%(field) }, )
            p.add_text("Target", font_size=24)
            p.camera_position = 'yx'
            
            p.subplot(1, 0)
            p.add_mesh(pdata_trg, cmap='gnuplot2', scalar_bar_args={'title': '%s-trg'%(field) }, show_edges=False,)
            p.add_text("Target", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            
            p.subplot(0, 1)
            actor2 = p.add_mesh(pdata_pred, cmap='gnuplot2', scalar_bar_args={'title': '%s-pred'%(field) },)
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'yx'
            
            p.subplot(1, 1)
            actor3 = p.add_mesh(pdata_pred, cmap='gnuplot2', scalar_bar_args={'title': '%s-pred'%(field) }, show_edges=False, )
            p.add_text("Prediction", font_size=24)
            p.camera_position = 'xy'
            p.camera.roll += 90
            
            fname = f'out_idTest_{field}_%02d.png'%(counter)
            outPath_pics = os.path.join(PATH_TO_SAVED_DATA, fname)
            p.screenshot(filename = outPath_pics,
                          scale = 2
                      )
            p.close()
            time.sleep(1)
            
    time.sleep(1)


