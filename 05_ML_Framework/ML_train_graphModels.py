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

import os, json, gc
import torch
import os.path as osp
from types import SimpleNamespace
from datetime import datetime
import numpy as np
from models.MLP import MLP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in list(range(torch.cuda.device_count())) )
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("CUDA DEVICE(S):",torch.cuda.device_count(), flush=True)

#%%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
# use_cuda = torch.cuda.is_available()
# device = 'cuda' if use_cuda else 'cpu'
# if use_cuda:
#     print('Using GPU')
# else:
#     # if device=='cpu' and torch.backends.mps.is_available():
#     #     device = torch.device("mps")
#     print('Using', device)
    
# Get current date and time
current_time = datetime.now()

#%% Utility functions
def load_file_splits(json_path):
    """
    Load the file splits from a JSON file.
    Returns a dictionary with training, validation, ID testing, and OOD testing file lists.
    """
    try:
        with open(json_path, "r") as json_file:
            file_splits = json.load(json_file)
        print(f"File splits loaded successfully from {json_path}")
        return file_splits
    except FileNotFoundError:
        print(f"Error: The JSON file '{json_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The JSON file '{json_path}' contains invalid JSON.")
        return None

def get_nb_trainable_params(model):
   '''
   Return the number of trainable parameters
   '''
   model_parameters = filter(lambda p: p.requires_grad, model.parameters())
   return sum([np.prod(p.size()) for p in model_parameters])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64) ):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
    
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().data.numpy()
    return x

# ===== Modified CharDataset =====
class CharDataset(Dataset):
    def __init__(self, array, device=None):
        """
        Args:
            array (list): List of file paths.
            device (torch.device or None): If provided, data is moved to this device on load.
        """
        self.data = array
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataset = torch.load(self.data[idx], map_location=self.device)
        if self.device is not None:
            if isinstance(dataset, list):
                dataset = [d.to(self.device) for d in dataset]
            else:
                dataset = dataset.to(self.device)
        return dataset

# ===== Modified construct_dataloaders =====
def construct_dataloaders(dataStore_path, dynamic_loading=True, shuffle=False, train_device=None, val_device=None):
    
    get_fileNames = os.listdir(dataStore_path)
    val_files = [os.path.join(dataStore_path, f) for f in get_fileNames if 'val_' in f]
    train_files = [os.path.join(dataStore_path, f) for f in get_fileNames if 'val_' not in f]
    
    if dynamic_loading:
        print('Dynamic loading activated...', flush = True)
        Predataset_val = CharDataset(val_files, device=val_device)  # Validation data remains on CPU (device=None)
        Predataset_train = CharDataset(train_files, device=train_device)  # Training data loaded on device (e.g. GPU)
      
        train_loader = DataLoader(
            Predataset_train,
            batch_size=1,
            pin_memory=False, ## data is already on GPU
            shuffle=shuffle,
            collate_fn=lambda x: x[0],
            sampler=DistributedSampler(Predataset_train)
        )
      
        val_loader = DataLoader(
            Predataset_val,
            batch_size=1,
            pin_memory=False,
            shuffle=shuffle,
            collate_fn=lambda x: x[0],
            sampler=DistributedSampler(Predataset_val)
        )
    else:       
        datasetV = [torch.load(d, map_location=val_device) for d in val_files]
        val_loader = DataLoader(datasetV, batch_size = 1, shuffle=shuffle, 
                                num_workers=0, pin_memory=False, persistent_workers=False)
        
        datasetT = [torch.load(d, map_location=train_device) for d in train_files]
        train_loader = DataLoader(datasetT, batch_size = 1, shuffle=shuffle, 
                                  num_workers=0, pin_memory=False, persistent_workers=False)
        
        
        
    return train_loader, val_loader

#%% ML functions
def createModels(args):
    restart_training = False
    PATH_TO_MODEL = "PATH_FOR_RESTART" 
    for i in range(args.nmodel):
        
        if args.model[i] == 'GraphSAGE':
            hparams = {
                "encoder": [10, 128, 128, 64],
                "decoder": [64, 128, 128, 7],
                "nb_hidden_layers": 4,
                "size_hidden_layers": 64,
                "batch_size": 128,
                "nb_epochs": 752,
                "lr": 1e-04,
                "max_neighbors": 4,
                "bn_bool": True,
                "r": 0.05
            }

        elif args.model[i] == 'GUNet':
            hparams = {
                "encoder": [10, 128, 128, 64],
                "decoder": [64, 128, 128, 7],
                "layer": "SAGE",
                "pool": "random",
                "nb_scale": 5,
                "pool_ratio": [0.5, 0.5, 0.5, 0.5],
                "list_r": [0.05, 0.2, 0.5, 1, 10],
                "size_hidden_layers": 64,
                "batchnorm": True,
                "res": False,
                "batch_size": 128,
                "nb_epochs": 752,
                "lr": 1e-4,
                "max_neighbors": 4,
                "r": 0.05
            }
        
        elif args.model[i] == 'PointGNNCon':
            hparams = {
                "encoder": [10, 128, 128, 64],
                "decoder": [64, 128, 128, 7],
                "nb_hidden_layers": 4,
                "size_hidden_layers": 64,
                "batch_size": 128,
                "nb_epochs": 752,
                "lr": 1e-4,
                "max_neighbors": 4,
                "bn_bool": True,
                "r": 0.05
            }
        
        encoder = MLP(hparams['encoder'], batch_norm = False)
        decoder = MLP(hparams['decoder'], batch_norm = False)

        if args.model[i] == 'GraphSAGE':
            from models.GraphSAGE import GraphSAGE
            model = GraphSAGE(hparams, encoder, decoder)
            if restart_training:
                state_dict = torch.load(PATH_TO_MODEL, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"{args.model[i]} model reloaded for training.")

        elif args.model[i] == 'GUNet':
            from models.GUNet import GUNet
            model = GUNet(hparams, encoder, decoder)
            if restart_training:
                state_dict = torch.load(PATH_TO_MODEL, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"{args.model[i]} model reloaded for training.")
        
        elif args.model[i] == 'PointGNNCon':
            from models.PointGNNCon import PointGNNCon
            model = PointGNNCon(hparams, encoder, decoder)
            if restart_training:
                state_dict = torch.load(PATH_TO_MODEL, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"{args.model[i]} model reloaded for training.")
        
    return model, hparams


@torch.no_grad()
def test(device, model, test_loader, criterion = 'MSE'):
    model.train(False)
    model.eval()
    
    avg_loss_per_var = np.zeros(7)
    avg_loss = 0
    avg_loss_surf_var = np.zeros(3)
    avg_loss_surf = 0
    
    for dataF in test_loader:
        counter = 0
        for data_clone in dataF:
            data_clone = data_clone.to(device)
            out = model(data_clone)       
    
            targets = data_clone.y
            if criterion == 'MSE' or 'MSE_weighted':
                loss_criterion = nn.MSELoss(reduction = 'none')
            elif criterion == 'MAE':
                loss_criterion = nn.L1Loss(reduction = 'none')
    
            loss_per_var = loss_criterion(out, targets).mean(dim = 0)
            loss = loss_per_var.mean()
            loss_surf_var = loss_criterion(out[:,0:3], targets[:,0:3] ).mean(dim = 0) ## loss on selected channels
            
            loss_surf = loss_surf_var.mean()    
            avg_loss_per_var += loss_per_var.cpu().numpy()
            avg_loss += loss.cpu().numpy()
            avg_loss_surf_var += loss_surf_var.cpu().numpy()
            avg_loss_surf += loss_surf.cpu().numpy()
            counter += 1
        
        avg_loss=avg_loss/counter
        avg_loss_per_var=avg_loss_per_var/counter
        avg_loss_surf_var=avg_loss_surf_var/counter
        avg_loss_surf=avg_loss_surf/counter

    return avg_loss, \
            avg_loss_per_var, \
            avg_loss_surf_var, \
            avg_loss_surf,
 


def train(device, model, train_loader, val_loader, optimizer, scheduler, 
          log_path, hparams, base_lr, warmup_epochs, criterion='MSE', epochs=251):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    pbar_train = tqdm(range(epochs), desc="Training Epochs", position=0, leave=True)

    for epoch in pbar_train:
        model.train(True)
        avg_loss_per_var = torch.zeros(4, device = device)
        avg_loss = 0.0
        avg_loss_surf_var = torch.zeros(3, device = device) 
        avg_loss_surf = 0.0

        start_time = datetime.now()
        
        for dataF in train_loader:
            counter = 0
            databar_train = tqdm(dataF, desc="Data Loading", position=1, leave=False)
            for data_clone in databar_train:
                # data_clone = data_clone.to(device)
		        # Training data is already on device, so we don't call .to(device) here.     

                out = model(data_clone)
                targets = data_clone.y
        
                if criterion == 'MSE' or criterion == 'MSE_weighted':
                    loss_criterion = nn.MSELoss(reduction = 'none')
                elif criterion == 'MAE':
                    loss_criterion = nn.L1Loss(reduction = 'none')
                loss_per_var = loss_criterion(out, targets).mean(dim = 0)
                total_loss = loss_per_var.mean()
    
        
                loss_surf_var = loss_criterion(out, targets).mean(dim = 0)
                loss_surf = loss_surf_var.mean()
                 
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                avg_loss_per_var=to_numpy(avg_loss_per_var)
                avg_loss_surf_var=to_numpy(avg_loss_surf_var)
                

                avg_loss_per_var = np.concatenate(
                    (avg_loss_per_var, loss_per_var.cpu().data.numpy()), axis=0)
                avg_loss += total_loss
                avg_loss_surf_var = np.concatenate((avg_loss_surf_var, loss_surf_var.cpu().data.numpy()), axis=0)
                avg_loss_surf += loss_surf
                counter += 1

            
            avg_loss = avg_loss.cpu().data.numpy()/counter
            avg_loss_per_var = avg_loss_per_var/counter
            avg_loss_surf_var = avg_loss_surf_var/counter
            avg_loss_surf = avg_loss_surf.cpu().data.numpy()/counter
        
        val_loss, val_loss_per_var, val_surf_var, val_surf  = test(device, model, val_loader, criterion = 'MSE')
        
        end_time = datetime.now()
        total_time = torch.tensor((end_time-start_time).seconds).cuda()
        
        ### Warmup the LR 
        if epoch < warmup_epochs:
            warmup_lr = base_lr*(epoch+1)/warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step(val_loss) # Update learning rate after each epoch
        current_lr = optimizer.param_groups[0]['lr']
        # Only machine rank==0 saves the model and prints the metrics    
        if local_rank == 0:
            # print("epoch num -", epoch+1, flush=True)
            if (epoch+1) % 10 == 0:
                torch.cuda.synchronize(device)
                torch.save(model.module.state_dict(), osp.join(log_path, 'model_%04d.pt'%(epoch+1) ))
                print('Model saved.', flush=True)
                # dist.barrier()
                
                with open(osp.join(log_path, 'model_run_'+str(epoch+1) + '_log.json'), 'a') as f:
                    json.dump(
                        {
                            'regression': 'Total',
                            'loss': 'MSE',
                            'hparams': hparams,
                            'current_lr': current_lr,
                            'Average train loss':avg_loss,
                            'Average train loss surf': avg_loss_surf,
                            'Val loss': val_loss,
                            'Val loss per var': val_loss_per_var,
                        }, f, indent=12, cls=NumpyEncoder
                    )
                print('Model and Logs saved.', flush=True)
                
            print(f"\n(Epoch {epoch+1}/{epochs}) Time: {total_time}s")
            print(f"(Epoch {epoch+1}/{epochs}) Average train loss: {avg_loss}, Average train loss surf: {avg_loss_surf}")
            print(f"(Epoch {epoch+1}/{epochs}) Val loss: {val_loss}, Val loss per var: {val_loss_per_var}")  

#%% DDP functions
def init_distributed():
    
  dist_url = "env://"
  
  rank = int(os.environ["RANK"]) 
  # world_size = int(os.environ['WORLD_SIZE']) 
  # local_rank = int(os.environ['LOCAL_RANK']) 
  torch.cuda.set_device(rank)
  dist.init_process_group(backend="nccl", #"nccl" for using GPUs, "gloo" for using CPUs
                          init_method=dist_url,)

def cleanup():
  print("Cleaning up the distributed environment...")
  dist.destroy_process_group()
  print("Distributed environment has been properly closed")


def main():
    models_select = [ 'GraphSAGE', ] ## extendable to train multiple models in sequential 
    args = SimpleNamespace()
    args.model=[]
    args.task=[]
    for i in range(len(models_select)):
        args.model.append( models_select[i] )  ## The model you want to train, choose between GraphSAGE, GUNet and etc.
        args.task.append( models_select[i]+current_time.strftime("_date_%Y_%m_%d_time_%H_%M_%S")+"_run01" )  ## Scenario/ML Model iteration name  
    args.nmodel = len(models_select)
    args.weight = 1.0           ## Weight in front of the surface loss (default: 1), type=float
    name_mod = models_select[0] ## only using the first and ONE model
    
    log_path = osp.join('OUTPUT_FOLDER', args.task[0], args.model[0] ) # path where you want to save model and logs
    dataStore_path = "dataset_forGraphs_from_03b_script/training/"  ## Note training folder includes val_batch_xx dataset too :: see README
    
    ## Save model
    Path(log_path).mkdir(parents = True, exist_ok = True)
    local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = torch.device("cuda", local_rank)
    
    model, hparams = createModels(args=args)
    model = model.to(DEVICE)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    weight_decay = 0
    warmup_epochs = 10 #int(0.03*hparams['nb_epochs']) ### 3% of total epochs are warmup epochs
    base_lr = hparams['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'], weight_decay = weight_decay)
    
    params_model = get_nb_trainable_params(model).astype('float')
    print('\n\nNumber of parameters:', params_model, flush=True)
    print("hparams:: ", hparams, flush=True)
    print("weight decay:: ", weight_decay, flush=True)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=9, min_lr=1e-7, verbose=True)

    # Construct dataloaders.
    # For training, load data on to DEVICE; for validation, keep it on CPU.
    train_loader, val_loader = construct_dataloaders(dataStore_path=dataStore_path, 
                                                     dynamic_loading=True, shuffle=False,
                                                     train_device=DEVICE, val_device=DEVICE)

    train(DEVICE, model, train_loader, val_loader, optimizer, lr_scheduler, 
          log_path, hparams, base_lr, warmup_epochs, 
          criterion='MSE', epochs=hparams["nb_epochs"])
    
    if local_rank == 0:
        cleanup()
        params_model = get_nb_trainable_params(model).astype('float')
        print('Number of parameters:', params_model)
        torch.save(model.module.state_dict(), osp.join(log_path, 'model'))
         
        with open(osp.join(log_path, name_mod + '_log.json'), 'a') as f:
            json.dump(
                {
                    'regression': 'Total',
                    'loss': 'MSE',
                    'nb_parameters': params_model,
                    'hparams': hparams,
                }, f, indent = 12, cls = NumpyEncoder
            )
        print('Model and Logs saved.')
    

#%%
if __name__ == '__main__':

    init_distributed()
    gc.collect()
    main()

















