# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:18:53 2026

@author: Facundo
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from src.utils import find_spatial_shift_subpixel
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind

#%%
def compute_rate_maps_from_sparse(sparse_spikes, traj, selected_neurons, L, dt, nx=30, ny=30):
    """
    Computes spatial firing rate maps for a subset of neurons efficiently.
    """
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    
    # Map continuous trajectory to 1D spatial bin indices
    x_idx = np.clip(np.floor((x_pos / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((y_pos / L) * ny).astype(int), 0, ny - 1)
    spatial_idx = y_idx * nx + x_idx  # Shape: (steps,)
    
    # Compute Occupancy Map (time spent in each bin in seconds)
    occupancy_1d = np.bincount(spatial_idx, minlength=nx * ny) * dt
    safe_occupancy = np.where(occupancy_1d > 0, occupancy_1d, 1.0)
    
    # Filter sparse spikes to only include the selected neurons
    selected_neurons = np.asarray(selected_neurons)
    mask = np.isin(sparse_spikes['neuron_idx'], selected_neurons)
    
    filt_time_idx = sparse_spikes['time_idx'][mask]
    filt_neuron_idx = sparse_spikes['neuron_idx'][mask]
    filt_counts = sparse_spikes['counts'][mask]
    
    # Get the 1D spatial bin index for every single spike
    spike_spatial_idx = spatial_idx[filt_time_idx]
    
    # Generate rate maps
    num_selected = len(selected_neurons)
    rate_maps = np.zeros((num_selected, ny, nx))
    
    for i, neuron_id in enumerate(selected_neurons):
        n_mask = (filt_neuron_idx == neuron_id)
        n_spatial_idx = spike_spatial_idx[n_mask]
        n_counts = filt_counts[n_mask]
        
        # Sum spikes in each spatial bin using counts as weights
        spike_map_1d = np.bincount(n_spatial_idx, weights=n_counts, minlength=nx * ny)
        
        # Divide by occupancy and reshape to 2D
        rate_map_2d = (spike_map_1d / safe_occupancy).reshape(ny, nx)
        
        # Optional: Set unvisited bins to 0 
        rate_map_2d[occupancy_1d.reshape(ny, nx) == 0] = 0 
        
        rate_maps[i] = rate_map_2d
        
    return rate_maps


def load_and_compute_maps_from_sparse(num, base_dir=None, nx=30, ny=30):
    """
    Scans the Simulation directory, loads the trajectory and sparse spikes, 
    and computes the firing rate maps on the fly.
    """
    if base_dir is None:
        base_dir = os.path.split(os.getcwd())[0]
        
    sim_folder = os.path.join(base_dir, f'SimulationSpiking-num{num}')
    
    if not os.path.exists(sim_folder):
        raise FileNotFoundError(f"Simulation directory not found: {sim_folder}")
        
    loaded_maps = {}
    loaded_maps_conj = {}
    
    traj_list = []
    for fldr_idx, folder_name in enumerate(os.listdir(sim_folder)):
        if not folder_name.startswith('incl_ang'):
            continue
            
        folder_path = os.path.join(sim_folder, folder_name)
        
        # We use parameters_complete to ensure we have dt
        param_path = os.path.join(folder_path, 'parameters_complete.pkl')
        traj_path = os.path.join(folder_path, 'traj.npy')
        omni_spikes_path = os.path.join(folder_path, 'sparse_spikes_omni.pkl')
        conj_spikes_path = os.path.join(folder_path, 'sparse_spikes_conj.pkl')
        
        if not os.path.exists(param_path) or not os.path.exists(traj_path):
            print(f"Skipping {folder_name}: Missing files.")
            continue
            
        with open(param_path, 'rb') as file:
            params = pickle.load(file)
            
        traj = np.load(traj_path)
        traj_list.append(traj)
        L = params['L']
        dt = params['dt']
        incl_idx = params.get('inclination_angle')
        
        with open(omni_spikes_path, 'rb') as file:
            omni_spikes = pickle.load(file)
        with open(conj_spikes_path, 'rb') as file:
            conj_spikes = pickle.load(file)

        # Find total number of neurons by taking the max index + 1
        N_omni = np.max(omni_spikes['neuron_idx']) + 1 if len(omni_spikes['neuron_idx']) > 0 else 0
        N_conj = np.max(conj_spikes['neuron_idx']) + 1 if len(conj_spikes['neuron_idx']) > 0 else 0

        # Compute maps for all active neurons
        print(f"Computing maps for {folder_name}...")
        rate_map_omni = compute_rate_maps_from_sparse(
            omni_spikes, traj, np.arange(N_omni), L, dt, nx=nx, ny=ny)
        rate_map_conj = compute_rate_maps_from_sparse(
            conj_spikes, traj, np.arange(N_conj), L, dt, nx=nx, ny=ny)
        
        loaded_maps[fldr_idx] = {}
        loaded_maps[fldr_idx]['spiking maps'] = {}
        loaded_maps_conj[fldr_idx] = {}
        loaded_maps_conj[fldr_idx]['spiking maps'] = {}
        
        loaded_maps[fldr_idx]['rate maps'] = rate_map_omni
        loaded_maps[fldr_idx]['spiking maps']['neuron_idx'] = omni_spikes['neuron_idx']
        loaded_maps[fldr_idx]['spiking maps']['time_idx'] = omni_spikes['time_idx']
        
        loaded_maps_conj[fldr_idx]['rate maps'] = rate_map_conj
        loaded_maps_conj[fldr_idx]['spiking maps']['neuron_idx'] = conj_spikes['neuron_idx']
        loaded_maps_conj[fldr_idx]['spiking maps']['time_idx'] = conj_spikes['time_idx']
        
        print(f"Finished {folder_name}. Shapes: Omni {rate_map_omni.shape}, Conj {rate_map_conj.shape}")
            
    return loaded_maps, loaded_maps_conj, traj_list


def compute_cross_corrs(fr_maps0,fr_maps1,sd=2,smooth = False):
    CrossCorr = np.zeros((fr_maps0.shape[0],fr_maps0.shape[1]*2-1,fr_maps0.shape[1]*2-1))
    
    for ind in tqdm(range(CrossCorr.shape[0]), desc="Cross Correlograms"):
        rm1 = (fr_maps0[ind] - fr_maps0[ind].mean())/fr_maps0[ind].std()
        rm2 = (fr_maps1[ind] - fr_maps1[ind].mean())/fr_maps1[ind].std()
        if smooth:
            rm1 = gaussian_filter(rm1, (sd,sd),mode='constant', cval=0)
            rm2 = gaussian_filter(rm2, (sd,sd), mode='constant', cval=0)
        CrossCorr[ind] = correlate2d(rm1, 
                                     rm2,
                                     mode='full', boundary='fill', fillvalue=0)
        
    shifts = np.zeros((fr_maps0.shape[0],2))
    for i in tqdm(range(fr_maps0.shape[0]), desc="Spatial Shift"):
        sx, sy = find_spatial_shift_subpixel(CrossCorr[i], n = 3, search_radius_pixels = 7)#[:,::-1].T)
        shifts[i,0], shifts[i,1] = sx, sy
    
    return shifts, CrossCorr

#%%
if __name__ == "__main__":
    num = 2
    
    SMOOTH = False
    NEURON_IDX = 54
    
    path2load = os.path.split(os.getcwd())[0]    
    path2load = os.path.join(path2load,f'SimulationSpiking-num{num}')

    # Load parameters to get L and other info for plotting
    path2load_0 = os.path.join(path2load,f'incl_ang{0}')
    with open(os.path.join(path2load_0,'parameters_reduced.pkl'),'rb') as file:
        parameters = pickle.load(file)
                
    L = parameters['L']
    
    # Compute maps from sparse data (shape is already N, ny, nx)
    omni_maps, conj_maps, traj_list = load_and_compute_maps_from_sparse(num=num, nx=30, ny=30)
    
    nfr = omni_maps[0]['rate maps'].shape[1]
    N = omni_maps[0]['rate maps'].shape[0]
    N_conj = conj_maps[0]['rate maps'].shape[0]
    
    shifts_x = []
    shifts_y = []
    
    shifts_conj_x = []
    shifts_conj_y = []
    sd = 1.5
    plt.figure(40,figsize=(18,24))
    
    i_0 = 22
    i_f = 37
    
    angles = [r'$\pi/6$',r'$\pi/3$','0\'']
    
    for i in range(1,4):
        # Arrays are already (N, ny, nx), no reshaping needed
        plt.figure(1,figsize=(15,20))
        shifts, CrossCorr = compute_cross_corrs(
            omni_maps[0]['rate maps'],
            omni_maps[i]['rate maps'],
            smooth=SMOOTH,sd=sd)
        shifts_y.append((shifts[:,0]) * L / nfr)
        shifts_x.append(shifts[:,1] * L / nfr)    
        
        plt.subplot(4,3,i)
        plt.imshow(CrossCorr.mean(axis=0)[::-1,:],cmap='jet')
        plt.axis('off')
        
        plt.subplot(4,3,i+3)
        plt.imshow(CrossCorr.mean(axis=0)[::-1,:][i_0:i_f,i_0:i_f],cmap='jet')
        plt.axis('off')
        
        shifts, CrossCorr = compute_cross_corrs(
            conj_maps[0]['rate maps'],
            conj_maps[i]['rate maps'],
            smooth=SMOOTH,sd=sd)
        shifts_conj_y.append((shifts[:,0]) * L / nfr)
        shifts_conj_x.append(shifts[:,1] * L / nfr)    
        
        plt.subplot(4,3,i+6)
        idx = np.where(omni_maps[i]['spiking maps']['neuron_idx'] == NEURON_IDX)[0]
        idx = omni_maps[i]['spiking maps']['time_idx'][idx]
        plt.plot(traj_list[i][:,0],traj_list[i][:,1])
        plt.plot(traj_list[i][idx,0],traj_list[i][idx,1],'.',
                 color='r',markersize=10)
        plt.axis('off')
        plt.subplot(4,3,i+9)
        plt.imshow(gaussian_filter(omni_maps[i]['rate maps'][NEURON_IDX],
                                   (sd,sd),mode='constant', cval=0)[::-1,:],cmap='jet')
        plt.axis('off')
        
        # plt.figure(2,figsize=(5,5))
        # plt.subplot(3,3,i)
        # plt.imshow(CrossCorr.mean(axis=0)[::-1,:],cmap='jet')
        # plt.axis('off')
        
        # plt.subplot(3,3,i+3)
        # plt.imshow(CrossCorr.mean(axis=0)[::-1,:][i_0:i_f,i_0:i_f],cmap='jet')
        # plt.axis('off')
        
        # plt.subplot(3,3,i+6)
        # idx = np.where(conj_maps[i]['spiking maps']['neuron_idx'] == NEURON_IDX)[0]
        # idx = conj_maps[i]['spiking maps']['time_idx'][idx]
        # plt.plot(traj[:,0],traj[:,1])
        # plt.plot(traj[idx,0],traj[idx,1],'.')
        # plt.axis('off')
        
        
    plt.show()
        
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    sns.boxplot(shifts_x,fill=False)
    sns.stripplot(shifts_x)
    plt.title('Omni x shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.subplot(222)
    sns.boxplot(shifts_y,fill=False)
    sns.stripplot(shifts_y)
    plt.title('Omni y shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.subplot(223)
    sns.boxplot(shifts_conj_x,fill=False)
    sns.stripplot(shifts_conj_x)
    plt.title('Conj x shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.subplot(224)
    sns.boxplot(shifts_conj_y,fill=False)
    sns.stripplot(shifts_conj_y)
    plt.title('Conj y shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.tight_layout()
    plt.show()
    
    # plt.figure(figsize=(10,10))
    # idx = [1,2,3]
    
    # # Arrays are already 2D (ny, nx), no reshaping needed
    # rm0 = gaussian_filter(omni_maps[0][NEURON_IDX], (1,1),
    #                       mode='constant', cval=0)
    
    # for i in range(3):
    #     rm1 = omni_maps[idx[i]][NEURON_IDX]
    #     rm1_conj = conj_maps[idx[i]][NEURON_IDX]
        
    #     if SMOOTH:
    #         rm1 = gaussian_filter(rm1,(sd,sd),
    #                               mode='constant', cval=0)[:,:]
    #         rm1_conj = gaussian_filter(rm1_conj,(sd,sd),
    #                                    mode='constant', cval=0)[:,:]
                                       
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(correlate2d((rm0-rm0.mean())/np.std(rm0),
    #                            (rm1-rm1.mean())/np.std(rm1))[::-1,:], cmap='jet')
    #     plt.axis('off')
    #     plt.title(f'Angle {angles[i]}',fontsize=15)
        
    #     plt.subplot(3,3,3+i+1)
    #     plt.imshow(rm1,cmap='jet')
    #     plt.title(f'Omni Mean fr {omni_maps[idx[i]][0].mean():.4f}',
    #               fontsize=15)
    #     plt.axis('off')
        
    #     plt.subplot(3,3,6+i+1)
    #     plt.imshow(rm1_conj,cmap='jet')
    #     plt.title(f'Conj Mean fr {conj_maps[idx[i]][0].mean():.4f}',
    #               fontsize=15)
    #     plt.axis('off')
        
    with open(os.path.join(path2load_0,'parameters_complete.pkl'),'rb') as file:
        parameters_complete = pickle.load(file)

    print(f"d_asym = {parameters_complete['d_asym']:.2f}")
    print(f"l_torus = {parameters_complete['l_torus']:.2f}")
    print(f"input_std = {parameters_complete['input_std']:.2f}")
    print(f"m = {parameters_complete['m']:.2f}")
    if 'thresh' in parameters_complete:
        print(f"thresh = {parameters_complete['thresh']}")