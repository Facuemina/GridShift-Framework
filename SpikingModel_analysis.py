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
import pandas as pd
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


def load_and_compute_maps_from_sparse(num, sim_folder, nx=30, ny=30):
    """
    Scans the Simulation directory, loads the trajectory and sparse spikes, 
    and computes the firing rate maps on the fly.
    """
    # if base_dir is None:
    #     base_dir = os.path.split(os.getcwd())[0]
        
    # sim_folder = os.path.join(base_dir, f'SimulationSpiking-num{num}')
    
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
    # Compute displacement of fr_maps1 with respect to fr_maps0
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
        sy, sx = find_spatial_shift_subpixel(CrossCorr[i], n = 7, search_radius_pixels = 7)
        shifts[i,0], shifts[i,1] = sx, sy
    
    return shifts, CrossCorr

#%% SPATIAL SHIFTS BY HEAD DIRECTION (UP vs DOWN)

def compute_directional_rate_maps(spiking_maps, traj, is_cond, num_neurons, L, nx=30, ny=30):
    """Computes rate maps filtering by a specific trajectory condition (e.g., HD range)."""
    x_idx = np.clip(np.floor((traj[:, 0] / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((traj[:, 1] / L) * ny).astype(int), 0, ny - 1)
    spatial_idx = y_idx * nx + x_idx

    # Occupancy for the specific condition
    occ = np.bincount(spatial_idx[is_cond], minlength=nx * ny)
    safe_occ = np.where(occ > 0, occ, 1.0)

    t_idx = spiking_maps['time_idx']
    n_idx = spiking_maps['neuron_idx']

    # Filter spikes that occurred during the condition
    valid_spikes_mask = is_cond[t_idx]
    t_idx_cond = t_idx[valid_spikes_mask]
    n_idx_cond = n_idx[valid_spikes_mask]
    spike_spatial_idx = spatial_idx[t_idx_cond]

    maps = np.zeros((num_neurons, ny, nx))
    for neuron_id in range(num_neurons):
        mask = (n_idx_cond == neuron_id)
        spike_map = np.bincount(spike_spatial_idx[mask], minlength=nx * ny)
        rm = (spike_map / safe_occ).reshape(ny, nx)
        rm[occ.reshape(ny, nx) == 0] = 0 
        maps[neuron_id] = rm
        
    return maps

   
#%%
if __name__ == "__main__":
    #LISTA DE BUENOS: 10
    num = 1
    superficial = 'False'
    SMOOTH = True
    NEURON_IDX = 78#78#64, 86, 31,71np.random.randint(100)
    
    path2load = os.path.split(os.getcwd())[0]    
    if superficial == 'True':
        path2load = os.path.join(path2load,f'SimulationSpikingSuperficial-num{num}')
    elif superficial == 'False':
        path2load = os.path.join(path2load,f'SimulationSpikingDeep-num{num}')
    else:
        path2load = os.path.join(path2load,f'SimulationSpikingDeep-num{num}')

    # Load parameters to get L and other info for plotting
    path2load_0 = os.path.join(path2load,f'incl_ang{1}')
    with open(os.path.join(path2load_0,'parameters_reduced.pkl'),'rb') as file:
        parameters = pickle.load(file)
                
    mean_fr_omni = []
    mean_fr_conj = []
    L = parameters['L']
    
    # Compute maps from sparse data (shape is already N, ny, nx)
    omni_maps, conj_maps, traj_list = load_and_compute_maps_from_sparse(num=num, 
                                                                        sim_folder=path2load,
                                                                        nx=30, ny=30)
    # load resolution and number of neurons
    nfr = omni_maps[0]['rate maps'].shape[1]
    N = omni_maps[0]['rate maps'].shape[0]
    N_conj = conj_maps[0]['rate maps'].shape[0]
    
    shifts_x = []
    shifts_y = []
    
    shifts_conj_x = []
    shifts_conj_y = []
    sd = 2#2.5
    plt.figure(40,figsize=(18,24))
    
    i_0 = 22
    i_f = 37
    
    angles = ['0',r'$\pi/6$',r'$\pi/3$','0\'']
    
    np.random.seed(32)
    conj_idx = np.random.permutation(N_conj)[:100]
    
#%% FIRING RATE MAPS, CROSS CORRELATIONS AND SPIKING MAPS
    for i in range(4):
        shape = omni_maps[i]['rate maps'].shape
        mean_fr_omni.append(omni_maps[i]['rate maps'].reshape((N,shape[1]*shape[2])).mean(axis=1))
        mean_fr_conj.append(conj_maps[i]['rate maps'].reshape((N_conj,shape[1]*shape[2])).mean(axis=1))
        if i>0:
            # Arrays are already (N, ny, nx), no reshaping needed
            plt.figure(1,figsize=(15,10))
            shifts, CrossCorr = compute_cross_corrs(
                omni_maps[0]['rate maps'],
                omni_maps[i]['rate maps'],
                smooth=SMOOTH,sd=sd)
            shifts_y.append((shifts[:,1]) * L / nfr)
            shifts_x.append(shifts[:,0] * L / nfr)    
            
            plt.subplot(2,3,i)
            plt.imshow(CrossCorr.mean(axis=0),cmap='jet')#,origin='lower')
            plt.axis('off')
            
            plt.subplot(2,3,i+3)
            plt.imshow(CrossCorr.mean(axis=0)[i_0:i_f,i_0:i_f],cmap='jet')#,
                       # origin='lower')
            plt.axis('off')
            
            shifts, CrossCorr = compute_cross_corrs(
                conj_maps[0]['rate maps'][conj_idx],
                conj_maps[i]['rate maps'][conj_idx],
                smooth=SMOOTH,sd=sd)
            shifts_conj_y.append((shifts[:,1]) * L / nfr)
            shifts_conj_x.append(shifts[:,0] * L / nfr)    
        
        plt.figure(2,figsize=(20,10))
        plt.subplot(2,4,i+1)
        idx = np.where(omni_maps[i]['spiking maps']['neuron_idx'] == NEURON_IDX)[0]
        idx = omni_maps[i]['spiking maps']['time_idx'][idx]
        plt.plot(traj_list[i][:,0],traj_list[i][:,1],color='gray',alpha=0.7)
        plt.plot(traj_list[i][idx,0],traj_list[i][idx,1],'.',
                 color='r',markersize=10)
        plt.axis('off')
        plt.title(angles[i])
        plt.subplot(2,4,i+5)
        RM = gaussian_filter(omni_maps[i]['rate maps'][NEURON_IDX],
                                   (sd,sd),mode='constant', cval=0)
        plt.imshow(RM,
                   cmap='jet',origin='lower')
        plt.axis('off')
        plt.title(f'Max Fr: {RM.max():.2f}, Mean Fr: {RM.mean():.2f}')
        
        plt.figure(3,figsize=(20,10))
        plt.subplot(2,4,i+1)
        idx = np.where(conj_maps[i]['spiking maps']['neuron_idx'] == conj_idx[NEURON_IDX])[0]
        idx = conj_maps[i]['spiking maps']['time_idx'][idx]
        plt.plot(traj_list[i][:,0],traj_list[i][:,1],color='gray',alpha=0.7)
        plt.plot(traj_list[i][idx,0],traj_list[i][idx,1],'.',
                 color='r',markersize=10)
        plt.axis('off')
        plt.title(angles[i])
        plt.subplot(2,4,i+5)
        RM = gaussian_filter(conj_maps[i]['rate maps'][conj_idx[NEURON_IDX]],
                                   (sd,sd),mode='constant', cval=0)
        plt.imshow(RM,
                   cmap='jet',origin='lower')
        plt.axis('off')
        plt.title(f'Max Fr: {RM.max():.2f}, Mean Fr: {RM.mean():.2f}')
                
    plt.show()

#%% SPATIAL SHIFTS    
    stripplot = False
    plt.figure(4,figsize=(10,10))
    ## OMNIDIRECTIONAL CELLS
    plt.subplot(221)
    sns.boxplot(shifts_x,fill=False)
    if stripplot:
        sns.stripplot(shifts_x)
    plt.plot([-1,3],[0,0],'--k')
    F, P = f_oneway(shifts_x[0],shifts_x[1],shifts_x[2])
    plt.title(f'Omni x shift, Anova p-val={P:.2e}')
    plt.xlim([-.5,2.5])
    plt.xticks([0,1,2],labels=angles[1:])
    
    plt.subplot(222)
    sns.boxplot(shifts_y,fill=False)
    if stripplot:
        sns.stripplot(shifts_y)
    plt.plot([-1,3],[0,0],'--k')
    F, P = f_oneway(shifts_y[0],shifts_y[1],shifts_y[2])
    plt.title(f'Omni y shift, Anova p-val={P:.2e}')
    plt.xlim([-.5,2.5])
    plt.xticks([0,1,2],labels=angles[1:])
    # plt.ylim([-2.5,6])
    
    ## CONJUNCTIVE CELLS
    plt.subplot(223)
    sns.boxplot(shifts_conj_x,fill=False)
    if stripplot:
        sns.stripplot(shifts_conj_x)
    plt.plot([-1,3],[0,0],'--k')
    F, P = f_oneway(shifts_conj_x[0],shifts_conj_x[1],shifts_conj_x[2])
    plt.title(f'Conj x shift, Anova p-val={P:.2e}')
    plt.xlim([-.5,2.5])
    plt.xticks([0,1,2],labels=angles[1:])
    
    plt.subplot(224)
    sns.boxplot(shifts_conj_y,fill=False)
    if stripplot:
        sns.stripplot(shifts_conj_y)
    plt.plot([-1,3],[0,0],'--k')
    F, P = f_oneway(shifts_conj_y[0],shifts_conj_y[1],shifts_conj_y[2])
    plt.title(f'Conj y shift, Anova p-val={P:.2e}')
    plt.xlim([-.5,2.5])
    plt.xticks([0,1,2],labels=angles[1:])
    
    plt.tight_layout()
    plt.show()
    
#%% MEAN FIRING RATE ACROSS SESSIONS
    ## OMNIDIRECTIONAL CELLS    
    plt.figure(5,figsize=(10,5))
    plt.subplot(121)
    plt.boxplot(mean_fr_omni)
    F, P = f_oneway(mean_fr_omni[0],mean_fr_omni[1],mean_fr_omni[2],mean_fr_omni[3])
    plt.title(f'Omni FR, Anova p-val={P:.2e}')
    plt.subplot(122)
    F, P = f_oneway(mean_fr_conj[0],
                    mean_fr_conj[1],
                    mean_fr_conj[2],
                    mean_fr_conj[3])
    plt.title(f'Conj FR, Anova p-val={P:.2e}')
    plt.boxplot(mean_fr_conj)
    plt.show()
    
    ## CONJUNCTIVE CELLS   
    with open(os.path.join(path2load_0,'parameters_complete.pkl'),'rb') as file:
        parameters_complete = pickle.load(file)
        
    # plt.subplot(212)
    MEAN_HD_corrected = np.zeros((parameters_complete['hd_modules'],4,N_conj//parameters_complete['hd_modules']))
    hds = np.linspace(0,2*np.pi,parameters_complete['hd_modules']+1)[:-1]#[::2]
    
    dhd = hds[1]
    for i_hd in range(len(hds)):
        IDX = np.arange(i_hd * N_conj//parameters_complete['hd_modules'],
                        (i_hd+1) * N_conj//parameters_complete['hd_modules'])
        for i in range(4):
            hd_dummy = np.mod(traj_list[i][:,-1] -hds[i_hd] + np.pi, 2*np.pi)
            
            corr_fact = (len(hd_dummy)/np.sum((hd_dummy>np.pi-dhd) & (hd_dummy<np.pi+dhd)))
            MEAN_HD_corrected[i_hd,i,:] = (mean_fr_conj[i][IDX] * corr_fact)
        
    plt.figure(6,figsize=(5 * len(hds)//2,5))
    for i_hd in range(len(hds)):
        plt.subplot(2,len(hds)//2,i_hd+1)
        F, P = f_oneway(MEAN_HD_corrected[i_hd][0],
                        MEAN_HD_corrected[i_hd][1],
                        MEAN_HD_corrected[i_hd][2],
                        MEAN_HD_corrected[i_hd][3])
        plt.title(f'HD={hds[i_hd]:.2f}, Conj FR, Anova p-val={P:.2e}')
        plt.boxplot(MEAN_HD_corrected[i_hd].T)
        plt.xticks(np.arange(1,1+len(angles)),labels=angles)
    plt.tight_layout()
        
        
    #%%
    print("Computing Directional Rate Maps and Shifts...")
    
    omni_up, omni_down = [], []
    conj_up, conj_down = [], []

    # 1. Compute Up and Down rate maps for all sessions
    for i in range(4):
        hd = np.mod(traj_list[i][:, 2], 2 * np.pi)
        is_up = (hd >= 0) & (hd < np.pi)
        is_down = (hd >= np.pi) & (hd < 2 * np.pi)

        omni_up.append(compute_directional_rate_maps(omni_maps[i]['spiking maps'], traj_list[i], is_up, N, L, nx=nfr, ny=nfr))
        omni_down.append(compute_directional_rate_maps(omni_maps[i]['spiking maps'], traj_list[i], is_down, N, L, nx=nfr, ny=nfr))

        conj_up.append(compute_directional_rate_maps(conj_maps[i]['spiking maps'], traj_list[i], is_up, N_conj, L, nx=nfr, ny=nfr))
        conj_down.append(compute_directional_rate_maps(conj_maps[i]['spiking maps'], traj_list[i], is_down, N_conj, L, nx=nfr, ny=nfr))

    # 2. Compute cross-correlograms and spatial shifts against baseline (Session 0)
    data_shifts = []
    session_labels = angles[1:]  # [pi/6, pi/3, 0']

    for i, ses_label in enumerate(session_labels, start=1):
        # Omni Up
        s_up, _ = compute_cross_corrs(omni_up[0], omni_up[i], smooth=SMOOTH, sd=sd)
        # Omni Down
        s_down, _ = compute_cross_corrs(omni_down[0], omni_down[i], smooth=SMOOTH, sd=sd)
        
        for sx, sy in zip(s_up[:, 0] * L / nfr, s_up[:, 1] * L / nfr):
            data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 'Direction': 'Up [0, $\pi$)', 'Type': 'Omni'})
        for sx, sy in zip(s_down[:, 0] * L / nfr, s_down[:, 1] * L / nfr):
            data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 'Direction': 'Down [$\pi$, 2$\pi$)', 'Type': 'Omni'})

        # Conj Up (subset indexing applied as in the original code)
        s_c_up, _ = compute_cross_corrs(conj_up[0][conj_idx], conj_up[i][conj_idx], smooth=SMOOTH, sd=sd)
        # Conj Down
        s_c_down, _ = compute_cross_corrs(conj_down[0][conj_idx], conj_down[i][conj_idx], smooth=SMOOTH, sd=sd)
        
        for sx, sy in zip(s_c_up[:, 0] * L / nfr, s_c_up[:, 1] * L / nfr):
            data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 'Direction': 'Up [0, $\pi$)', 'Type': 'Conj'})
        for sx, sy in zip(s_c_down[:, 0] * L / nfr, s_c_down[:, 1] * L / nfr):
            data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 'Direction': 'Down [$\pi$, 2$\pi$)', 'Type': 'Conj'})

    df_shifts = pd.DataFrame(data_shifts)

    # 3. Plotting the paired distributions
    plt.figure(7, figsize=(14, 12))

    # Omni X
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_shifts[df_shifts['Type'] == 'Omni'], x='Session', y='Shift X', hue='Direction', fill=False)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Omnidirectional Cells - Shift X')

    # Omni Y
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_shifts[df_shifts['Type'] == 'Omni'], x='Session', y='Shift Y', hue='Direction', fill=False)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Omnidirectional Cells - Shift Y')

    # Conj X
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_shifts[df_shifts['Type'] == 'Conj'], x='Session', y='Shift X', hue='Direction', fill=False)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Conjunctive Cells - Shift X')

    # Conj Y
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_shifts[df_shifts['Type'] == 'Conj'], x='Session', y='Shift Y', hue='Direction', fill=False)
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Conjunctive Cells - Shift Y')

    plt.tight_layout()
    plt.show()

    print(f"d_asym = {parameters_complete['d_asym']:.2f}")
    print(f"l_torus = {parameters_complete['l_torus']:.2f}")
    print(f"input_std = {parameters_complete['input_std']:.2f}")
    print(f"m = {parameters_complete['m']:.2f}")
    if 'thresh' in parameters_complete:
        print(f"thresh = {parameters_complete['thresh']}")
    print(f"inclination_dir = {parameters_complete['inclination_dir']}")