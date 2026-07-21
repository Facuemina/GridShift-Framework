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
def load_omni_maps(num, target_d_asym=0, target_l_torus=0, base_dir=None):
    """
    Scans the Simulation directory and loads true_rate_map_omni.npy 
    for folders that match the target asymmetry and periodicity.
    """
    # Replicate your main job's path logic if no base directory is provided
    if base_dir is None:
        base_dir = os.path.split(os.getcwd())[0]
        
    sim_folder = os.path.join(base_dir, f'Simulation-num{num}')
    
    if not os.path.exists(sim_folder):
        raise FileNotFoundError(f"Simulation directory not found: {sim_folder}")
        
    # Dictionary to store the arrays, keyed by the inclination angle index
    loaded_maps = {}
    loaded_maps_conj = {}
    # Iterate through all inclination angle folders
    for folder_name in os.listdir(sim_folder):
        if not folder_name.startswith('incl_ang'):
            continue
            
        folder_path = os.path.join(sim_folder, folder_name)
        param_path = os.path.join(folder_path, 'parameters_reduced.pkl')
        map_path = os.path.join(folder_path, 'true_rate_map_omni.npy')
        map_conj_path = os.path.join(folder_path, 'true_rate_map_conj.npy')
        
        if not os.path.exists(param_path) or not os.path.exists(map_path):
            print(f"Skipping {folder_name}: Missing .pkl or .npy file.")
            continue
            
        # 1. Load the parameters to check if they match our targets
        with open(param_path, 'rb') as file:
            params = pickle.load(file)
            
        # 2. Check conditions (note: your script saved l_torus under the key 'l')
        if 1:#params.get('d_asym') == target_d_asym and params.get('l') == target_l_torus:
            # 3. Load the numpy array
            rate_map = np.load(map_path)
            rate_map_conj = np.load(map_conj_path)
            
            # Use the inclination index as the dictionary key
            incl_idx = params.get('inclination_angle')
            loaded_maps[incl_idx] = rate_map
            loaded_maps_conj[incl_idx] = rate_map_conj
            
            print(f"Loaded map for incl_ang {incl_idx} (Shape: {rate_map.shape})")
            
    return loaded_maps, loaded_maps_conj

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
        sx, sy = find_spatial_shift_subpixel(CrossCorr[i], n = 7, search_radius_pixels = 7)#[:,::-1].T)
        shifts[i,0], shifts[i,1] = sx, sy
    
    return shifts, CrossCorr

#%%
if __name__ == "__main__":
    # Example execution based on your main job parameters
    num = 8
    
    SMOOTH = False
    
    NEURON_IDX = 54
    
    path2load = os.path.split(os.getcwd())[0]    
    path2load = os.path.join(path2load,f'SimulationFiringRate-num{num}')

    path2load = os.path.join(path2load,f'incl_ang{0}')
    with open(os.path.join(path2load,'parameters_reduced.pkl'),'rb') as file:
        parameters = pickle.load(file)
                
    L = parameters['L']
    
    omni_maps, conj_maps = load_omni_maps(num=num)#, target_d_asym=d_asym, target_l_torus=l_torus)
    
    nfr = int(np.sqrt(omni_maps[0].shape[1]))
    N = omni_maps[0].shape[0]
    N_conj = conj_maps[0].shape[0]
    shifts_x = []
    shifts_y = []
    
    shifts_conj_x = []
    shifts_conj_y = []
    sd = 2
    plt.figure(40,figsize=(15,20))
    
    i_0 = 22
    i_f = 37
    
    
    angles = [r'$\pi/6$',r'$\pi/3$','0\'']
    for i in range(1,4):
        shifts, CrossCorr = compute_cross_corrs(
            omni_maps[0].reshape((N,nfr,nfr)),
            omni_maps[i].reshape((N,nfr,nfr)),
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
            conj_maps[0].reshape((N_conj,nfr,nfr)),
            conj_maps[i].reshape((N_conj,nfr,nfr)),
            smooth=SMOOTH,sd=sd)
        shifts_conj_y.append((shifts[:,0]) * L / nfr)
        shifts_conj_x.append(shifts[:,1] * L / nfr)    
        
        plt.subplot(4,3,i+6)
        plt.imshow(CrossCorr.mean(axis=0)[::-1,:],cmap='jet')
        plt.axis('off')
        
        
        plt.subplot(4,3,i+9)
        plt.imshow(CrossCorr.mean(axis=0)[::-1,:][i_0:i_f,i_0:i_f],cmap='jet')
        plt.axis('off')
    plt.show()
        
    plt.figure(i,figsize=(10,10))
    plt.subplot(221)
    sns.boxplot(shifts_x,fill=False)
    sns.stripplot(shifts_x)
    # f_oneway(shifts_x)
    plt.title('Omni x shift')
    # plt.ylim([-1.5,2])
    plt.xticks([0,1,2],labels=angles)
    plt.subplot(222)
    sns.boxplot(shifts_y,fill=False)
    sns.stripplot(shifts_y)
    # plt.ylim([-1.5,2])
    plt.title('Omni y shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.subplot(223)
    sns.boxplot(shifts_conj_x,fill=False)
    sns.stripplot(shifts_conj_x)
    plt.title('Conj x shift')
    plt.xticks([0,1,2],labels=angles)
    # plt.ylim([-1.5,2])
    plt.subplot(224)
    sns.boxplot(shifts_conj_y,fill=False)
    sns.stripplot(shifts_conj_y)
    # plt.ylim([-1.5,2])
    plt.title('Conj y shift')
    plt.xticks([0,1,2],labels=angles)
    
    plt.tight_layout()
    
    plt.show()
    
    plt.figure(figsize=(10,10))
    idx = [1,2,3]
    rm0 = gaussian_filter(omni_maps[0][NEURON_IDX].reshape((nfr,nfr)),(1,1),
                          mode='constant', cval=0)
    
    for i in range(3):
        rm1 = omni_maps[idx[i]][NEURON_IDX].reshape((nfr,nfr))
        rm1_conj = conj_maps[idx[i]][NEURON_IDX].reshape((nfr,nfr))
        if SMOOTH:
            rm1 = gaussian_filter(rm1,(sd,sd),
                                  mode='constant', cval=0)[:,:]
            rm1_conj = gaussian_filter(rm1_conj,(sd,sd),
                                       mode='constant', cval=0)[:,:]
        plt.subplot(3,3,i+1)
        plt.imshow(correlate2d((rm0-rm0.mean())/np.std(rm0),
                               (rm1-rm1.mean())/np.std(rm1))[::-1,:], cmap='jet')
        
        plt.axis('off')
        plt.title(f'Angle {angles[i]}',fontsize=15)
        plt.subplot(3,3,3+i+1)
        plt.imshow(rm1,cmap='jet')
        plt.title(f'Omni Mean fr {omni_maps[i].reshape((N,nfr,nfr))[0].mean():.4f}',
                  fontsize=15)
        plt.axis('off')
        
        plt.subplot(3,3,6+i+1)
        plt.imshow(rm1_conj,cmap='jet')
        plt.title(f'Conj Mean fr {conj_maps[i].reshape((N_conj,nfr,nfr))[0].mean():.4f}',# \n Max fr {conj_maps[i].reshape((N_conj,nfr,nfr))[0].max():.4f} ',
                  fontsize=15)
        plt.axis('off')
        
    with open(os.path.join(path2load,'parameters_complete.pkl'),'rb') as file:
        parameters_complete = pickle.load(file)

    print(f'd_asym ={parameters_complete['d_asym']:.2f}')
    print(f'l_torus = {parameters_complete['l_torus']:.2f}')
    print(f'input_std = {parameters_complete['input_std']:.2f}')
    print(f'm = {parameters_complete['m']:.2f}')
    if 'thresh' in parameters_complete.keys():
        print(f'thresh = {parameters_complete['thresh']}')