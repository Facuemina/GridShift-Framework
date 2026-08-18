# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:09:28 2026

@author: Facundo
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.ndimage import gaussian_filter

from src.utils import (
    load_and_compute_maps_from_sparse,
    compute_cross_corrs,
    compute_directional_rate_maps
)

if __name__ == "__main__":
    #%% =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    num = 95
    superficial = 'False'
    VISUAL = 'OFF'
    SMOOTH = False
    NEURON_IDX = 45#53#61
    fact = 1.67
    nfr = int(30 * fact)
    sd = int(2 * fact)
    i_0 = int(22 * fact)
    i_f = int(37 * fact) + 2
    
       
    path2load = os.path.split(os.getcwd())[0]    
    path2load = os.path.join(path2load, f'Simulation-2layer-VISUAL_{VISUAL}-num{num}')
    
    if len(os.listdir(path2load)) == 2:
        angles = ['0', r'$\pi/3$']
    elif len(os.listdir(path2load)) == 4:
        angles = ['0', r'$\pi/6$', r'$\pi/3$', "0'"]
    elif len(os.listdir(path2load)) == 5:
        angles = ['0', r'$\pi/6$', r'$\pi/3$', r'$\pi/3$D', "0'"]
    
    path2load_0 = os.path.join(path2load, f'incl_ang{1}')
    
    with open(os.path.join(path2load_0, 'parameters_complete.pkl'), 'rb') as file:
        parameters_complete = pickle.load(file)
        
    L = parameters_complete['L']
    hd_modules = parameters_complete['hd_modules']
    dt = parameters_complete['dt']

    #%% =========================================================================
    # 2. DATA LOADING & PREPARATION
    # =========================================================================
    print("Loading data and computing rate maps...")
    omni_maps, conj_maps1, conj_maps2, traj_list = load_and_compute_maps_from_sparse(
        num=num, sim_folder=path2load, nx=nfr, ny=nfr
    )
        
    nfr = omni_maps[0]['rate maps'].shape[1]
    N = omni_maps[0]['rate maps'].shape[0]
    N_conj1 = conj_maps1[0]['rate maps'].shape[0]
    N_conj2 = conj_maps2[0]['rate maps'].shape[0]
    
    # Subsample neurons for plotting/analysis
    np.random.seed(32)
    conj_idx1 = np.random.permutation(N_conj1)[:100]#np.arange(N_conj1)#
    conj_idx2 = np.random.permutation(N_conj2)[:100]#np.arange(N_conj2)#
    
    # Identify preferred head directions
    pref_hd_all1 = np.repeat(np.linspace(0, 2*np.pi, hd_modules+1)[:-1], N_conj1 // hd_modules)
    pref_hd_all2 = np.repeat(np.linspace(0, 2*np.pi, hd_modules+1)[:-1], N_conj2 // hd_modules)
    
    pref_hd_subset1 = pref_hd_all1[conj_idx1]
    up_HD_idx1 = conj_idx1[np.where((pref_hd_subset1 > 0) & (pref_hd_subset1 < np.pi))[0]]
    down_HD_idx1 = conj_idx1[np.where((pref_hd_subset1 > np.pi) & (pref_hd_subset1 < 2*np.pi))[0]]
    
    pref_hd_subset2 = pref_hd_all2[conj_idx2]
    up_HD_idx2 = conj_idx2[np.where((pref_hd_subset2 > 0) & (pref_hd_subset2 < np.pi))[0]]
    down_HD_idx2 = conj_idx2[np.where((pref_hd_subset2 > np.pi) & (pref_hd_subset2 < 2*np.pi))[0]]

    #%% =========================================================================
    # 3. COMPUTATIONS (Mean FRs, Cross-Correlations, Shifts)
    # =========================================================================
    print("Computing Cross Correlations and Shifts...")
    mean_fr_omni, mean_fr_conj1, mean_fr_conj2 = [], [], []
    mean_fr_conj_up1, mean_fr_conj_down1 = [], []
    mean_fr_conj_up2, mean_fr_conj_down2 = [], []
    
    CrossCorr_omni, CrossCorr_conj1, CrossCorr_conj2 = [], [], []
    shifts_x, shifts_y = [], []
    shifts_conj1_x, shifts_conj1_y = [], []
    shifts_conj2_x, shifts_conj2_y = [], []
    
    for i in range(len(angles)):
        shape = omni_maps[i]['rate maps'].shape
        
        # Mean Firing Rates
        mean_fr_omni.append(omni_maps[i]['rate maps'].reshape((N, shape[1]*shape[2])).mean(axis=1))
        
        mean_fr_conj1.append(conj_maps1[i]['rate maps'].reshape((N_conj1, shape[1]*shape[2])).mean(axis=1))
        mean_fr_conj_up1.append(mean_fr_conj1[i][up_HD_idx1])
        mean_fr_conj_down1.append(mean_fr_conj1[i][down_HD_idx1])

        mean_fr_conj2.append(conj_maps2[i]['rate maps'].reshape((N_conj2, shape[1]*shape[2])).mean(axis=1))
        mean_fr_conj_up2.append(mean_fr_conj2[i][up_HD_idx2])
        mean_fr_conj_down2.append(mean_fr_conj2[i][down_HD_idx2])
        
        # Cross-Correlations and Shifts (relative to baseline session 0)
        shifts_omni, CC_omni = compute_cross_corrs(
            omni_maps[0]['rate maps'], omni_maps[i]['rate maps'], smooth=SMOOTH, sd=sd)
        CrossCorr_omni.append(CC_omni)
        
        shifts_c1, CC_c1 = compute_cross_corrs(
            conj_maps1[0]['rate maps'][conj_idx1], conj_maps1[i]['rate maps'][conj_idx1], smooth=SMOOTH, sd=sd)
        CrossCorr_conj1.append(CC_c1)
        
        shifts_c2, CC_c2 = compute_cross_corrs(
            conj_maps2[0]['rate maps'][conj_idx2], conj_maps2[i]['rate maps'][conj_idx2], smooth=SMOOTH, sd=sd)
        CrossCorr_conj2.append(CC_c2)
        
        if i > 0:
            if len(shifts_c2)<N_conj2:
                sh = 0
            else:
                sh = 0 * shifts_c2.reshape((hd_modules,
                                        N_conj2 // hd_modules,
                                        2)).mean(axis=0)
            dshift = shifts_omni - sh
            shifts_y.append(dshift[:, 1] * L / nfr)
            shifts_x.append(dshift[:, 0] * L / nfr)    
            
            shifts_conj1_y.append(shifts_c1[:, 1] * L / nfr)
            shifts_conj1_x.append(shifts_c1[:, 0] * L / nfr)    
            
            shifts_conj2_y.append(shifts_c2[:, 1] * L / nfr)
            shifts_conj2_x.append(shifts_c2[:, 0] * L / nfr)

    #%% =========================================================================
    # 4. PLOTTING: Rate Maps and Cross Corrs
    # =========================================================================
# for NEURON_IDX in [12,45,70,79]:    
    print("Generating standard plots...")
    # bound_x0, bound_x1 = 5, 34
    # bound_y0, bound_y1 = 28, 50
    bound_x0, bound_x1 = 25, 50
    bound_y0, bound_y1 = 0, 27

    for i in range(len(angles)):
        if i > 0:
            # Figure 1: Mean Cross Correlograms
            plt.figure(1, figsize=(15, 10))
            plt.subplot(2, 3, i)
            plt.imshow(CrossCorr_omni[i].mean(axis=0), cmap='jet')
            plt.axis('off')
            plt.title(f'Mean CC: {angles[i]}')
            
            plt.subplot(2, 3, i+3)
            plt.imshow(CrossCorr_omni[i].mean(axis=0)[i_0:i_f, i_0:i_f], cmap='jet')
            plt.axis('off')
            
        # Figure 2: Trajectory Overlay - Omni
        if len(omni_maps[i]['spiking maps']['time_idx']) == 0:
            print('No Omni maps!')
        else:
            plt.figure(2, figsize=(16, 12))
            
            # Overlay Map
            plt.subplot(3, 4, i+1)
            idx = np.where(omni_maps[i]['spiking maps']['neuron_idx'] == NEURON_IDX)[0]
            idx = omni_maps[i]['spiking maps']['time_idx'][idx]
            plt.plot(traj_list[i][:, 0], traj_list[i][:, 1], color='gray', alpha=0.7)
            plt.plot(traj_list[i][idx, 0], traj_list[i][idx, 1], '.', color='k', markersize=10)
            
            idx_red = np.where((traj_list[i][idx, 0] > bound_x0) & (traj_list[i][idx, 0] < bound_x1) & 
                               (traj_list[i][idx, 1] > bound_y0) & (traj_list[i][idx, 1] < bound_y1))[0]
            xred = traj_list[i][idx, 0][idx_red]
            yred = traj_list[i][idx, 1][idx_red]
            
            if i == 0:
                yred_cm = yred.mean()
                
            plt.plot(xred, yred, '.', color='r', markersize=10)
            plt.plot([0, 50], [yred_cm, yred_cm], '--', color=[.0, .0, .8], linewidth=5)
            plt.axis('off')
            plt.title(f'Omni Traj: {angles[i]}')
            
            # Raw Rate Map
            plt.subplot(3, 4, i+5)
            RM_omni = omni_maps[i]['rate maps'][NEURON_IDX]
            plt.imshow(RM_omni, cmap='jet', origin='lower')
            plt.axis('off')
            plt.title(f'Max: {RM_omni.max():.2f}, Mean: {RM_omni.mean():.2f}')
            
            # Single Neuron Cross Corr
            plt.subplot(3, 4, i+9)
            _, CC_OMNI_single = compute_cross_corrs(
                omni_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                omni_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                smooth=SMOOTH, sd=sd)
            plt.imshow(CC_OMNI_single[0], cmap='jet', origin='lower')
            plt.axis('off')

        # Figure 3 & 4: Trajectory Overlay - Conjunctive 1 & 2
        for cm_i, (c_maps, c_idx) in enumerate(zip([conj_maps1, conj_maps2], [conj_idx1, conj_idx2])):
            plt.figure(3 + cm_i, figsize=(16, 12))
            
            plt.subplot(3, 4, i+1)
            idx_c = np.where(c_maps[i]['spiking maps']['neuron_idx'] == c_idx[NEURON_IDX])[0]
            idx_c = c_maps[i]['spiking maps']['time_idx'][idx_c]
            plt.plot(traj_list[i][:, 0], traj_list[i][:, 1], color='gray', alpha=0.7)
            plt.plot(traj_list[i][idx_c, 0], traj_list[i][idx_c, 1], '.', color='r', markersize=10)
            plt.axis('off')
            plt.title(f'Conj{cm_i+1} Traj: {angles[i]}')
            
            plt.subplot(3, 4, i+5)
            RM_conj = c_maps[i]['rate maps'][c_idx[NEURON_IDX]]
            plt.imshow(RM_conj, cmap='jet', origin='lower')
            plt.axis('off')
            plt.title(f'Max: {RM_conj.max():.2f}, Mean: {RM_conj.mean():.2f}')
            
            plt.subplot(3, 4, i+9)
            _, CC_CONJ_single = compute_cross_corrs(
                c_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                c_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                smooth=SMOOTH, sd=sd)
            plt.imshow(CC_CONJ_single[0], cmap='jet', origin='lower')
            plt.axis('off')

    plt.show()

    #%% =========================================================================
    # 5. PLOTTING: Spatial Shifts (Boxplots)
    # =========================================================================
    plt.figure(5, figsize=(10, 10))
    
    # Omnidirectional Cells
    plt.subplot(3, 2, 1)
    sns.boxplot(data=shifts_x, fill=False)
    plt.plot([-1, 3], [0, 0], '--k')
    if len(shifts_x) > 1:
        F, P = friedmanchisquare(*shifts_x)
        plt.title(f'Omni X Shift (F p-val={P:.2e})')
    plt.xlim([-.5, 2.5])
    plt.xticks(range(len(angles)-1), labels=angles[1:])
    
    plt.subplot(3, 2, 2)
    sns.boxplot(data=shifts_y, fill=False)
    plt.plot([-1, 3], [0, 0], '--k')
    if len(shifts_y) > 1:
        F, P = friedmanchisquare(*shifts_y)
        plt.title(f'Omni Y Shift (F p-val={P:.2e})')
    plt.xlim([-.5, 2.5])
    plt.xticks(range(len(angles)-1), labels=angles[1:])
    
    # Conjunctive Cells 1 & 2
    for i_shift, (shifts_conj_x, shifts_conj_y) in enumerate(zip([shifts_conj1_x, shifts_conj2_x], [shifts_conj1_y, shifts_conj2_y])):
        
        plt.subplot(3, 2, 2 * i_shift + 3)
        sns.boxplot(data=shifts_conj_x, fill=False)
        plt.plot([-1, 3], [0, 0], '--k')
        if len(shifts_conj_x) > 1:
            F, P = friedmanchisquare(*shifts_conj_x)
            plt.title(f'Conj{i_shift+1} X Shift (F p-val={P:.2e})')
        plt.xlim([-.5, 2.5])
        plt.xticks(range(len(angles)-1), labels=angles[1:])
        
        plt.subplot(3, 2, 2 * i_shift + 4)
        sns.boxplot(data=shifts_conj_y, fill=False)
        plt.plot([-1, 3], [0, 0], '--k')
        if len(shifts_conj_y) > 1:
            F, P = friedmanchisquare(*shifts_conj_y)
            plt.title(f'Conj{i_shift+1} Y Shift (F p-val={P:.2e})')
        plt.xlim([-.5, 2.5])
        plt.xticks(range(len(angles)-1), labels=angles[1:])
    
    plt.tight_layout()
    plt.show()
    
    #%% =========================================================================
    # 6. PLOTTING: Mean Firing Rates
    # =========================================================================
    plt.figure(6, figsize=(15, 5))
    
    plt.subplot(1, 5, 1)
    plt.boxplot(mean_fr_omni)
    if len(mean_fr_omni)>2:
        F, P = friedmanchisquare(*mean_fr_omni)
        plt.title(f'Omni FR (F p-val={P:.2e})')
    plt.xticks(np.arange(1, len(angles)+1), labels=angles)
    
    plt.subplot(1, 5, 2)
    plt.boxplot(mean_fr_conj_up1)
    if len(mean_fr_conj_up1)>2:
        F, P = friedmanchisquare(*mean_fr_conj_up1)
        plt.title(f'Conj1 UP FR (F p-val={P:.2e})')
    plt.xticks(np.arange(1, len(angles)+1), labels=angles)
    
    plt.subplot(1, 5, 3)
    plt.boxplot(mean_fr_conj_down1)
    if len(mean_fr_conj_down1)>2:
        F, P = friedmanchisquare(*mean_fr_conj_down1)
        plt.title(f'Conj1 DOWN FR (F p-val={P:.2e})')
    plt.xticks(np.arange(1, len(angles)+1), labels=angles)
    
    plt.subplot(1, 5, 4)
    plt.boxplot(mean_fr_conj_up1)
    if len(mean_fr_conj_up1)>2:
        F, P = friedmanchisquare(*mean_fr_conj_up2)
        plt.title(f'Conj1 UP FR (F p-val={P:.2e})')
    plt.xticks(np.arange(1, len(angles)+1), labels=angles)
    
    plt.subplot(1, 5, 5)
    plt.boxplot(mean_fr_conj_down1)
    if len(mean_fr_conj_down1)>2:
        F, P = friedmanchisquare(*mean_fr_conj_down2)
        plt.title(f'Conj1 DOWN FR (F p-val={P:.2e})')
    plt.xticks(np.arange(1, len(angles)+1), labels=angles)
    
    plt.tight_layout()
    plt.show()
    
    #%% =========================================================================
    # 7. DIRECTIONAL ANALYSIS & DATAFRAME CREATION
    # =========================================================================
    print("Computing Directional Rate Maps and Shifts...")
    omni_up, omni_down = [], []
    conj1_up, conj1_down = [], []
    conj2_up, conj2_down = [], []

    for i in range(len(angles)):
        is_up = (traj_list[i][:, 2] >= 0) & (traj_list[i][:, 2] < np.pi)
        is_down = (traj_list[i][:, 2] >= np.pi) & (traj_list[i][:, 2] < 2 * np.pi)

        # Compute Directional Maps for Omni
        omni_up.append(compute_directional_rate_maps(
            omni_maps[i]['spiking maps'], traj_list[i], is_up, N, L, dt, nx=nfr, ny=nfr))
        omni_down.append(compute_directional_rate_maps(
            omni_maps[i]['spiking maps'], traj_list[i], is_down, N, L, dt, nx=nfr, ny=nfr))
            
        # Compute Directional Maps for Conj1
        conj1_up.append(compute_directional_rate_maps(
            conj_maps1[i]['spiking maps'], traj_list[i], is_up, N_conj1, L, dt, nx=nfr, ny=nfr))
        conj1_down.append(compute_directional_rate_maps(
            conj_maps1[i]['spiking maps'], traj_list[i], is_down, N_conj1, L, dt, nx=nfr, ny=nfr))
            
        # Compute Directional Maps for Conj2
        conj2_up.append(compute_directional_rate_maps(
            conj_maps2[i]['spiking maps'], traj_list[i], is_up, N_conj2, L, dt, nx=nfr, ny=nfr))
        conj2_down.append(compute_directional_rate_maps(
            conj_maps2[i]['spiking maps'], traj_list[i], is_down, N_conj2, L, dt, nx=nfr, ny=nfr))

    data_shifts = []
    session_labels = angles[1:] 
    
    # Bundle the populations to avoid repeating code
    cell_groups = [
        ('Omni', omni_up, omni_down, np.arange(N)),
        ('Conj1', conj1_up, conj1_down, conj_idx1),
        ('Conj2', conj2_up, conj2_down, conj_idx2)
    ]
    
    for i, ses_label in enumerate(session_labels, start=1):
        for cell_name, maps_up, maps_down, subset_idx in cell_groups:
            
            # --- UP Trajectory Analysis ---
            s_up, _ = compute_cross_corrs(maps_up[0][subset_idx], maps_up[i][subset_idx], smooth=SMOOTH, sd=sd)
            mean_fr_up = maps_up[i][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
            
            for j, (sx, sy) in enumerate(zip(s_up[:, 0] * L / nfr, s_up[:, 1] * L / nfr)):
                data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 
                                    'Mean Fr': mean_fr_up[j], 'Direction': r'Up [0, pi)', 'Type': cell_name})
            
            # --- DOWN Trajectory Analysis ---
            s_down, _ = compute_cross_corrs(maps_down[0][subset_idx], maps_down[i][subset_idx], smooth=SMOOTH, sd=sd)
            mean_fr_down = maps_down[i][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)  
                          
            for j, (sx, sy) in enumerate(zip(s_down[:, 0] * L / nfr, s_down[:, 1] * L / nfr)):
                data_shifts.append({'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 
                                    'Mean Fr': mean_fr_down[j], 'Direction': r'Down [pi, 2 pi)', 'Type': cell_name})

    df_shifts = pd.DataFrame(data_shifts)
    
    #%% =========================================================================
    # 8. PLOTTING: Directional Comparisons
    # =========================================================================
    
    for fig_idx, (cell_name, maps_up, maps_down, subset_idx) in enumerate(cell_groups, start=7):
        plt.figure(fig_idx, figsize=(15, 5))
        
        # --- 1. SEPARATE FIRING RATE LOGIC ---
        if cell_name == 'Omni':
            # Omni: Firing rate based on Trajectory maps
            fr_0_up = maps_up[0][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
            fr_pi3_up = maps_up[1][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
            
            fr_0_down = maps_down[0][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
            fr_pi3_down = maps_down[1][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
            
            title_up = f'{cell_name} Cells (UP Traj)'
            title_down = f'{cell_name} Cells (DOWN Traj)'
            
        else:
            # Conjunctive: Firing rate based on Preferred Angle (using total maps)
            if cell_name == 'Conj1':
                total_maps = conj_maps1
                idx_up = up_HD_idx1
                idx_down = down_HD_idx1
            else: # Conj2
                total_maps = conj_maps2
                idx_up = up_HD_idx2
                idx_down = down_HD_idx2
                
            fr_0_up = total_maps[0]['rate maps'][idx_up].reshape((len(idx_up), -1)).mean(axis=1)
            fr_pi3_up = total_maps[1]['rate maps'][idx_up].reshape((len(idx_up), -1)).mean(axis=1)
            
            fr_0_down = total_maps[0]['rate maps'][idx_down].reshape((len(idx_down), -1)).mean(axis=1)
            fr_pi3_down = total_maps[1]['rate maps'][idx_down].reshape((len(idx_down), -1)).mean(axis=1)
            
            title_up = f'{cell_name} UP Neurons'
            title_down = f'{cell_name} DOWN Neurons'
    
        # --- 2. UP PLOT ---
        plt.subplot(1, 3, 1)
        W_up, P_up = wilcoxon(fr_0_up, fr_pi3_up)
        
        plt.boxplot([fr_0_up, fr_pi3_up])
        plt.xticks([1, 2], ['Session 0', r'Session $\pi/3$'])
        plt.title(f'{title_up}\nWilcoxon p={P_up:.2e}')
        plt.ylabel('Mean Firing Rate (Hz)')
        
        # --- 3. DOWN PLOT ---
        plt.subplot(1, 3, 2)
        W_down, P_down = wilcoxon(fr_0_down, fr_pi3_down)
        
        plt.boxplot([fr_0_down, fr_pi3_down])
        plt.xticks([1, 2], ['Session 0', r'Session $\pi/3$'])
        plt.title(f'{title_down}\nWilcoxon p={P_down:.2e}')
        plt.ylabel('Mean Firing Rate (Hz)')
        
        # --- 4. Y-SHIFTS PLOT ---
        plt.subplot(1, 3, 3)
        if (cell_name == "Omni") and (len(shifts_c2)==N_conj2):
            
            sh1 = 0*np.array(df_shifts[(df_shifts['Type'] == 'Conj2') & 
                        (df_shifts['Session'] == angles[1]) & 
                        (df_shifts['Direction'] == 'Up [0, pi)')]['Shift Y']).reshape((hd_modules,N_conj2//hd_modules)).mean(axis=0)
            sh2 = 0*np.array(df_shifts[(df_shifts['Type'] == 'Conj2') & 
                        (df_shifts['Session'] == angles[1]) & 
                        (df_shifts['Direction'] == 'Down [pi, 2 pi)')]['Shift Y']).reshape((hd_modules,N_conj2//hd_modules)).mean(axis=0)
        else:
            sh1, sh2 = 0, 0
        data1 = df_shifts[(df_shifts['Type'] == cell_name) & 
                          (df_shifts['Session'] == angles[1]) & 
                          (df_shifts['Direction'] == 'Up [0, pi)')]['Shift Y'] - sh1
                          
        data2 = df_shifts[(df_shifts['Type'] == cell_name) & 
                          (df_shifts['Session'] == angles[1]) & 
                          (df_shifts['Direction'] == 'Down [pi, 2 pi)')]['Shift Y'] - sh2
                          
        plt.boxplot([data1, data2])
        plt.axhline(0, color='k', linestyle='--')
        
        W_shift, P_shift = wilcoxon(data1, data2)
        plt.title(f'{cell_name} Shift Y (Session $\pi/3$)\nWilcoxon p={P_shift:.2e}')
        plt.xticks([1, 2], labels=['Up [0, pi)', 'Down [pi, 2 pi)'])
    
        plt.tight_layout()
        plt.show()

    # =========================================================================
    # 9. FINAL PARAMETER PRINT
    # =========================================================================
    print(f"l_asym = {parameters_complete['l_asym']:.2f}")
    print(f"l_torus = {parameters_complete['l_torus']:.2f}")
    print(f"input_std = {parameters_complete['input_std']:.2f}")
    if 'thresh' in parameters_complete:
        print(f"thresh = {parameters_complete['thresh']}")
    print(f"inclination_dir = {parameters_complete['inclination_dir']}")