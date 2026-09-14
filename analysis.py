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
from scipy.stats import friedmanchisquare, wilcoxon, mannwhitneyu

from matplotlib.patches import ConnectionPatch
from utils import (
    load_and_compute_maps_from_sparse,
    compute_cross_corrs,
    compute_directional_rate_maps,
    find_spatial_shift_subpixel
)
#%%
if __name__ == "__main__":
    #%% =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    num = 8
    SMOOTH = True
    NEURON_IDX = 34
    fact = 1.67
    nfr = 50
    sd = 3
    i_0 = 36
    i_f = 63
       
    path2load = os.path.split(os.getcwd())[0]    
    path2load = os.path.join(path2load, f'Simulation-spatial-num{num}')
    
    angles = ['0°', r'30°', r'60°', "0°'"]
    
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
    omni_maps, traj_list = load_and_compute_maps_from_sparse(
        num=num, sim_folder=path2load, nx=nfr, ny=nfr
    )
        
    nfr = omni_maps[0]['rate maps'].shape[1]
    N = omni_maps[0]['rate maps'].shape[0]
    
    
    #%% =========================================================================
    # 3. COMPUTATIONS (Mean FRs, Cross-Correlations, Shifts)
    # =========================================================================
    print("Computing Cross Correlations and Shifts...")
    mean_fr_omni = []
    CrossCorr_omni = []
    shifts_x, shifts_y = [], []
    
    idx_discard = np.array([])
    
    for i in range(len(angles)):
        
        # Cross-Correlations and Shifts (relative to baseline session 0)
        shifts_omni, CC = compute_cross_corrs(
            omni_maps[0]['rate maps'], omni_maps[i]['rate maps'], smooth=SMOOTH, sd=sd)
        CrossCorr_omni.append(CC)
        
        idx = np.where(np.isnan(CC.mean(axis=2).mean(axis=1)))[0]
        idx_discard = np.concat((idx_discard,idx))
        
        
        if i>0:
            shifts_x.append(shifts_omni[:,0])
            shifts_y.append(shifts_omni[:,1])
    
    idx_discard = np.unique(idx_discard)
    idx_analyze = np.array([i for i in range(N) if i not in idx_discard])
    
    for i in range(len(angles)):
        shape = omni_maps[i]['rate maps'].shape
        if i>0:
            shifts_x[i-1] = shifts_x[i-1][idx_analyze]
            shifts_y[i-1] = shifts_y[i-1][idx_analyze]
        # Mean Firing Rates
        mean_fr_omni.append(omni_maps[i]['rate maps'][idx_analyze].reshape((len(idx_analyze), shape[1]*shape[2])).mean(axis=1))
        
    #%% =========================================================================
    # 4. PLOTTING: Rate Maps and Cross Corrs
    # =========================================================================

    print("Generating standard plots...")
    
    bound_x0, bound_x1 = 22, 50
    bound_y0, bound_y1 = 0, 27

    row1_axes = []
    
    fig = plt.figure(2, figsize=(16, 12))
    
    for i in range(len(angles)):
        if i > 0:
            # Figure 1: Mean Cross Correlograms
            plt.figure(1, figsize=(15, 10))
            plt.subplot(2, 4, i)
            CC_MEAN = np.nanmean(CrossCorr_omni[i][idx_analyze,:],axis=0)
            plt.imshow(CC_MEAN, cmap='jet')
            plt.axis('off')
            plt.title(f'Mean CC: {angles[i]}')
            
            plt.subplot(2, 4, i+4)
            plt.imshow(CC_MEAN[i_0:i_f, i_0:i_f], cmap='jet')
            DeltaP = find_spatial_shift_subpixel(CC_MEAN, n=7, search_radius_pixels=7)
            plt.title(rf'$\Delta_P$ = {DeltaP[0]:.2f}')
            plt.axis('off')
            
        # Figure 2: Trajectory Overlay - Omni
        if len(omni_maps[i]['spiking maps']['time_idx']) == 0:
            print('No Omni maps!')
        else:
            plt.figure(2, figsize=(16, 12))
            
            # Overlay Map
            ax = plt.subplot(3, 5, i+1)
            row1_axes.append(ax) # Save the axis reference for later
            
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
            
            plt.axis('off')
            plt.title(f'Omni Traj: {angles[i]}')
            
            # Raw Rate Map
            plt.subplot(3, 5, i+6)
            RM_omni = omni_maps[i]['rate maps'][NEURON_IDX]
            plt.imshow(RM_omni, cmap='jet', origin='lower')
            plt.axis('off')
            plt.title(f'Max: {RM_omni.max():.2f} Hz, \n Mean: {RM_omni.mean():.2f} Hz')
            
            # Single Neuron Cross Corr
            plt.subplot(3, 5, i+11)
            _, CC_OMNI_single = compute_cross_corrs(
                omni_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                omni_maps[i]['rate maps'][NEURON_IDX:NEURON_IDX+1],
                smooth=SMOOTH, sd=sd)
            plt.imshow(CC_OMNI_single[0], cmap='jet', origin='lower')
            plt.axis('off')
    
    # Select the first subplot (index 0) and the fourth subplot (index 3)
    ax_start = row1_axes[0]
    ax_end = row1_axes[3] 
    
    # Create the line bridging ax_start to ax_end
    con = ConnectionPatch(xyA=(0, yred_cm), xyB=(50, yred_cm),
                      coordsA="data", coordsB="data",
                      axesA=ax_start, axesB=ax_end,
                      color=[.0, .0, .8], linestyle='--', linewidth=2)

    # Add the line to the FIGURE, not the subplot
    con.set_annotation_clip(False)
    fig.add_artist(con)      

    plt.figure(1)
    plt.savefig(os.path.join(path2load,'Figures/CrossCorrelograms_zoom.svg'),format='svg')

    plt.figure(2)
    plt.savefig(os.path.join(path2load,'Figures/SpatialMaps.svg'),format='svg')

    plt.show()

    #%% =========================================================================
    # 5. PLOTTING: Spatial Shifts (Boxplots)
    # =========================================================================
    plt.figure(5, figsize=(10, 5))
    
    # Omnidirectional Cells
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=shifts_x, fill=False)
    plt.plot([-1, len(shifts_x)], [0, 0], '--k')
    if len(shifts_x) > 2:
        F, P = friedmanchisquare(*shifts_x)
        plt.title(f'Omni X Shift (F p-val={P:.2e})')
    elif len(shifts_x) > 1:
        U, P = mannwhitneyu(*shifts_x)
        plt.title(f'Omni X Shift (U p-val={P:.2e})')
    plt.xlim([-.5, len(shifts_x) - .5])
    plt.xticks(range(len(angles)-1), labels=angles[1:])
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=shifts_y, fill=False)
    plt.plot([-1, len(shifts_y)], [0, 0], '--k')
    if len(shifts_y) > 2:
        F, P = friedmanchisquare(*shifts_y)
        plt.title(f'Omni Y Shift (F p-val={P:.2e})')
    elif len(shifts_y) > 1:
       U, P = mannwhitneyu(*shifts_y)
       plt.title(f'Omni Y Shift (U p-val={P:.2e})')
    plt.xlim([-.5, len(shifts_y)-.5])
    plt.xticks(range(len(angles)-1), labels=angles[1:])
    
       
    plt.tight_layout()
    
    plt.figure(5)
    plt.savefig(os.path.join(path2load,'Figures/Shifts_Y.svg'),format='svg')

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
    
    plt.ylim([0,1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(path2load,'Figures/Mean_fr.svg'),format='svg')
    plt.show()
    
    #%% =========================================================================
    # 7. DIRECTIONAL ANALYSIS & DATAFRAME CREATION
    # =========================================================================
    print("Computing Directional Rate Maps and Shifts...")
    omni_up, omni_down = [], []

    for i in range(len(angles)):
        is_up = (traj_list[i][:, 2] >= 0) & (traj_list[i][:, 2] < np.pi)
        is_down = (traj_list[i][:, 2] >= np.pi) & (traj_list[i][:, 2] < 2 * np.pi)

        # Compute Directional Maps for Omni
        omni_up.append(compute_directional_rate_maps(
            omni_maps[i]['spiking maps'], traj_list[i], is_up, N, L, dt, nx=nfr, ny=nfr))
        omni_down.append(compute_directional_rate_maps(
            omni_maps[i]['spiking maps'], traj_list[i], is_down, N, L, dt, nx=nfr, ny=nfr))
            
        omni_up[i] = omni_up[i][idx_analyze]
        omni_down[i] = omni_down[i][idx_analyze]
        
    data_shifts = []
    session_labels = angles[1:] 
    
    # Bundle the populations to avoid repeating code
    cell_groups = [
        ('Omni', omni_up, omni_down, np.arange(len(idx_analyze))),
        
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

#%%
from scipy.stats import kruskal, pearsonr

def get_spatial_correlation(acg1, acg2):
    """Computes the Pearson correlation between two 2D autocorrelograms."""
    # Mask out NaNs (e.g., circular borders in grid cell autocorrelograms)
    valid = ~np.isnan(acg1) & ~np.isnan(acg2)
    
    # If the overlap is too small, return NaN to avoid spurious correlations
    if np.sum(valid) < 10: 
        return np.nan
        
    r, _ = pearsonr(acg1[valid], acg2[valid])
    return r

def sample_pair_correlations(list_a, list_b, n_pairs=350, seed=0):
    """Randomly samples unique cell pairs and computes their correlation."""
    rng = np.random.default_rng(seed)
    
    # 1. Generate all possible unique pair indices
    if list_a is list_b:
        # Within-session: extract upper triangle indices (no self-pairs, no duplicates)
        idx_pairs = np.array(np.triu_indices(len(list_a), k=1)).T
    else:
        # Cross-session: cartesian product of all indices
        idx_pairs = np.array(np.meshgrid(np.arange(len(list_a)), 
                                         np.arange(len(list_b)))).T.reshape(-1, 2)
        
    # 2. Sample N pairs without replacement
    max_possible_pairs = len(idx_pairs)
    sample_size = min(n_pairs, max_possible_pairs)
    sampled_indices = rng.choice(idx_pairs, size=sample_size, replace=False)
    
    # 3. Compute correlations
    correlations = []
    for i, j in sampled_indices:
        r = get_spatial_correlation(list_a[i], list_b[j])
        if not np.isnan(r):
            correlations.append(r)
            
    return np.array(correlations)


# DIRECTIONAL COMPARISONS
plt.figure(7,figsize=(10,5))
plt.subplot(1, 2, 1)

data1 = df_shifts[(df_shifts['Type'] == cell_name) & 
                  (df_shifts['Session'] == angles[1]) & 
                  (df_shifts['Direction'] == 'Up [0, pi)')]['Shift Y'] 
                  
data2 = df_shifts[(df_shifts['Type'] == cell_name) & 
                  (df_shifts['Session'] == angles[1]) & 
                  (df_shifts['Direction'] == 'Down [pi, 2 pi)')]['Shift Y']
                  
plt.boxplot([data1, data2])
plt.axhline(0, color='k', linestyle='--')

W_shift, P_shift = wilcoxon(data1, data2)
plt.title(f'Shift Y (Session 60°)\n Wilcoxon p={P_shift:.2e}')
plt.xticks([1, 2], labels=['Up [0, pi)', 'Down [pi, 2 pi)'])

# CORRELATION ANALYSIS
# Generate distributions
_, acg_0_deg = compute_cross_corrs(
    omni_maps[3]['rate maps'], omni_maps[3]['rate maps'], smooth=SMOOTH, sd=sd)
_, acg_60_deg = compute_cross_corrs(
    omni_maps[2]['rate maps'], omni_maps[2]['rate maps'], smooth=SMOOTH, sd=sd)

dist_0_0 = sample_pair_correlations(acg_0_deg, acg_0_deg, n_pairs=350)
dist_60_60 = sample_pair_correlations(acg_60_deg, acg_60_deg, n_pairs=350)
dist_0_60 = sample_pair_correlations(acg_0_deg, acg_60_deg, n_pairs=350)

# Kruskal-Wallis H-test
h_stat, p_val = kruskal(dist_0_0, dist_60_60, dist_0_60)
print(f"Kruskal-Wallis test: H = {h_stat:.2f}, p = {p_val:.2f}")


plt.subplot(1, 2, 2)
# Visualization
plt.boxplot([dist_0_0, dist_60_60, dist_0_60])
plt.xticks([1,2,3],labels=['0° vs 0°', '60° vs 60°', '0° vs 60°'])
plt.yticks([0.6,0.7,0.8,0.9,1.])
plt.ylabel('Pearson Correlation (r)')
plt.title(f"Population Analysis \n Kruskal-Wallis test: H = {h_stat:.2f}, p = {p_val:.2f}")

plt.savefig(os.path.join(path2load,'Figures/UP_DOWN-Correlation.svg'),format='svg')

plt.show()
    #%% =========================================================================
    # 10. DATAFRAMES: Firing Rates and Friedman/Dunn Tests
    # =========================================================================
import scikit_posthocs as sp
print("Generating Firing Rate DataFrame...")

fr_data = []

# 1. Loop over all sessions/angles to extract Firing Rates
for i, ang in enumerate(angles):
    # Extract TOTAL Firing Rates (computed in Section 3)
    for n_idx, fr in enumerate(mean_fr_omni[i]):
        fr_data.append({'Neuron_ID': n_idx, 'Cell_Type': 'Omni', 'Angle': ang, 'Direction': 'Total', 'Firing_Rate': fr})
    
    # Extract UP and DOWN Firing Rates (using maps_up/maps_down from Section 7)
    for cell_name, maps_up, maps_down, subset_idx in cell_groups:
        fr_up = maps_up[i][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
        fr_down = maps_down[i][subset_idx].reshape((len(subset_idx), -1)).mean(axis=1)
        
        for j, fr in enumerate(fr_up):
            # Here subset_idx is correct because maps_up was already subsetted
            fr_data.append({'Neuron_ID': subset_idx[j], 'Cell_Type': cell_name, 'Angle': ang, 'Direction': 'Up', 'Firing_Rate': fr})
        for j, fr in enumerate(fr_down):
            fr_data.append({'Neuron_ID': subset_idx[j], 'Cell_Type': cell_name, 'Angle': ang, 'Direction': 'Down', 'Firing_Rate': fr})
            
df_firing_rates = pd.DataFrame(fr_data)
print(f"Created df_firing_rates with {len(df_firing_rates)} rows.")


print("Generating Friedman Tests and Dunn Post-Hoc DataFrame...")

# 2. Compute Friedman tests for all shift types
friedman_results = []

shift_dict = {
    'Omni_X': shifts_x,
    'Omni_Y': shifts_y
}

for shift_name, shift_list in shift_dict.items():
    # Friedman test requires at least 2 groups (i.e., > 2 angular sessions)
    if len(shift_list) > 1: 
        stat, p_val = friedmanchisquare(*shift_list)
        friedman_results.append({
            'Shift_Type': shift_name,
            'Friedman_Stat': stat,
            'p_value': p_val,
            'Significant': p_val < 0.05
        })
    else:
        friedman_results.append({
            'Shift_Type': shift_name,
            'Friedman_Stat': np.nan,
            'p_value': np.nan,
            'Significant': False
        })
        
df_friedman = pd.DataFrame(friedman_results)

# 3. Compute Post hoc Dunn's test strictly for Omni Shift Y
if len(shifts_y) > 1:
    # Run Dunn's test with Bonferroni correction
    df_dunn_omni_y = sp.posthoc_dunn(shifts_y, p_adjust='bonferroni')
    
    # Format the matrix headers/indexes to match your session labels
    session_labels = angles[1:]
    df_dunn_omni_y.columns = session_labels
    df_dunn_omni_y.index = session_labels
    
    print("\nOmni Y-Shift Dunn's Test Results:")
    print(df_dunn_omni_y)
else:
    df_dunn_omni_y = pd.DataFrame()
    print("\nNot enough sessions to run a post hoc Dunn's test on Omni Y-shifts.")