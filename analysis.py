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
    find_spatial_shift_subpixel,
    compute_grid_metrics,
    sample_pair_correlations
)
#%%
if __name__ == "__main__":
    #%% =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    num = 0
    SMOOTH = True
    NEURON_IDX = 1
    fact = 1.67
    nfr = 50
    sd = 3
    i_0 = 36
    i_f = 63
        
    path2load = os.getcwd()  
    path2load = os.path.join(path2load, 'data', f'Simulation-spatial-num{num}')
    
    path2save = os.path.join(path2load,'Figures')
    
    
    path2load_0 = os.path.join(path2load, f'incl_ang{1}')
    
    with open(os.path.join(path2load_0, 'parameters_complete.pkl'), 'rb') as file:
        parameters_complete = pickle.load(file)
        
    L = parameters_complete['L']
    hd_modules = parameters_complete['hd_modules']
    dt = parameters_complete['dt']
    
    if not os.path.exists(os.path.join(path2load,'Figures')):
        os.mkdir(os.path.join(path2load,'Figures'))
    
    if len(os.listdir(path2load)) == 3:
        angles = ['0°', r'60°']
    else:
        angles = ['0°', r'30°', r'60°', "0°'"]
    # angles = ['0°', r'60°']
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
    # 3. COMPUTATIONS (Cross-Correlations, Shifts)
    # =========================================================================
    print("Computing Cross Correlations and Shifts...")
    
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

    #%% =========================================================================
    # 4. PLOTTING: Rate Maps and Cross Corrs
    # =========================================================================

    print("Generating standard plots...")
    
    bound_y0, bound_y1 = 15, 40    
    bound_x0, bound_x1 = 9, 34
        
    row1_axes = []
    
    fig = plt.figure(2, figsize=(20, 12))
    
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
            plt.figure(2)#, figsize=(16, 12))
            
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
    ax_end = row1_axes[-1] 
    
    # Create the line bridging ax_start to ax_end
    con = ConnectionPatch(xyA=(0, yred_cm), xyB=(50, yred_cm),
                      coordsA="data", coordsB="data",
                      axesA=ax_start, axesB=ax_end,
                      color=[.0, .0, .8], linestyle='--', linewidth=2)

    # Add the line to the FIGURE, not the subplot
    con.set_annotation_clip(False)
    fig.add_artist(con)      

    plt.figure(1)
    plt.savefig(os.path.join(path2load,'Figures','CrossCorrelograms_zoom.svg'),format='svg')

    plt.figure(2)
    plt.savefig(os.path.join(path2load,'Figures','SpatialMaps.svg'),format='svg')

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
    plt.ylim([-6.5,6.5])
    
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
    plt.ylim([-6.5,6.5])
    
        
    plt.tight_layout()
    
    plt.figure(5)
    plt.savefig(os.path.join(path2load,'Figures','Shifts_Y.svg'),format='svg')

    #%% =========================================================================
    # 6. DIRECTIONAL ANALYSIS & DATAFRAME CREATION
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
    
    for i, ses_label in enumerate(angles): # <--- NOW LOOPING OVER ALL ANGLES
        for cell_name, maps_up, maps_down, subset_idx in cell_groups:
            
            # --- UP Trajectory Analysis ---
            _, AC_up = compute_cross_corrs(maps_up[i][subset_idx], maps_up[i][subset_idx], smooth=SMOOTH, sd=sd)
            
            if i == 0:
                # Session 0: No shift, assign NaNs
                s_up_x = [np.nan] * len(subset_idx)
                s_up_y = [np.nan] * len(subset_idx)
            else:
                # Session > 0: Compute spatial shift relative to baseline
                s_up, _ = compute_cross_corrs(maps_up[0][subset_idx], maps_up[i][subset_idx], smooth=SMOOTH, sd=sd)
                s_up_x = s_up[:, 0] * L / nfr
                s_up_y = s_up[:, 1] * L / nfr
            
            for j, (sx, sy) in enumerate(zip(s_up_x, s_up_y)):
                score, spacing = compute_grid_metrics(AC_up[j], bin_size=L/nfr) 
                
                data_shifts.append({'Cell Index': idx_analyze[j], 
                                    'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 
                                    'Grid Score': score, 'Grid Spacing': spacing, 
                                    'Direction': r'Up', 'Type': cell_name})
            
            # --- DOWN Trajectory Analysis ---
            _, AC_down = compute_cross_corrs(maps_down[i][subset_idx], maps_down[i][subset_idx], smooth=SMOOTH, sd=sd)
            
            if i == 0:
                s_down_x = [np.nan] * len(subset_idx)
                s_down_y = [np.nan] * len(subset_idx)
            else:
                s_down, _ = compute_cross_corrs(maps_down[0][subset_idx], maps_down[i][subset_idx], smooth=SMOOTH, sd=sd)
                s_down_x = s_down[:, 0] * L / nfr
                s_down_y = s_down[:, 1] * L / nfr
                          
            for j, (sx, sy) in enumerate(zip(s_down_x, s_down_y)):
                score, spacing = compute_grid_metrics(AC_down[j], bin_size=L/nfr) 
                
                data_shifts.append({'Cell Index': idx_analyze[j], 
                                    'Session': ses_label, 'Shift X': sx, 'Shift Y': sy, 
                                    'Grid Score': score, 'Grid Spacing': spacing, 
                                    'Direction': r'Down', 'Type': cell_name})

    df_shifts = pd.DataFrame(data_shifts)
#%%
    from scipy.stats import kruskal


    # DIRECTIONAL COMPARISONS
    plt.figure(7,figsize=(10,5))
    plt.subplot(1, 2, 1)
    idx2plot = min([len(angles),3])-1

    data1 = df_shifts[(df_shifts['Type'] == cell_name) & 
                      (df_shifts['Session'] == angles[idx2plot]) & 
                      (df_shifts['Direction'] == 'Up')]['Shift Y'] 
                      
    data2 = df_shifts[(df_shifts['Type'] == cell_name) & 
                      (df_shifts['Session'] == angles[idx2plot]) & 
                      (df_shifts['Direction'] == 'Down')]['Shift Y']
                      
    plt.boxplot([data1, data2])
    plt.axhline(0, color='k', linestyle='--')

    W_shift, P_shift = wilcoxon(data1, data2)
    plt.title(f'Shift Y (Session {angles[idx2plot]})\n Wilcoxon p={P_shift:.2e}')
    plt.xticks([1, 2], labels=['Up', 'Down'])

    # CORRELATION ANALYSIS
    # Generate distributions
    _, acg_0_deg = compute_cross_corrs(
        omni_maps[0]['rate maps'], omni_maps[0]['rate maps'], smooth=SMOOTH, sd=sd)
    _, acg_60_deg = compute_cross_corrs(
        omni_maps[idx2plot]['rate maps'], omni_maps[idx2plot]['rate maps'], smooth=SMOOTH, sd=sd)

    dist_0_0, idx_00 = sample_pair_correlations(acg_0_deg, acg_0_deg, n_pairs=350)
    dist_60_60, idx_6060 = sample_pair_correlations(acg_60_deg, acg_60_deg, n_pairs=350)
    dist_0_60, idx_060 = sample_pair_correlations(acg_0_deg, acg_60_deg, n_pairs=350)

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

    plt.savefig(os.path.join(path2load,'Figures','UP_DOWN-Correlation.svg'),format='svg')

    plt.show()
    #%% =========================================================================
    # 10. DATAFRAMES: Firing Rates and Friedman/Dunn Tests
    # =========================================================================
    import scikit_posthocs as sp

    print("Generating Friedman Tests and Dunn Post-Hoc DataFrame...")

    # 1. Compute Friedman tests for all shift types
    friedman_results = []

    shift_dict = {
        'Omni_X': shifts_x,
        'Omni_Y': shifts_y
    }

    for shift_name, shift_list in shift_dict.items():
        # Friedman test requires at least 2 groups (i.e., > 2 angular sessions)
        if len(shift_list) > 1: 
            stat, p_val_f = friedmanchisquare(*shift_list)
            friedman_results.append({
                'Shift_Type': shift_name,
                'Friedman_Stat': stat,
                'p_value': p_val_f,
                'Significant': p_val_f < 0.05
            })
        else:
            friedman_results.append({
                'Shift_Type': shift_name,
                'Friedman_Stat': np.nan,
                'p_value': np.nan,
                'Significant': False
            })
            
    df_friedman = pd.DataFrame(friedman_results)

    # 2. Compute Post hoc Dunn's test strictly for Omni Shift Y
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

    #%% =========================================================================
    # 11. DATA SAVING: Consolidate Output Data and Save to Disk
    # =========================================================================
    print("Structuring and Saving all spatial shifts, correlations, and statistics...")

    # 1. Expand df_shifts to include 'Total' (Omni) directional shifts
    omni_shifts_data = []
    
    # <--- NOW LOOPING OVER ALL ANGLES --->
    for idx_ses, ses_label in enumerate(angles):
        
        # AC is computed for the current session (idx_ses)
        _, AC_omni = compute_cross_corrs(omni_maps[idx_ses]['rate maps'][idx_analyze], 
                                         omni_maps[idx_ses]['rate maps'][idx_analyze], smooth=SMOOTH, sd=sd) 
        
        if idx_ses == 0:
            # Baseline session: shifts are NaN
            sx_list = [np.nan] * len(idx_analyze)
            sy_list = [np.nan] * len(idx_analyze)
        else:
            # Future sessions: shift_x and shift_y lists correspond to sessions beyond 0°
            # So index `idx_ses - 1` maps correctly (e.g., idx_ses 1 pulls from index 0)
            sx_list = shifts_x[idx_ses - 1]
            sy_list = shifts_y[idx_ses - 1]
            
        for j, (sx, sy) in enumerate(zip(sx_list, sy_list)): 
            
            score, spacing = compute_grid_metrics(AC_omni[j], bin_size=L/nfr) 
            
            omni_shifts_data.append({
                'Cell Index': idx_analyze[j], 
                'Session': ses_label,
                'Shift X': sx,
                'Shift Y': sy,
                'Grid Score': score,     
                'Grid Spacing': spacing, 
                'Direction': 'Total',
                'Type': 'Omni'
            })
            
    df_all_shifts = pd.concat([df_shifts, pd.DataFrame(omni_shifts_data)], ignore_index=True)
    
    # 2. Consolidate Pair Correlations with True Cell Indices
    df_corr = pd.concat([
        pd.DataFrame({
            'Comparison': '0° vs 0°', 
            'Pearson_r': dist_0_0,
            'Cell_1': idx_analyze[idx_00[:, 0]],    # Maps index 'i' to true Cell ID
            'Cell_2': idx_analyze[idx_00[:, 1]]     # Maps index 'j' to true Cell ID
        }),
        pd.DataFrame({
            'Comparison': '60° vs 60°', 
            'Pearson_r': dist_60_60,
            'Cell_1': idx_analyze[idx_6060[:, 0]], 
            'Cell_2': idx_analyze[idx_6060[:, 1]]
        }),
        pd.DataFrame({
            'Comparison': '0° vs 60°', 
            'Pearson_r': dist_0_60,
            'Cell_1': idx_analyze[idx_060[:, 0]], 
            'Cell_2': idx_analyze[idx_060[:, 1]]
        })
    ], ignore_index=True)

    # 3. Consolidate scalar statistics into a single Summary DataFrame
    stats_summary = [
        {
            'Test': 'Wilcoxon (Up vs Down Y-Shift)',
            'Group/Session': angles[idx2plot],
            'Statistic_Name': 'W',
            'Statistic_Value': W_shift,
            'p_value': P_shift
        },
        {
            'Test': 'Kruskal-Wallis (Correlations)',
            'Group/Session': '0° vs 0°, 60° vs 60°, 0° vs 60°',
            'Statistic_Name': 'H',
            'Statistic_Value': h_stat,
            'p_value': p_val
        }
    ]
    
    # Append Friedman test results to the statistical summary
    for _, row in df_friedman.iterrows():
        stats_summary.append({
            'Test': f"Friedman ({row['Shift_Type']})",
            'Group/Session': 'All Sessions',
            'Statistic_Name': 'Chi-Square',
            'Statistic_Value': row['Friedman_Stat'],
            'p_value': row['p_value']
        })
    df_stats_summary = pd.DataFrame(stats_summary)

    # 4. Save data to Excel and Pickle for flexibility
    excel_path = os.path.join(path2load, 'Figures', 'Consolidated_Results.xlsx')
    pickle_path = os.path.join(path2load, 'Figures', 'Consolidated_Results.pkl')
    
    # Export to Excel with multiple organized sheets
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_all_shifts.to_excel(writer, sheet_name='All_Shifts', index=False)
            df_corr.to_excel(writer, sheet_name='Correlations', index=False)
            df_stats_summary.to_excel(writer, sheet_name='Statistics_Summary', index=False)
            if not df_dunn_omni_y.empty:
                df_dunn_omni_y.to_excel(writer, sheet_name='Dunns_PostHoc_OmniY', index=True)
        print(f"Tabular data effectively saved to Excel at: {excel_path}")
    except ModuleNotFoundError:
        print("Excel dependencies not met (install openpyxl/xlsxwriter). Falling back to Pickle only.")

    # Export a comprehensive dictionary to Pickle
    saved_data_dict = {
        'all_shifts_df': df_all_shifts,
        'correlations_df': df_corr,
        'statistics_df': df_stats_summary,
        'dunn_posthoc_df': df_dunn_omni_y
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(saved_data_dict, f)
        
    print(f"Data objects correctly saved to Pickle at: {pickle_path}")