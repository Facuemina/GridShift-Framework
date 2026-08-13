# -*- coding: utf-8 -*-
"""
Created for batch processing simulations 1 to 5.
Extracts FR, Spatial Shifts (Total, Up, Down), and Grid Metrics.
Outputs to a CSV.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import rotate, maximum_filter

from src.utils import (
    load_and_compute_maps_from_sparse,
    compute_cross_corrs,
    compute_directional_rate_maps
)

def compute_grid_metrics(autocorr):
    """
    Computes grid score and spacing based on the spatial autocorrelogram.
    - Spacing: Mean distance from the central peak to the 6 nearest peaks.
    - Score: Mean(corr at 60, 120) - Mean(corr at 30, 90, 150).
    """
    # 1. Find local peaks using scipy's maximum_filter
    min_distance = 3
    window_size = 2 * min_distance + 1
    
    # Create a boolean mask of local maxima
    local_max_mask = (maximum_filter(autocorr, size=window_size) == autocorr)
    
    # Filter out flat areas or background noise by ensuring peaks are above the mean
    local_max_mask &= (autocorr > np.mean(autocorr))
    
    # Get coordinates of the peaks
    peaks = np.argwhere(local_max_mask)

    if len(peaks) < 7:
        return np.nan, np.nan

    center = np.array(autocorr.shape) // 2
    distances = np.linalg.norm(peaks - center, axis=1)

    # Sort peaks by distance to center
    sorted_idx = np.argsort(distances)
    
    # The closest is the center itself (dist ~ 0). The next 6 are the inner hexagon.
    if len(sorted_idx) < 7:
        return np.nan, np.nan

    hexagon_peaks = peaks[sorted_idx[1:7]]
    hexagon_distances = distances[sorted_idx[1:7]]
    spacing = np.mean(hexagon_distances)

    # Create annulus mask to isolate the inner peaks for rotation correlation
    inner_radius = hexagon_distances.min() * 0.5
    outer_radius = hexagon_distances.max() * 1.5
    
    Y, X = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)

    masked_ac = np.copy(autocorr)
    masked_ac[~mask] = 0

    # Compute correlations with rotated versions
    corrs = {}
    for angle in [30, 60, 90, 120, 150]:
        rotated_ac = rotate(masked_ac, angle, reshape=False)
        # Correlate only the valid overlapping masked regions
        valid = mask & (rotated_ac != 0)
        if np.sum(valid) > 1:
            r = np.corrcoef(masked_ac[valid], rotated_ac[valid])[0, 1]
        else:
            r = np.nan
        corrs[angle] = r

    # Compute grid score
    try:
        score = np.nanmean([corrs[60], corrs[120]]) - np.nanmean([corrs[30], corrs[90], corrs[150]])
    except:
        score = np.nan

    return score, spacing

if __name__ == "__main__":
    #%% =========================================================================
    # 1. CONFIGURATION
    # =========================================================================
    SMOOTH = False
    fact = 1.67
    nfr_init = int(30 * fact)
    sd = int(2 * fact)
    
    # Using clean string angles for the CSV
    angle_labels = ['0°', '30°', '60°', "0°'"]
    
    all_data = [] # Will hold dictionaries for each neuron's observation
    base_path = os.path.split(os.getcwd())[0]  

    #%% =========================================================================
    # 2. BATCH PROCESSING LOOP
    # =========================================================================
    for num in range(1, 6):
        print(f"\nProcessing Rat (num) = {num}...")
        path2load = os.path.join(base_path, f'Simulation-2layer-num{num}')
        
        # Guard in case a folder doesn't have all angles
        current_angles = angle_labels[:len(os.listdir(path2load))]
        
        path2load_0 = os.path.join(path2load, f'incl_ang{1}')
        with open(os.path.join(path2load_0, 'parameters_complete.pkl'), 'rb') as file:
            parameters_complete = pickle.load(file)
            
        L = parameters_complete['L']
        dt = parameters_complete['dt']

        print("  Loading data and computing rate maps...")
        omni_maps, conj_maps1, conj_maps2, traj_list = load_and_compute_maps_from_sparse(
            num=num, sim_folder=path2load, nx=nfr_init, ny=nfr_init
        )
            
        nfr = omni_maps[0]['rate maps'].shape[1]
        N_omni = omni_maps[0]['rate maps'].shape[0]
        N_conj1 = conj_maps1[0]['rate maps'].shape[0]
        N_conj2 = conj_maps2[0]['rate maps'].shape[0]

        # Group populations for simplified iteration
        cell_groups = [
            ('Omni', N_omni, omni_maps),
            ('Conj1', N_conj1, conj_maps1),
            ('Conj2', N_conj2, conj_maps2)
        ]

        print("  Computing metrics for all sessions and cells...")
        for i, ses_label in enumerate(current_angles):
            
            is_up = (traj_list[i][:, 2] >= 0) & (traj_list[i][:, 2] < np.pi)
            is_down = (traj_list[i][:, 2] >= np.pi) & (traj_list[i][:, 2] < 2 * np.pi)

            for cell_name, N_cells, maps_list in cell_groups:
                
                # Compute Directional Maps
                maps_up = compute_directional_rate_maps(
                    maps_list[i]['spiking maps'], traj_list[i], is_up, N_cells, L, dt, nx=nfr, ny=nfr)
                maps_down = compute_directional_rate_maps(
                    maps_list[i]['spiking maps'], traj_list[i], is_down, N_cells, L, dt, nx=nfr, ny=nfr)
                
                # Fetch baseline maps for shift calculations
                baseline_maps = maps_list[0]['rate maps']
                baseline_up = compute_directional_rate_maps(
                    maps_list[0]['spiking maps'], traj_list[0], (traj_list[0][:, 2] >= 0) & (traj_list[0][:, 2] < np.pi), N_cells, L, dt, nx=nfr, ny=nfr)
                baseline_down = compute_directional_rate_maps(
                    maps_list[0]['spiking maps'], traj_list[0], (traj_list[0][:, 2] >= np.pi) & (traj_list[0][:, 2] < 2 * np.pi), N_cells, L, dt, nx=nfr, ny=nfr)

                # Current session rate maps
                curr_maps = maps_list[i]['rate maps']
                
                # 1. Shifts against 0° session (Total, Up, Down)
                if i == 0:
                    shifts_tot = np.zeros((N_cells, 2))
                    shifts_up = np.zeros((N_cells, 2))
                    shifts_down = np.zeros((N_cells, 2))
                else:
                    s_tot, _ = compute_cross_corrs(baseline_maps, curr_maps, smooth=SMOOTH, sd=sd)
                    s_up, _ = compute_cross_corrs(baseline_up, maps_up, smooth=SMOOTH, sd=sd)
                    s_down, _ = compute_cross_corrs(baseline_down, maps_down, smooth=SMOOTH, sd=sd)
                    
                    shifts_tot = s_tot * L / nfr
                    shifts_up = s_up * L / nfr
                    shifts_down = s_down * L / nfr

                # 2. Mean Firing Rates
                mean_fr_tot = curr_maps.reshape((N_cells, -1)).mean(axis=1)

                # 3. Autocorrelograms for Grid Score & Spacing
                _, autocorrelograms = compute_cross_corrs(curr_maps, curr_maps, smooth=SMOOTH, sd=sd)

                # Populate data for each neuron
                for neuron_idx in range(N_cells):
                    
                    grid_score, spacing = compute_grid_metrics(autocorrelograms[neuron_idx])
                    
                    all_data.append({
                        'Rat': num,
                        'Session_Angle': ses_label,
                        'Cell_Type': cell_name,
                        'Neuron_ID': neuron_idx,
                        'Mean_FR': mean_fr_tot[neuron_idx],
                        'Shift_X_Total': shifts_tot[neuron_idx, 0],
                        'Shift_Y_Total': shifts_tot[neuron_idx, 1],
                        'Shift_X_Up': shifts_up[neuron_idx, 0],
                        'Shift_Y_Up': shifts_up[neuron_idx, 1],
                        'Shift_X_Down': shifts_down[neuron_idx, 0],
                        'Shift_Y_Down': shifts_down[neuron_idx, 1],
                        'Grid_Score': grid_score,
                        'Spacing': spacing * (L / nfr) # Convert spacing to spatial metric if needed
                    })

    #%% =========================================================================
    # 3. EXPORT TO CSV
    # =========================================================================
    print("\nBatch processing complete. Exporting to CSV...")
    df_results = pd.DataFrame(all_data)
    output_path = os.path.join(base_path, 'all_rats_metrics.csv')
    df_results.to_csv(output_path, index=False)
    
    print(f"File successfully saved to: {output_path}")
    print(df_results.head())