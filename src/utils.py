# -*- coding: utf-8 -*-
"""
utils.py

Helper functions:
    - 2D trajectory generator
    - Netwrok connectivity builders
    - Firing rate map computations
    - Spatial shift computations
"""

import os
import pickle
import numpy as np
import numpy.linalg as la
import jax.numpy as jnp
from jax import jit, random, lax
from numbers import Number
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
from tqdm import tqdm

def generate2D_pos(seed, steps, Lx, Ly, v, sigma_theta, delta_t, periodic=False):
    two_pi = 2 * jnp.pi
    
    key = random.PRNGKey(seed)
    key, k1, k2, k3 = random.split(key, 4)
    mydir = random.uniform(k1, minval=0.0, maxval=two_pi)
    x = random.uniform(k2, minval=0.0, maxval=Lx)
    y = random.uniform(k3, minval=0.0, maxval=Ly)
    if isinstance(v, Number):
        v = jnp.ones(steps) * v
        
    if periodic:
        @jit
        def out_of_bounds(xp, yp):
            return False
    else:
        @jit
        def out_of_bounds(xp, yp):
            return (xp < 0) | (xp > Lx) | (yp < 0) | (yp > Ly)

    @jit
    def step(carry, v_i):
        x, y, mydir, key = carry
        
        key, k_ang = random.split(key)

        mydir = jnp.mod(
            mydir + random.normal(k_ang) * sigma_theta * jnp.sqrt(delta_t),
            two_pi,
        )
        x_new = x + v_i * jnp.cos(mydir) * delta_t
        y_new = y + v_i * jnp.sin(mydir) * delta_t

        def cond_fn(state):
            x_t, y_t, dir_t, key_t, count = state
            return out_of_bounds(x_t, y_t)

        def body_fn(state):
            x_t, y_t, dir_t, key_t, count = state
            x_t = x_t - v_i * jnp.cos(dir_t) * delta_t
            y_t = y_t - v_i * jnp.sin(dir_t) * delta_t
            key_t, ksub = random.split(key_t)
            dir_t = jnp.mod(
                dir_t
                + random.normal(ksub)
                * sigma_theta
                * jnp.sqrt(delta_t)
                / 3.0,
                two_pi,
            )
            x_t = x_t + v_i * jnp.cos(dir_t) * delta_t
            y_t = y_t + v_i * jnp.sin(dir_t) * delta_t
            return (x_t, y_t, dir_t, key_t, count + 1)

        init_state = (x_new, y_new, mydir, key, jnp.array(0))
        x_final, y_final, dir_final, key_final, _ = lax.while_loop(cond_fn, body_fn, init_state)

        return (x_final, y_final, dir_final, key_final), (x_final, y_final, dir_final)

    init_carry = (x, y, mydir, key)
    _, traj = lax.scan(step, init_carry, v, length=steps)
    return traj

def map2torus_fn(x, y, l, phase=jnp.array([0., 0.]), orientation=0.):
    angle = jnp.pi / 3

    x = x - phase[0]
    y = y - phase[1]
    
    c, s = jnp.cos(-orientation), jnp.sin(-orientation)
    xr_ = c * x - s * y
    yr_ = s * x + c * y
    
    u_l = xr_ - yr_ / jnp.tan(angle)

    yr_folded = yr_ % (l * jnp.sin(angle))
    xr_folded = (u_l % l) + yr_folded / jnp.tan(angle)

    return xr_folded, yr_folded

@jit
def distance_torus_sq(X, Y, l):
    angle = jnp.pi / 3
    
    X_x, X_y = X[..., 0], X[..., 1]
    Y_x, Y_y = Y[..., 0], Y[..., 1]
    
    distances_sq = jnp.full_like(X_x - Y_x, jnp.inf)
    
    for m in [-1, 0, 1]:
        for n in [-1, 0, 1]:                       
            dx = X_x - Y_x + m * l + n * l * jnp.cos(angle)
            dy = X_y - Y_y + n * l * jnp.sin(angle)
            
            D_sq = dx**2 + dy**2
            distances_sq = jnp.minimum(distances_sq, D_sq)
            
    return distances_sq

def build_torus_connectivity(
    X_pre, 
    X_post, 
    sigma, 
    torus_l, 
    l_asym=0.0, 
    hd_pre=None, 
    orientation=0.0,
    map2torus_fn=map2torus_fn,
    distance_torus_fn=distance_torus_sq
):
    if hd_pre is None or l_asym == 0.0:
        X_target = X_pre
    else:
        dx = l_asym * jnp.cos(hd_pre)
        dy = l_asym * jnp.sin(hd_pre)
        
        x_shifted = X_pre[:, 0] + dx
        y_shifted = X_pre[:, 1] + dy
        
        x_mapped, y_mapped = map2torus_fn(x_shifted, y_shifted, l=torus_l, orientation=orientation)
        X_target = jnp.column_stack((x_mapped, y_mapped))

    X_target_exp = X_target[None, :, :]
    X_post_exp = X_post[:, None, :]
    
    dist_sq_matrix = distance_torus_sq(X_target_exp, X_post_exp, torus_l)
    W = jnp.exp(- (dist_sq_matrix) / (2 * sigma ** 2))
    
    return W

def build_feedforward_connectivity(
    X_space, 
    X_phases, 
    sigma, 
    torus_l, 
    orientation=0.0
):
    x_mapped, y_mapped = map2torus_fn(X_space[:, 0], X_space[:, 1], l=torus_l, orientation=orientation)
    X_space_folded = jnp.column_stack((x_mapped, y_mapped))
    
    X_target_exp = X_space_folded[None, :, :] 
    X_phases_exp = X_phases[:, None, :]        
    
    dist_matrix_sq = distance_torus_sq(X_target_exp, X_phases_exp, torus_l)
    W = jnp.exp(- dist_matrix_sq / (2 * sigma ** 2))
    
    return W

def generate_uniform_toroidal_phase_distribution(nx, ny, l):
    x = jnp.zeros(nx*ny)
    y = jnp.zeros(nx*ny)
    seq = jnp.linspace(0, l, nx, False)
    for iy in range(ny):
        x = x.at[iy*nx:(iy+1)*nx].set(seq + iy/ny*l*jnp.cos(jnp.pi/3))
        y = y.at[iy*nx:(iy+1)*nx].set(iy/ny*l*jnp.sin(jnp.pi/3))
    return x, y

def compute_rate_maps_from_sparse_utils(sparse_spikes, traj, selected_neurons, L, dt, nx=30, ny=30):
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    
    x_idx = np.clip(np.floor((x_pos / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((y_pos / L) * ny).astype(int), 0, ny - 1)
    spatial_idx = y_idx * nx + x_idx 

    occupancy_1d = np.bincount(spatial_idx, minlength=nx * ny) * dt
    safe_occupancy = np.where(occupancy_1d > 0, occupancy_1d, 1.0)
    
    selected_neurons = np.asarray(selected_neurons)
    mask = np.isin(sparse_spikes['neuron_idx'], selected_neurons)
    
    filt_time_idx = sparse_spikes['time_idx'][mask]
    filt_neuron_idx = sparse_spikes['neuron_idx'][mask]
    filt_counts = sparse_spikes['counts'][mask]
    
    spike_spatial_idx = spatial_idx[filt_time_idx]
    
    num_selected = len(selected_neurons)
    rate_maps = np.zeros((num_selected, ny, nx))
    
    for i, neuron_id in enumerate(selected_neurons):
        n_mask = (filt_neuron_idx == neuron_id)
        n_spatial_idx = spike_spatial_idx[n_mask]
        n_counts = filt_counts[n_mask]
        
        spike_map_1d = np.bincount(n_spatial_idx, weights=n_counts, minlength=nx * ny)
        rate_map_2d = (spike_map_1d / safe_occupancy).reshape(ny, nx)
        rate_map_2d[occupancy_1d.reshape(ny, nx) == 0] = 0 
        
        rate_maps[i] = rate_map_2d
        
    return rate_maps

def find_spatial_shift_subpixel(corr_map, n=3, search_radius_pixels=None):
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n must be an odd integer >= 3, but got {n}")
    h = (n - 1) // 2

    shape = corr_map.shape
    center_y, center_x = shape[0] // 2, shape[1] // 2

    if search_radius_pixels is not None:
        search_map = corr_map.copy()
        y, x = np.indices(shape)
        dist_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        search_map[dist_from_center > search_radius_pixels] = -np.inf
        peak_y, peak_x = np.unravel_index(np.argmax(search_map), shape)

        if np.isinf(search_map[peak_y, peak_x]):
            print("Warning: No peak found within search_radius. Returning (0,0) shift.")
            return (0.0, 0.0)
    else:
        peak_y, peak_x = np.unravel_index(np.argmax(corr_map), shape)

    if (peak_y < h or peak_y >= shape[0] - h or
            peak_x < h or peak_x >= shape[1] - h):
        print(f"Warning: Peak is too close to border for {n}x{n} fit. Returning integer-pixel shift.")
        return (float(center_y - peak_y), float(center_x - peak_x))

    z = corr_map[peak_y - h: peak_y + h + 1, peak_x - h: peak_x + h + 1]

    y, x = np.array(list(np.ndindex(n, n))).T - h
    A = np.vstack([x**2, y**2, x * y, x, y, np.ones(n * n)]).T

    z_flat = z.flatten()
    try:
        p = la.lstsq(A, z_flat, rcond=None)[0]
    except la.LinAlgError:
        print("Warning: Linear algebra error. Returning integer shift.")
        return (float(center_y - peak_y), float(center_x - peak_x))
    a, b, c, d, e, f = p

    M = np.array([[2 * a, c], [c, 2 * b]])
    v = np.array([-d, -e])

    try:
        offsets = la.solve(M, v)
        x_offset, y_offset = offsets
    except la.LinAlgError:
        print("Warning: Singular matrix in vertex calculation. Returning integer shift.")
        return (float(center_y - peak_y), float(center_x - peak_x))

    if abs(x_offset) > h or abs(y_offset) > h:
        print(f"Warning: Sub-pixel offset > {h}. Fit unstable. Returning integer shift.")
        return (float(center_y - peak_y), float(center_x - peak_x))

    subpixel_y = peak_y + y_offset
    subpixel_x = peak_x + x_offset
    final_shift_y = center_y - subpixel_y
    final_shift_x = center_x - subpixel_x
    return (final_shift_y, final_shift_x)

def compute_rate_maps_from_sparse(sparse_spikes, traj, selected_neurons, L, dt, nx=30, ny=30, speed_thresh=2.5, sigma_val=3.0, min_events=10):
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    
    vx = np.gradient(x_pos, dt)
    vy = np.gradient(y_pos, dt)
    speed = np.sqrt(vx**2 + vy**2)
    
    valid_speed_mask = speed > speed_thresh
    
    x_idx = np.clip(np.floor((x_pos / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((y_pos / L) * ny).astype(int), 0, ny - 1)
    spatial_idx = y_idx * nx + x_idx
    
    occupancy_1d = np.bincount(spatial_idx[valid_speed_mask], minlength=nx * ny) * dt
    safe_occupancy = np.where(occupancy_1d > 0, occupancy_1d, 1.0)
    occupancy_2d = occupancy_1d.reshape(ny, nx)
    
    selected_neurons = np.asarray(selected_neurons)
    mask = np.isin(sparse_spikes['neuron_idx'], selected_neurons)
    
    filt_time_idx = sparse_spikes['time_idx'][mask]
    filt_neuron_idx = sparse_spikes['neuron_idx'][mask]
    filt_counts = sparse_spikes['counts'][mask]
    
    spike_speed_mask = valid_speed_mask[filt_time_idx]
    filt_time_idx = filt_time_idx[spike_speed_mask]
    filt_neuron_idx = filt_neuron_idx[spike_speed_mask]
    filt_counts = filt_counts[spike_speed_mask]
    
    spike_spatial_idx = spatial_idx[filt_time_idx]
    
    num_selected = len(selected_neurons)
    rate_maps = np.zeros((num_selected, ny, nx))
    
    bin_size = L / nx
    sigma_bins = sigma_val / bin_size
    
    for i, neuron_id in enumerate(selected_neurons):
        n_mask = (filt_neuron_idx == neuron_id)
        n_counts = filt_counts[n_mask]
        
        if np.sum(n_counts) <= min_events:
            rate_maps[i] = np.zeros((ny, nx))
            continue
            
        n_spatial_idx = spike_spatial_idx[n_mask]
        
        spike_map_1d = np.bincount(n_spatial_idx, weights=n_counts, minlength=nx * ny)
        
        raw_rate_map_2d = (spike_map_1d / safe_occupancy).reshape(ny, nx)
        raw_rate_map_2d[occupancy_2d == 0] = 0
        
        smoothed_rate_map = gaussian_filter(raw_rate_map_2d, sigma=sigma_bins, mode='constant', cval=0)
        rate_maps[i] = smoothed_rate_map
        
    return rate_maps

def load_and_compute_maps_from_sparse(num, sim_folder, nx=30, ny=30, sigma=3):
    if not os.path.exists(sim_folder):
        raise FileNotFoundError(f"Simulation directory not found: {sim_folder}")
        
    loaded_maps = {}
    loaded_maps_conj1 = {}
    loaded_maps_conj2 = {}
    traj_list = []
    
    folder_list = [l for l in os.listdir(sim_folder) if os.path.isdir(os.path.join(sim_folder, l))]
    
    for fldr_idx, folder_name in enumerate(folder_list):
        folder_path = os.path.join(sim_folder, folder_name)
        
        param_path = os.path.join(folder_path, 'parameters_complete.pkl')
        traj_path = os.path.join(folder_path, 'traj.npy')
        omni_spikes_path = os.path.join(folder_path, 'sparse_spikes_omni.pkl')
        conj_spikes_path1 = os.path.join(folder_path, 'sparse_spikes_conj1.pkl')
        conj_spikes_path2 = os.path.join(folder_path, 'sparse_spikes_conj2.pkl')
        
        if not os.path.exists(param_path) or not os.path.exists(traj_path):
            print(f"Skipping {folder_name}: Missing files.")
            continue
            
        with open(param_path, 'rb') as file:
            params = pickle.load(file)
            
        traj = np.load(traj_path)
        traj_list.append(traj)
        
        L = params['L']
        dt = params['dt']
        
        with open(omni_spikes_path, 'rb') as file:
            omni_spikes = pickle.load(file)
        with open(conj_spikes_path1, 'rb') as file:
            conj_spikes1 = pickle.load(file)
        with open(conj_spikes_path2, 'rb') as file:
            conj_spikes2 = pickle.load(file)

        N_omni = np.max(omni_spikes['neuron_idx']) + 1 if len(omni_spikes['neuron_idx']) > 0 else 0
        N_conj1 = np.max(conj_spikes1['neuron_idx']) + 1 if len(conj_spikes1['neuron_idx']) > 0 else 0
        N_conj2 = np.max(conj_spikes2['neuron_idx']) + 1 if len(conj_spikes2['neuron_idx']) > 0 else 0

        print(f"Computing maps for {folder_name}...")
        rate_map_omni = compute_rate_maps_from_sparse(
            omni_spikes, traj, np.arange(N_omni), L, dt, nx=nx, ny=ny, sigma_val=sigma)
        rate_map_conj1 = compute_rate_maps_from_sparse(
            conj_spikes1, traj, np.arange(N_conj1), L, dt, nx=nx, ny=ny, sigma_val=sigma)
        rate_map_conj2 = compute_rate_maps_from_sparse(
            conj_spikes2, traj, np.arange(N_conj2), L, dt, nx=nx, ny=ny, sigma_val=sigma)
        
        loaded_maps[fldr_idx] = {'spiking maps': {}}
        loaded_maps_conj1[fldr_idx] = {'spiking maps': {}}
        loaded_maps_conj2[fldr_idx] = {'spiking maps': {}}
        
        loaded_maps[fldr_idx]['rate maps'] = rate_map_omni
        loaded_maps[fldr_idx]['spiking maps']['neuron_idx'] = omni_spikes['neuron_idx']
        loaded_maps[fldr_idx]['spiking maps']['time_idx'] = omni_spikes['time_idx']
        
        loaded_maps_conj1[fldr_idx]['rate maps'] = rate_map_conj1
        loaded_maps_conj1[fldr_idx]['spiking maps']['neuron_idx'] = conj_spikes1['neuron_idx']
        loaded_maps_conj1[fldr_idx]['spiking maps']['time_idx'] = conj_spikes1['time_idx']
        
        loaded_maps_conj2[fldr_idx]['rate maps'] = rate_map_conj2
        loaded_maps_conj2[fldr_idx]['spiking maps']['neuron_idx'] = conj_spikes2['neuron_idx']
        loaded_maps_conj2[fldr_idx]['spiking maps']['time_idx'] = conj_spikes2['time_idx']
        
        print(f"Finished {folder_name}. Shapes: Omni {rate_map_omni.shape}, Conj1 {rate_map_conj1.shape}, Conj2 {rate_map_conj2.shape}")
            
    return loaded_maps, loaded_maps_conj1, loaded_maps_conj2, traj_list

def compute_cross_corrs(fr_maps0, fr_maps1, sd=2, smooth=False):
    CrossCorr = np.zeros((fr_maps0.shape[0], fr_maps0.shape[1]*2-1, fr_maps0.shape[1]*2-1))
    
    for ind in tqdm(range(CrossCorr.shape[0]), desc="Cross Correlograms"):
        rm1 = (fr_maps0[ind] - fr_maps0[ind].mean()) / fr_maps0[ind].std()
        rm2 = (fr_maps1[ind] - fr_maps1[ind].mean()) / fr_maps1[ind].std()
        
        if smooth:
            rm1 = gaussian_filter(rm1, (sd, sd), mode='constant', cval=0)
            rm2 = gaussian_filter(rm2, (sd, sd), mode='constant', cval=0)
            
        CrossCorr[ind] = correlate2d(rm1, rm2, mode='full', boundary='fill', fillvalue=0)
        
    shifts = np.zeros((fr_maps0.shape[0], 2))
    for i in tqdm(range(fr_maps0.shape[0]), desc="Spatial Shift"):
        sy, sx = find_spatial_shift_subpixel(CrossCorr[i], n=7, search_radius_pixels=7)
        shifts[i, 0], shifts[i, 1] = sx, sy
    
    return shifts, CrossCorr

def compute_directional_rate_maps(spiking_maps, traj, is_cond, num_neurons, L, dt, nx=30, ny=30):
    x_idx = np.clip(np.floor((traj[:, 0] / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((traj[:, 1] / L) * ny).astype(int), 0, ny - 1)
    spatial_idx = y_idx * nx + x_idx

    occ = np.bincount(spatial_idx[is_cond], minlength=nx * ny) * dt
    safe_occ = np.where(occ > 0, occ, 1.0)

    t_idx = spiking_maps['time_idx']
    n_idx = spiking_maps['neuron_idx']

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