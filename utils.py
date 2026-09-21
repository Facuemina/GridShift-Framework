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
from scipy.ndimage import rotate
from scipy.stats import pearsonr

# from skimage.feature import peak_local_max


def generate2D_pos(steps, Lx, Ly, v, sigma_theta, delta_t, periodic=False, seed=0):
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

@jit
def gaussian(x, y, std, L):
    dx = jnp.minimum(jnp.abs(x - y), L - jnp.abs(x - y))
    norm = 1 / jnp.sqrt(2 * jnp.pi * std**2)
    return jnp.exp(-0.5 * (dx / std) ** 2) * norm

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
    is_scalar_zero = isinstance(l_asym, (int, float)) and l_asym == 0.0

    if hd_pre is None or is_scalar_zero:
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


def generate_uniform_toroidal_phase_distribution(nx, ny, l):
    x = jnp.zeros(nx*ny)
    y = jnp.zeros(nx*ny)
    seq = jnp.linspace(0, l, nx, False)
    for iy in range(ny):
        x = x.at[iy*nx:(iy+1)*nx].set(seq + iy/ny*l*jnp.cos(jnp.pi/3))
        y = y.at[iy*nx:(iy+1)*nx].set(iy/ny*l*jnp.sin(jnp.pi/3))
    return x, y

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
    traj_list = []
    
    folder_list = [l for l in os.listdir(sim_folder) if (os.path.isdir(os.path.join(sim_folder, l)) and 'incl_ang' in l)]
    
    for fldr_idx, folder_name in enumerate(folder_list):
        folder_path = os.path.join(sim_folder, folder_name)
        
        param_path = os.path.join(folder_path, 'parameters_complete.pkl')
        traj_path = os.path.join(folder_path, 'traj.npy')
        spikes_path = os.path.join(folder_path, 'sparse_spikes.pkl')
        
        if not os.path.exists(param_path) or not os.path.exists(traj_path):
            print(f"Skipping {folder_name}: Missing files.")
            continue
            
        with open(param_path, 'rb') as file:
            params = pickle.load(file)
            
        traj = np.load(traj_path)
        traj_list.append(traj)
        
        L = params['L']
        dt = params['dt']
        
        with open(spikes_path, 'rb') as file:
            spikes = pickle.load(file)

        N = np.max(spikes['neuron_idx']) + 1 if len(spikes['neuron_idx']) > 0 else 0
        
        print(f"Computing maps for {folder_name}...")
        rate_map = compute_rate_maps_from_sparse(
            spikes, traj, np.arange(N), L, dt, nx=nx, ny=ny, sigma_val=sigma)
        
        loaded_maps[fldr_idx] = {'spiking maps': {}}
        
        loaded_maps[fldr_idx]['rate maps'] = rate_map
        loaded_maps[fldr_idx]['spiking maps']['neuron_idx'] = spikes['neuron_idx']
        loaded_maps[fldr_idx]['spiking maps']['time_idx'] = spikes['time_idx']
        
        print(f"Finished {folder_name}. Shapes: {rate_map.shape}")
            
    return loaded_maps, traj_list


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

from scipy.ndimage import maximum_filter

def find_2d_peaks(autocorrelogram, min_distance=3):
    # 1. Define the neighborhood size based on your min_distance
    # A min_distance of 3 roughly translates to a 7x7 window (2*3 + 1)
    neighborhood_size = 2 * min_distance + 1
    
    # 2. Apply a maximum filter
    local_max = maximum_filter(autocorrelogram, size=neighborhood_size)
    
    # 3. Find where the original array equals the local maximum
    # (These are your peaks)
    peak_mask = (autocorrelogram == local_max)
    
    # 4. Get the coordinates of these peaks
    peaks = np.argwhere(peak_mask)
    
    return peaks

def compute_grid_metrics(autocorrelogram, bin_size=1.0):
    """
    Computes grid score and grid spacing from a 2D spatial autocorrelogram.
    
    Parameters:
    - autocorrelogram (2D numpy array): The spatial autocorrelogram of the firing rate map.
    - bin_size (float): The spatial size of each bin (e.g., in cm) to scale the spacing output.
    
    Returns:
    - grid_score (float): Mean correlation at (60, 120) minus mean at (30, 90, 150).
    - grid_spacing (float): Mean distance to the 6 nearest peaks (scaled by bin_size).
    """
    
    # 1. Find all local peaks in the autocorrelogram
    # min_distance prevents detecting multiple pixels on the same broad peak
    peaks = find_2d_peaks(autocorrelogram, min_distance=3)
    
    # The central peak is the absolute maximum of an autocorrelogram
    center_y, center_x = np.unravel_index(np.argmax(autocorrelogram), autocorrelogram.shape)
    center_coord = np.array([center_y, center_x])
    
    # 2. Calculate distances from the central peak to all other peaks
    distances = np.linalg.norm(peaks - center_coord, axis=1)
    
    # Sort peaks by distance
    sorted_indices = np.argsort(distances)
    
    # Ensure we have enough peaks to measure (1 center + 6 surrounding = 7 minimum)
    if len(peaks) < 7:
        raise ValueError("Not enough peaks detected to compute grid metrics.")
        
    # The closest peak is the center itself (distance = 0)
    # The next 6 closest are the inner vertices of the grid
    nearest_6_distances = distances[sorted_indices[1:7]]
    
    # 3. Compute Grid Spacing
    grid_spacing = np.mean(nearest_6_distances) * bin_size
    
    # 4. Prepare for Grid Score computation
    # Create a circular mask to isolate the central grid structure for correlation.
    # We mask up to slightly beyond the 6 inner peaks (1.5x is a standard heuristic).
    outer_radius = np.max(nearest_6_distances) * 1.5
    Y, X = np.ogrid[:autocorrelogram.shape[0], :autocorrelogram.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Optional but standard: exclude the central peak itself so it doesn't dominate the correlation
    inner_radius = np.min(nearest_6_distances) * 0.4
    mask = (dist_from_center <= outer_radius) & (dist_from_center >= inner_radius)
    
    def get_rotated_correlation(angle):
        # Rotate the autocorrelogram
        rotated_auto = rotate(autocorrelogram, angle, reshape=False, order=1)
        
        # Extract the masked pixels for both original and rotated arrays
        orig_pixels = autocorrelogram[mask]
        rot_pixels = rotated_auto[mask]
        
        # Calculate Pearson correlation
        corr, _ = pearsonr(orig_pixels, rot_pixels)
        return corr
    
    # 5. Compute Grid Score
    corr_30 = get_rotated_correlation(30)
    corr_60 = get_rotated_correlation(60)
    corr_90 = get_rotated_correlation(90)
    corr_120 = get_rotated_correlation(120)
    corr_150 = get_rotated_correlation(150)
    
    mean_60_120 = np.mean([corr_60, corr_120])
    mean_30_90_150 = np.mean([corr_30, corr_90, corr_150])
    
    grid_score = mean_60_120 - mean_30_90_150
    
    return grid_score, grid_spacing

def get_spatial_correlation(acg1, acg2):
    """Computes the Pearson correlation between two 2D autocorrelograms."""
    # Mask out NaNs (e.g., circular borders in grid cell autocorrelograms)
    valid = ~np.isnan(acg1) & ~np.isnan(acg2)
            
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
    valid_indices = [] 
    for i, j in sampled_indices:
        r = get_spatial_correlation(list_a[i], list_b[j])
        if not np.isnan(r):
            correlations.append(r)
            valid_indices.append([i, j])
            
    return np.array(correlations), np.array(valid_indices)