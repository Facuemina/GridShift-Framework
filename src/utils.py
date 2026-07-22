# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:21:39 2026

@author: Facundo
"""

import jax.numpy as jnp
from jax import vmap, jit, random, lax
from numbers import Number
import numpy as np
import numpy.linalg as la
from scipy.ndimage import gaussian_filter

def compute_occupancy_map_raw(traj, L, nx, ny):
    """
    Computes a 1D flattened occupancy map (time steps per spatial bin) 
    from a spatial trajectory.
    """
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    
    x_idx = jnp.clip(jnp.floor((x_pos / L) * nx).astype(jnp.int32), 0, nx - 1)
    y_idx = jnp.clip(jnp.floor((y_pos / L) * ny).astype(jnp.int32), 0, ny - 1)
    idx = y_idx * nx + x_idx
    
    occupancy = jnp.bincount(idx, length=nx * ny)
    
    return occupancy

# Apply jit explicitly here, passing the function as the first argument
compute_occupancy_map = jit(compute_occupancy_map_raw, static_argnums=(2, 3))

def generate2D_pos(seed, steps, Lx, Ly, v, sigma_theta, delta_t, periodic = False):
    """
    Reproduces the MATLAB behavior:
      - Random walk in 2D with angular noise
      - When crossing a boundary, retries with smaller angular noise (σθ/3)
        until the step lands inside.
    """
    two_pi = 2 * jnp.pi
    
    key = random.PRNGKey(seed)
    # initial random position and heading
    key, k1, k2, k3 = random.split(key, 4)
    mydir = random.uniform(k1, minval=0.0, maxval=two_pi)
    x = random.uniform(k2, minval=0.0, maxval=Lx)
    y = random.uniform(k3, minval=0.0, maxval=Ly)
    if isinstance(v,Number):
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

        # --- Main step (same as MATLAB first proposal)
        mydir = jnp.mod(
            mydir + random.normal(k_ang) * sigma_theta * jnp.sqrt(delta_t),
            two_pi,
        )
        x_new = x + v_i * jnp.cos(mydir) * delta_t
        y_new = y + v_i * jnp.sin(mydir) * delta_t

        # --- Retry loop when out of bounds
        def cond_fn(state):
            x_t, y_t, dir_t, key_t, count = state
            return out_of_bounds(x_t, y_t) #& (count < 50)

        def body_fn(state):
            x_t, y_t, dir_t, key_t, count = state
            # undo last move
            x_t = x_t - v_i * jnp.cos(dir_t) * delta_t
            y_t = y_t - v_i * jnp.sin(dir_t) * delta_t
            # new smaller angular noise
            key_t, ksub = random.split(key_t)
            dir_t = jnp.mod(
                dir_t
                + random.normal(ksub)
                * sigma_theta
                * jnp.sqrt(delta_t)
                / 3.0,
                two_pi,
            )
            # move again
            x_t = x_t + v_i * jnp.cos(dir_t) * delta_t
            y_t = y_t + v_i * jnp.sin(dir_t) * delta_t
            return (x_t, y_t, dir_t, key_t, count + 1)

        init_state = (x_new, y_new, mydir, key, jnp.array(0))
        x_final, y_final, dir_final, key_final, _ = lax.while_loop(cond_fn, body_fn, init_state)

        return (x_final, y_final, dir_final, key_final), (x_final, y_final, dir_final)

    init_carry = (x, y, mydir, key)
    _, traj = lax.scan(step, init_carry, v, length=steps)
    #traj = jnp.stack(traj, axis=1)  # shape (steps, 3)
    return traj

def map2torus_fn(x, y, l, phase=jnp.array([0., 0.]), orientation=0.):
    angle = jnp.pi / 3

    # 1. Shift and anti-rotate to the lattice frame
    # JAX handles multi-dimensional broadcasting automatically; no reshaping needed.
    x = x - phase[0]
    y = y - phase[1]
    
    c, s = jnp.cos(-orientation), jnp.sin(-orientation)
    xr_ = c * x - s * y
    yr_ = s * x + c * y
    
    # 2. Convert x to the oblique basis (projection along the x-axis)
    u_l = xr_ - yr_ / jnp.tan(angle)

    # 3. Fold the coordinates
    # Wrap y strictly inside the parallelogram height
    yr_folded = yr_ % (l * jnp.sin(angle))
    
    # Wrap the oblique x-coordinate, then transform back to Cartesian
    xr_folded = (u_l % l) + yr_folded / jnp.tan(angle)

    # 4. Rotate back to global coordinates 
    # (Uncomment if your distance function expects global rather than lattice coordinates)
    # xr_final = (c * xr_folded + s * yr_folded) + phase[0]
    # yr_final = (-s * xr_folded + c * yr_folded) + phase[1]
    # return xr_final, yr_final

    return xr_folded, yr_folded

@jit
def distance_torus_sq(X, Y, l):
    """
    Compute squared Euclidean distance on a flat twisted torus.
    Supports N-dimensional broadcasting (e.g., X of shape (N, 1, 2) and Y of shape (1, M, 2)).
    """
    angle = jnp.pi / 3
    
    # Extract coordinates. The '...' ensures it works for 2D, 3D, or any shaped arrays.
    X_x, X_y = X[..., 0], X[..., 1]
    Y_x, Y_y = Y[..., 0], Y[..., 1]
    
    # Initialize distances with infinity
    # This automatically takes the broadcasted shape of X_x - Y_x
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
    """
    Creates a Gaussian connectivity matrix between pre-synaptic and post-synaptic neurons 
    on a 2D twisted torus. Supports asymmetric connectivity based on head direction.

    Args:
        X_pre: (N_pre, 2) array of pre-synaptic neuron phases/positions.
        X_post: (N_post, 2) array of post-synaptic neuron phases/positions.
        sigma: Width of the Gaussian connectivity profile.
        torus_l: Scale/size of the torus (equivalent to 'l' in the reference code).
        l_asym: Magnitude of the asymmetric shift. Defaults to 0.0.
        hd_pre: (N_pre,) array of preferred head directions (in radians). Optional.
        orientation: Orientation parameter for the torus mapping.
        map2torus_fn: Your existing function to map (x,y) coordinates to the torus.
        distance_torus_fn: Your existing function to compute distances on the torus.

    Returns:
        W: (N_post, N_pre) connectivity weight matrix.
    """
    
    # 1. Compute Target Centers (Asymmetric Shift)
    if hd_pre is None or l_asym == 0.0:
        X_target = X_pre
    else:
        # Shift the target projection phase forward along the preferred HD
        dx = l_asym * jnp.cos(hd_pre)
        dy = l_asym * jnp.sin(hd_pre)
        
        x_shifted = X_pre[:, 0] + dx
        y_shifted = X_pre[:, 1] + dy
        
        # Remap the shifted coordinates back to the valid torus space
        x_mapped, y_mapped = map2torus_fn(x_shifted, y_shifted, l=torus_l, orientation=orientation)
        X_target = jnp.column_stack((x_mapped, y_mapped))

    # 2. Compute Pairwise Torus Distances
    # Expand dimensions to leverage broadcasting: 
    # X_target becomes (N_pre, 1, 2) and X_post becomes (1, N_post, 2)
    X_target_exp = X_target[None, :, :]
    X_post_exp = X_post[:, None, :]
    
    # Calculate the (N_pre, N_post) distance matrix
    dist_sq_matrix = distance_torus_sq(X_target_exp, X_post_exp, torus_l)
    
    # 3. Apply Gaussian Profile
    W = jnp.exp(- (dist_sq_matrix) / (2 * sigma ** 2))
    
    return W

def build_feedforward_connectivity(
    X_space, 
    X_phases, 
    sigma, 
    torus_l, 
    orientation=0.0
):
    """
    Maps spatial inputs (e.g., an arena or animal trajectory) to grid cell phases,
    generating a hexagonal firing field.

    Args:
        X_space: (N_space, 2) array of spatial coordinates in the real world.
        X_phases: (N_grid, 2) array of preferred phases for the grid cells.
        sigma: Width of the Gaussian firing field.
        torus_l: Grid spacing scale.
        orientation: Grid orientation angle.

    Returns:
        W: (N_grid, N_space) matrix representing the input to each grid cell 
           at each spatial location.
    """
    
    # 1. Fold all spatial coordinates into the fundamental torus domain
    # This is what creates the repeating hexagonal fields
    x_mapped, y_mapped = map2torus_fn(X_space[:, 0], X_space[:, 1], l=torus_l, orientation=orientation)
    X_space_folded = jnp.column_stack((x_mapped, y_mapped))
    
    # 2. Expand dimensions for pairwise broadcasting
    X_target_exp = X_space_folded[None, :, :]  # Shape: (N_space, 1, 2)
    X_phases_exp = X_phases[:, None, :]        # Shape: (1, N_grid, 2)
    
    # 3. Compute squared distance and apply Gaussian
    dist_matrix_sq = distance_torus_sq(X_target_exp, X_phases_exp, torus_l)
    
    W = jnp.exp(- dist_matrix_sq / (2 * sigma ** 2))
    
    return W

def generate_uniform_toroidal_phase_distribution(nx,ny,l):
    x = jnp.zeros(nx*ny)
    y = jnp.zeros(nx*ny)
    seq = jnp.linspace(0,l,nx,False)
    for iy in range(ny):
        x = x.at[iy*nx:(iy+1)*nx].set(seq + iy/ny*l*jnp.cos(jnp.pi/3))
        y = y.at[iy*nx:(iy+1)*nx].set(iy/ny*l*jnp.sin(jnp.pi/3))
    return x, y

def find_spatial_shift_subpixel(corr_map, n=3, search_radius_pixels=None):
    """
    Finds the sub-pixel shift by fitting a 2D quadratic to the
    N x N neighborhood around the integer peak.

    Args:
        corr_map (np.ndarray): The 2D cross-correlogram.
        n (int, optional): The size of the neighborhood to fit.
                           Must be an odd integer (e.g., 3, 5, 7).
                           Defaults to 3 (a 3x3 grid).
        search_radius_pixels (int, optional): If provided, only searches
            for the peak within this pixel radius of the map's center.
            This is used to ignore periodic side-peaks.

    Returns:
        tuple (float, float): The sub-pixel spatial shift in (shift_y, shift_x).
    """
    if n < 3 or n % 2 == 0:
        raise ValueError(f"n must be an odd integer >= 3, but got {n}")

    h = (n - 1) // 2

    # 1. Find integer peak (coarse search)
    shape = corr_map.shape
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # ---RESTRICT SEARCH AREA ---
    if search_radius_pixels is not None:
        # Create a map to search, copying the original
        search_map = corr_map.copy()
        
        # Create coordinate grids
        y, x = np.indices(shape)
        
        # Calculate distance from center for every pixel
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        
        # Mask out all pixels *outside* the search radius
        # by setting them to a very low value
        search_map[dist_from_center > search_radius_pixels] = -np.inf
        
        # Find the peak on this new *masked* map
        peak_y, peak_x = np.unravel_index(np.argmax(search_map), shape)
        
        if np.isinf(search_map[peak_y, peak_x]):
            print("Warning: No peak found within search_radius. "
                  "Returning (0,0) shift.")
            return (0.0, 0.0)
            
    else:
        # Original behavior: find the global maximum
        peak_y, peak_x = np.unravel_index(np.argmax(corr_map), shape)
    # -----------------------
    
    # 2. Handle edge cases (UPDATED with h)
    if (peak_y < h or peak_y >= shape[0] - h or
        peak_x < h or peak_x >= shape[1] - h):
        print(f"Warning: Peak is too close to border for {n}x{n} fit. "
              "Returning integer-pixel shift.")
        return (float(peak_y - center_y), float(peak_x - center_x))

    # 3. Extract n x n neighborhood
    z = corr_map[peak_y-h : peak_y+h+1, peak_x-h : peak_x+h+1]
    
    # 4. Create design matrix 'A'
    y, x = np.array(list(np.ndindex(n, n))).T - h
    A = np.vstack([x**2, y**2, x*y, x, y, np.ones(n*n)]).T
    
    # 5. Solve for parameters
    z_flat = z.flatten()
    try:
        p = la.lstsq(A, z_flat, rcond=None)[0]
    except la.LinAlgError:
        print("Warning: Linear algebra error. Returning integer shift.")
        return (float(peak_y - center_y), float(peak_x - center_x))

    a, b, c, d, e, f = p

    # 6. Find vertex
    M = np.array([[2*a, c], [c, 2*b]])
    v = np.array([-d, -e])
    
    try:
        offsets = la.solve(M, v)
        x_offset, y_offset = offsets
    except la.LinAlgError:
        print("Warning: Singular matrix in vertex calculation. Returning integer shift.")
        return (float(peak_y - center_y), float(peak_x - center_x))

    # 7. Check if reasonable
    if abs(x_offset) > h or abs(y_offset) > h:
        print(f"Warning: Sub-pixel offset > {h}. Fit unstable. Returning integer shift.")
        return (float(peak_y - center_y), float(peak_x - center_x))
            
    # 8. Calculate final sub-pixel shift
    subpixel_y = peak_y + y_offset
    subpixel_x = peak_x + x_offset
    final_shift_y = center_y - subpixel_y
    final_shift_x = center_x - subpixel_x

    return (final_shift_y, final_shift_x)

def compute_rate_maps_from_sparse(sparse_spikes, traj, selected_neurons, L, dt, nx=30, ny=30):
    """
    Computes spatial firing rate maps for a subset of neurons efficiently.
    
    Args:
        sparse_spikes (dict): Dictionary containing 'time_idx', 'neuron_idx', and 'counts'.
        traj (np.ndarray): Trajectory array of shape (steps, 3) where columns are [x, y, hd].
        selected_neurons (array-like): Array or list of neuron indices to process.
        L (float): Arena size.
        dt (float): Integration time step.
        nx (int, optional): Number of spatial bins in x. Defaults to 30.
        ny (int, optional): Number of spatial bins in y. Defaults to 30.
        
    Returns:
        np.ndarray: Rate maps of shape (len(selected_neurons), ny, nx).
    """
    x_pos = traj[:, 0]
    y_pos = traj[:, 1]
    
    # 1. Map continuous trajectory to 1D spatial bin indices
    x_idx = np.clip(np.floor((x_pos / L) * nx).astype(int), 0, nx - 1)
    y_idx = np.clip(np.floor((y_pos / L) * ny).astype(int), 0, ny - 1)
    # spatial_idx = x_idx * ny + y_idx  # Shape: (steps,)
    spatial_idx = y_idx * nx + x_idx # Shape: (steps,)
    # 2. Compute Occupancy Map (time spent in each bin in seconds)
    occupancy_1d = np.bincount(spatial_idx, minlength=nx * ny) * dt
    safe_occupancy = np.where(occupancy_1d > 0, occupancy_1d, 1.0)
    
    # 3. Filter sparse spikes to only include the selected neurons
    selected_neurons = np.asarray(selected_neurons)
    mask = np.isin(sparse_spikes['neuron_idx'], selected_neurons)
    
    filt_time_idx = sparse_spikes['time_idx'][mask]
    filt_neuron_idx = sparse_spikes['neuron_idx'][mask]
    filt_counts = sparse_spikes['counts'][mask]
    
    # Get the 1D spatial bin index for every single spike
    spike_spatial_idx = spatial_idx[filt_time_idx]
    
    # 4. Generate rate maps
    num_selected = len(selected_neurons)
    rate_maps = np.zeros((num_selected, ny, nx))
    
    # Loop through the subset of selected neurons (fast because it scales with subset size, not time)
    for i, neuron_id in enumerate(selected_neurons):
        n_mask = (filt_neuron_idx == neuron_id)
        n_spatial_idx = spike_spatial_idx[n_mask]
        n_counts = filt_counts[n_mask]
        
        # Sum spikes in each spatial bin using counts as weights
        spike_map_1d = np.bincount(n_spatial_idx, weights=n_counts, minlength=nx * ny)
        
        # Divide by occupancy and reshape to 2D
        rate_map_2d = (spike_map_1d / safe_occupancy).reshape(ny, nx)
        
        # Optional: Set unvisited bins to 0 (or np.nan if preferred for plotting)
        rate_map_2d[occupancy_1d.reshape(ny, nx) == 0] = 0 
        
        rate_maps[i] = rate_map_2d
        
    return rate_maps
    
    
    