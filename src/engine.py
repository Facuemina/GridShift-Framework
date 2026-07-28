#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine.py

Core physics formulations, matrix initializations, and highly 
optimized JAX integration loops for CANN simulations.
"""

from jax import numpy as jnp
from jax import jit, lax, random
from numbers import Number
# ---------------------------------------------------------
# INITIALIZATIONS & TOPOLOGY
# ---------------------------------------------------------

@jit
def gaussian(x, y, std, L):
    """Periodic Gaussian function."""
    dx = jnp.minimum(jnp.abs(x - y), L - jnp.abs(x - y))
    norm = 1 / jnp.sqrt(2 * jnp.pi * std**2)
    return jnp.exp(-0.5 * (dx / std) ** 2) * norm

@jit
def gaussian2D(X, Y, std, Lx, Ly = 0):
    """Periodic 2D Gaussian function.
    Inputs:
        X.shape = (N,2)
        Y.shape = (2,M)
        std = float
        Lx = float
        Ly = float
    """
    Ly = jnp.where(Ly,Ly,Lx)
    return gaussian(X[:,0:1], Y[0:1,:],std,Lx) * gaussian(X[:,1:2], Y[1:2,:],std,Ly)

def create_gaussian_connectivity_matrix(Nin, std, Lin, Nout=None, Lout=None, W0=1., periodic=True):
    """Generates a Gaussian weight matrix for the network."""
    x_in = jnp.linspace(0, Lin, Nin, 1 - periodic).reshape((1, Nin))
    Nout = Nin if Nout is None else Nout
    Lout = Lin if Lout is None else Lout
    x_out = jnp.linspace(0, Lout, Nout, 1 - periodic).reshape((Nout, 1)) / Lout * Lin
    return W0 * gaussian(x_in, x_out, std, Lin * periodic)

# ---------------------------------------------------------
# CORE PHYSICS ENGINE (UNJITTED)
# ---------------------------------------------------------

def _cann_dynamics(U, V, fU, W_ff, W_rec, I_vis, I_hd, neural_params, thresh=0):
    """
    Core ODE system for the CANN. 
    This function is inlined by the JAX compiler in the simulation loops.
    """
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    if isinstance(thresh, Number):
        if thresh == 0:
            def fun_thresh(U):
                return 0
        else:
            def fun_thresh(U):
                return jnp.quantile(U,thresh)
    else:
        thresh = jnp.array(thresh)
        if len(thresh)>1:
            def fun_thresh(U):
                return thresh[0]
        else:
            if thresh == 0:
                def fun_thresh(U):
                    return 0
            else:
                def fun_thresh(U):
                    return jnp.quantile(U,thresh)
    
    # Firing rate
    U_pos = jnp.maximum(U-fun_thresh(U), 0)
    norm_sq = jnp.sum(U_pos ** g)
    fU = 10 * gain * (U_pos ** g) / (1 + k * norm_sq)
    
    I_input = I_hd * (W_ff @ I_vis + W_rec @ fU)
    # Dynamics
    dU = (-U - V + I_input) * tau_inv
    dV = (-V + m * U) * tauv_inv
    
    return dU, dV, fU

def _omnidirectional_dynamics(U, V, W_ff, I_conj, I_vest, neural_params, thresh = 0):
    """
    Core ODE system for the CANN. 
    This function is inlined by the JAX compiler in the simulation loops.
    """
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    
    if isinstance(thresh, Number):
        if thresh == 0:
            def fun_thresh(U):
                return 0
        else:
            def fun_thresh(U):
                return jnp.quantile(U,thresh)
    else:
        thresh = jnp.array(thresh)
        if len(thresh)>1:
            def fun_thresh(U):
                return thresh[1]
        else:
            if thresh == 0:
                def fun_thresh(U):
                    return 0
            else:
                def fun_thresh(U):
                    return jnp.quantile(U,thresh)
            
    
    # Firing rate
    U_pos = jnp.maximum(U-fun_thresh(U), 0)
    norm_sq = jnp.sum(U_pos ** g)
    fU = gain * (U_pos ** g) / (1 + k * norm_sq)
    
    I_input = W_ff @ I_conj#(I_vest.T * W_ff) @ I_conj
    # Dynamics
    dU = (-U - V + I_input) * tau_inv
    dV = (-V + m * U) * tauv_inv
    
    return dU, dV, fU

# ---------------------------------------------------------
# SIMULATION LOOPS
# ---------------------------------------------------------

def run_rate_simulation(U0, V0, weights, traj, neural_params, 
                        input_params, pos, pref_hd, steps, 
                         nx=30, ny=30, thresh = 0):
    """Standard simulation with fixed weights and firing rate map tracking."""
    pi2 = 2 * jnp.pi
    Wvis_conj, Wrec_conj, Wconj_omni = weights
    
    U_conj0, U_omni0 = U0
    V_conj0, V_omni0 = V0
    
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    dt, sR, sHD_input, A_vis, A_hd, A_vest, L = input_params
    if isinstance(sHD_input,tuple):
        sHD, inc_sHD = sHD_input
    else:
        sHD = sHD_input
        inc_sHD = sHD_input
    
    # Firing rate initialization (Fixed jnp.maximum bug)
    U_pos = jnp.maximum(U_conj0, 0) 
    norm_sq = jnp.sum(U_pos ** g)
    fU_conj0 = gain * (U_pos ** g) / (1 + k * norm_sq)
    
    # Initialize the firing rate maps
    rm_conj0 = jnp.zeros((U_conj0.shape[0], nx * ny))
    rm_omni0 = jnp.zeros((U_omni0.shape[0], nx * ny))
    
    
    @jit
    def step(i, carry):
        # Unpack the fully flattened carry
        U_conj, U_omni, V_conj, V_omni, fU_conj, rm_conj, rm_omni = carry
        
        # Read trajectory cleanly
        x_pos, y_pos = traj[i, 0], traj[i, 1]
        hd = traj[i, 2]
        
        I_vis = A_vis * gaussian2D(pos, traj[i, :-1][:, None], sR, L)
        I_hd = A_hd * gaussian(pref_hd, hd, sHD, pi2)
        # I_vest = 1 + A_vest * gaussian(hd, jnp.pi/2, sHD, pi2) #GLOBAL
        I_vest = 1 + A_vest * gaussian(pref_hd, jnp.pi/2, inc_sHD, pi2) #LOCAL
        # I_hd = I_hd * I_vest
        # I_vest = 1 + 0*I_vest
        
        dU_conj, dV_conj, fU_conj_next = _cann_dynamics(
            U_conj, V_conj, fU_conj, Wvis_conj, Wrec_conj, I_vis, I_hd, 
            neural_params, thresh
        )
        
        dU_omni, dV_omni, fU_omni_next = _omnidirectional_dynamics(
            U_omni, V_omni, Wconj_omni, fU_conj, I_vest, neural_params,
        thresh)
        
        U_conj_next = U_conj + dt * dU_conj
        V_conj_next = V_conj + dt * dV_conj
        U_omni_next = U_omni + dt * dU_omni
        V_omni_next = V_omni + dt * dV_omni
        
        # --- Map 2D coordinate to 1D index ---
        # jnp.clip ensures that floating-point errors don't cause out-of-bounds indexing
        x_idx = jnp.clip(jnp.floor((x_pos / L) * nx).astype(jnp.int32), 0, nx - 1)
        y_idx = jnp.clip(jnp.floor((y_pos / L) * ny).astype(jnp.int32), 0, ny - 1)
        idx = y_idx * nx + x_idx
        
        # Update maps: broadcasting fU_conj_next (N,) into rm_conj[:, idx]
        rm_conj = rm_conj.at[:, idx].add(fU_conj_next[:,0] / steps)
        rm_omni = rm_omni.at[:, idx].add(fU_omni_next[:,0] / steps)
        
        # fori_loop strictly requires returning only the carry
        return (U_conj_next, U_omni_next, V_conj_next, V_omni_next, fU_conj_next, rm_conj, rm_omni)
    
    # Initialize the loop with the flattened tuple
    init_state = (U_conj0, U_omni0, V_conj0, V_omni0, fU_conj0, rm_conj0, rm_omni0)
    
    # Run loop
    final_state = lax.fori_loop(0, steps, step, init_state)
    
    # Unpack final state
    U_conj, U_omni, V_conj, V_omni, fU_conj, rate_map_conj, rate_map_omni = final_state
    
    return U_conj, U_omni, V_conj, V_omni, fU_conj, rate_map_conj, rate_map_omni

def run_spiking_simulation_2sup(rng_key, U0, V0, weights, traj, neural_params, 
                                input_params, pos, pref_hd, steps, thresh=.5,
                                inclination_dir = jnp.pi/2):
    """Standard simulation saving spike counts over time using lax.scan."""
    pi2 = 2 * jnp.pi
    Wvis_conj, Wrec_conj, Wconj_omni = weights
    
    U_conj0, U_omni0 = U0
    V_conj0, V_omni0 = V0
    
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    dt, sR, sHD_input, A_vis, A_hd, A_vest, L = input_params
    
    if isinstance(sHD_input,tuple):
        sHD, inc_sHD = sHD_input
    else:
        sHD = sHD_input
        inc_sHD = sHD_input
    
    # Firing rate initialization
    U_pos = jnp.maximum(U_conj0, 0) 
    norm_sq = jnp.sum(U_pos ** g)
    fU_conj0 = gain * (U_pos ** g) / (1 + k * norm_sq)
    
    # lax.scan passes the current element of `traj` directly to `traj_step`
    def step(carry, traj_step):
        key, U_conj, U_omni, V_conj, V_omni, fU_conj = carry
        
        # Split the key for this step
        key, subkey_conj, subkey_omni = random.split(key, 3)
        
        # Unpack trajectory for this specific time step
        x_pos, y_pos, hd = traj_step[0], traj_step[1], traj_step[2]
        
        # traj_step[:-1] gives [x_pos, y_pos]
        I_vis = A_vis * gaussian2D(pos, traj_step[:-1][:, None], sR, L)
        I_hd = A_hd * gaussian(pref_hd, hd, sHD, pi2) 
        I_vest = 1 + A_vest * gaussian(pref_hd, inclination_dir, inc_sHD, pi2)
        
        dU_conj, dV_conj, fU_conj_next = _cann_dynamics(
            U_conj, V_conj, fU_conj, Wvis_conj, Wrec_conj, I_vis, I_hd, neural_params,
        thresh)
        
        dU_omni, dV_omni, fU_omni_next = _omnidirectional_dynamics(
            U_omni, V_omni, Wconj_omni, fU_conj, I_vest, neural_params,
        thresh)
        
        U_conj_next = U_conj + dt * dU_conj
        V_conj_next = V_conj + dt * dV_conj
        U_omni_next = U_omni + dt * dU_omni
        V_omni_next = V_omni + dt * dV_omni
        
        # Generate Poisson spikes and cast to int8 to save RAM during the scan
        # output shapes before flattening: (N_neurons, 1)
        spikes_conj = random.poisson(subkey_conj, fU_conj_next * dt).astype(jnp.int8)
        spikes_omni = random.poisson(subkey_omni, fU_omni_next * dt).astype(jnp.int8)
        
        new_carry = (key, U_conj_next, U_omni_next, V_conj_next, V_omni_next, fU_conj_next)
        
        # Output arrays to be stacked over time. We flatten them to (N,) first.
        stacked_outputs = (spikes_conj[:, 0], spikes_omni[:, 0])
        
        return new_carry, stacked_outputs
    
    # Initialize the loop (spatial maps are removed)
    init_carry = (rng_key, U_conj0, U_omni0, V_conj0, V_omni0, fU_conj0)
    
    # lax.scan iterates over the first dimension of `traj` (which is `steps`)
    final_carry, (all_spikes_conj, all_spikes_omni) = lax.scan(step, init_carry, traj)
    
    # Unpack final state
    _, U_conj, U_omni, V_conj, V_omni, fU_conj = final_carry
    
    # all_spikes_conj will have shape (steps, N_conj)
    # all_spikes_omni will have shape (steps, N_omni)
    
    
    return U_conj, U_omni, V_conj, V_omni, fU_conj, all_spikes_conj, all_spikes_omni


@jit
def vestibular_input(A_vest, pref_hd, inclination_dir, inc_ang, inc_sHD, pi2, norm):
    return A_vest * (jnp.cos(inc_ang) + jnp.sin(inc_ang) * gaussian(pref_hd, inclination_dir, inc_sHD, pi2))

# def vestibular_input(A_vest, pref_hd, inclination_dir, inc_ang, inc_sHD, pi2):
#     return (1 + A_vest * gaussian(pref_hd, inclination_dir, inc_sHD, pi2))

def run_spiking_simulation_2deep(rng_key, U0, V0, weights, traj, neural_params, 
                                input_params, pos, pref_hd, steps, thresh=.5,
                                inclination_dir = 3*jnp.pi/2):
    """Standard simulation saving spike counts over time using lax.scan."""
    pi2 = 2 * jnp.pi
    pi2_sqrt = jnp.sqrt(pi2)
    Wvis_conj, Wrec_conj, Wconj_omni = weights
    
    U_conj0, U_omni0 = U0
    V_conj0, V_omni0 = V0
    
    tau_inv, tauv_inv, m, k, g, gain = neural_params
    dt, sR, sHD_input, inc_ang, A_vis, A_hd, A_vest, L = input_params
    
    if isinstance(sHD_input,tuple):
        sHD, inc_sHD = sHD_input
    else:
        sHD = sHD_input
        inc_sHD = sHD_input
    
    norm = inc_sHD * pi2_sqrt
    # Firing rate initialization
    U_pos = jnp.maximum(U_conj0, 0) 
    norm_sq = jnp.sum(U_pos ** g)
    fU_conj0 = gain * (U_pos ** g) / (1 + k * norm_sq)
    
    # lax.scan passes the current element of `traj` directly to `traj_step`
    @jit
    def step(carry, traj_step):
        key, U_conj, U_omni, V_conj, V_omni, fU_conj = carry
        
        # Split the key for this step
        key, subkey_conj, subkey_omni = random.split(key, 3)
        
        # Unpack trajectory for this specific time step
        x_pos, y_pos, hd = traj_step[0], traj_step[1], traj_step[2]
        
        # traj_step[:-1] gives [x_pos, y_pos]
        I_vis = A_vis * gaussian2D(pos, traj_step[:-1][:, None], sR, 0)
        I_hd = A_hd * gaussian(pref_hd, hd, sHD, pi2) * vestibular_input(A_vest, pref_hd, inclination_dir, inc_ang, inc_sHD, pi2, norm)
        # I_hd = I_hd * (1 + A_vest * gaussian(pref_hd, inclination_dir, inc_sHD, pi2))
        
        I_vest = 1# + 0*pref_hd
        
        dU_conj, dV_conj, fU_conj_next = _cann_dynamics(
            U_conj, V_conj, fU_conj, Wvis_conj, Wrec_conj, I_vis, I_hd, neural_params,
        thresh)
        
        dU_omni, dV_omni, fU_omni_next = _omnidirectional_dynamics(
            U_omni, V_omni, Wconj_omni, fU_conj, I_vest, neural_params,
        thresh)
        
        U_conj_next = U_conj + dt * dU_conj
        V_conj_next = V_conj + dt * dV_conj
        U_omni_next = U_omni + dt * dU_omni
        V_omni_next = V_omni + dt * dV_omni
        
        # Generate Poisson spikes and cast to int8 to save RAM during the scan
        # output shapes before flattening: (N_neurons, 1)
        spikes_conj = random.poisson(subkey_conj, fU_conj_next * dt).astype(jnp.int8)
        spikes_omni = random.poisson(subkey_omni, fU_omni_next * dt).astype(jnp.int8)
        
        new_carry = (key, U_conj_next, U_omni_next, V_conj_next, V_omni_next, fU_conj_next)
        
        # Output arrays to be stacked over time. We flatten them to (N,) first.
        stacked_outputs = (spikes_conj[:, 0], spikes_omni[:, 0])
        
        return new_carry, stacked_outputs
    
    # Initialize the loop (spatial maps are removed)
    init_carry = (rng_key, U_conj0, U_omni0, V_conj0, V_omni0, fU_conj0)
    
    # lax.scan iterates over the first dimension of `traj` (which is `steps`)
    final_carry, (all_spikes_conj, all_spikes_omni) = lax.scan(step, init_carry, traj)
    
    # Unpack final state
    _, U_conj, U_omni, V_conj, V_omni, fU_conj = final_carry
    
    # all_spikes_conj will have shape (steps, N_conj)
    # all_spikes_omni will have shape (steps, N_omni)
    
    
    return U_conj, U_omni, V_conj, V_omni, fU_conj, all_spikes_conj, all_spikes_omni