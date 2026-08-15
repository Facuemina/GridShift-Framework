# -*- coding: utf-8 -*-
"""
engine.py

Core physics formulations, matrix initializations, and highly 
optimized JAX integration loops for CANN simulations.
"""

from jax import numpy as jnp
from jax import jit, lax, random
from numbers import Number

@jit
def gaussian(x, y, std, L):
    dx = jnp.minimum(jnp.abs(x - y), L - jnp.abs(x - y))
    norm = 1 / jnp.sqrt(2 * jnp.pi * std**2)
    return jnp.exp(-0.5 * (dx / std) ** 2) * norm

@jit
def gaussian2D(X, Y, std, Lx, Ly=0):
    Ly = jnp.where(Ly, Ly, Lx)
    return gaussian(X[:, 0:1], Y[0:1, :], std, Lx) * gaussian(X[:, 1:2], Y[1:2, :], std, Ly)

@jit
def vestibular_input(A_vest, pref_hd, inclination_dir, inc_ang, inc_sHD, pi2, norm):
    return (1 + A_vest * jnp.sin(inc_ang) * norm * gaussian(pref_hd, inclination_dir, inc_sHD, pi2))

def run_spiking_simulation_2layers(rng_key, U0, weights, traj, neural_params, 
                                input_params, pos, pref_hd, steps, thresh=[0,0,0],
                                inclination_dir=3*jnp.pi/2, a_vis=0):
    
    pi2 = 2 * jnp.pi
    pi2_sqrt = jnp.sqrt(pi2)
    
    U_conj10, U_conj20, U_omni0 = U0
    
    if len(weights) == 4:
        Wvis_conj1, Wrec_conj1, Wconj1_conj2, Wconj2_omni = weights
        #Wrec_conj2 = jnp.zeros((len(U_conj20),len(U_conj20)))
    else:
        Wvis_conj1, Wrec_conj1, Wconj1_conj2, Wrec_conj2, Wconj2_omni = weights
    pref_hd1, pref_hd2 = pref_hd
    
    
    tau_inv, k, g, gain = neural_params
    dt, sR, sHD_input, inc_ang, A_vis, A_hd, A_vest = input_params
        
    if isinstance(sHD_input, tuple):
        sHD, inc_sHD = sHD_input
    else:
        sHD = sHD_input
        inc_sHD = sHD_input
        
    if isinstance(a_vis, Number):
        a_vis = jnp.ones(steps)
    if isinstance(thresh, Number):
        thresh = [thresh] * 3
    if isinstance(gain, Number):
        gain = [gain] * 3
        
    norm = inc_sHD * pi2_sqrt
    
    @jit
    def step(carry, seq_vars):
        key, U_conj1, U_conj2, U_omni = carry
        
        key, subkey_conj1, subkey_conj2, subkey_omni = random.split(key, 4)
        
        traj_step, a_vis_step = seq_vars
        x_pos, y_pos, hd = traj_step[0], traj_step[1], traj_step[2]
        
        I_vis = a_vis_step * A_vis * gaussian2D(pos, traj_step[:-1][:, None], sR, 0)
        hd_modulation1 = gaussian(pref_hd1, hd, sHD, pi2)
        hd_modulation2 = gaussian(pref_hd2, hd, sHD, pi2)
        I_vest = vestibular_input(A_vest, pref_hd2, inclination_dir, inc_ang, inc_sHD, pi2, norm)
        
        I_hd1 = A_hd * hd_modulation1
        I_hd2 = A_hd * hd_modulation2
        
        U_pos1 = jnp.maximum(U_conj1 - thresh[0], 0)
        norm_sq = jnp.sum(U_pos1 ** g)
        fU_conj1_next = gain[0] * (U_pos1 ** g) / (1 + k[0] * norm_sq) 
        
        I_input1 = Wvis_conj1 @ I_vis + Wrec_conj1 @ fU_conj1_next
        I_input1 = I_input1 * I_hd1 
        dU_conj1 = (-U_conj1 + I_input1) * tau_inv
        
        U_pos2 = jnp.maximum(U_conj2 - thresh[1], 0)
        norm_sq = jnp.sum(U_pos2 ** g)
        fU_conj2_next = gain[1] * (U_pos2 ** g) / (1 + k[1] * norm_sq) 
        
        I_input2 = I_hd2 * I_vest * (Wconj1_conj2 @ fU_conj1_next)# + Wrec_conj2 @ fU_conj2_next) * 
        dU_conj2 = (-U_conj2 + I_input2) * tau_inv
        
        U_pos = jnp.maximum(U_omni - thresh[2], 0)
        norm_sq = jnp.sum(U_pos ** g)
        fU_omni_next = gain[2] * (U_pos ** g) / (1 + k[2] * norm_sq) 
        
        I_input_omni = Wconj2_omni @ fU_conj2_next
        dU_omni = (-U_omni + I_input_omni) * tau_inv
        
        U_conj1_next = U_conj1 + dt * dU_conj1
        U_conj2_next = U_conj2 + dt * dU_conj2
        U_omni_next = U_omni + dt * dU_omni
        
        spikes_conj1 = random.poisson(subkey_conj1, fU_conj1_next * dt).astype(jnp.int8)
        spikes_conj2 = random.poisson(subkey_conj2, fU_conj2_next * dt).astype(jnp.int8)
        spikes_omni = random.poisson(subkey_omni, fU_omni_next * dt).astype(jnp.int8)
        
        new_carry = (key, U_conj1_next, U_conj2_next, U_omni_next)
        stacked_outputs = (spikes_conj1[:, 0], spikes_conj2[:, 0], spikes_omni[:, 0])
        
        return new_carry, stacked_outputs
    
    init_carry = (rng_key, U_conj10, U_conj20, U_omni0)
    final_carry, (all_spikes_conj1, all_spikes_conj2, all_spikes_omni) = lax.scan(step, init_carry, (traj, a_vis))
    
    _, U_conj1, U_conj2, U_omni = final_carry
    Uf = (U_conj1, U_conj2, U_omni)
    
    return Uf, (all_spikes_conj1, all_spikes_conj2, all_spikes_omni)