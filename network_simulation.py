# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:19:30 2026

@author: Facundo
"""

import jax.numpy as jnp
from jax import random
from src.engine import *
from src.utils import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
#%% Set simulation and network parameters
L = 100 #arena size
l_torus = 40 #hexagonal periodicity
l_asym = 7.5
N_vis_sqrt = 40
N_vis = N_vis_sqrt ** 2

hd_modules = 8
N_conj_sqrt = 12
N_conj = N_conj_sqrt ** 2 * hd_modules
N_omni_sqrt = 10
N_omni = N_omni_sqrt ** 2

std = 5
angle_std = 1.2

tau = 0.01
tauv = 0.1
m = 0.5
k = 0.01
gain = 1
v = 20
dt = 0.003
steps = int(5 * 1e5)

A_vis = 30 * jnp.sqrt(2*jnp.pi*std**2) 
A_hd = jnp.sqrt(2*jnp.pi*angle_std**2)
inclination_ang = jnp.pi/6*0
A_vest = jnp.sqrt(2*jnp.pi*angle_std**2) * jnp.sin(inclination_ang) * 15

#%% Position and phase variables

# simulated rat trajectory and HD
traj = jnp.vstack(generate2D_pos(20, steps, L, L, v, 0.8, dt)).T

print('Done trajectory simulation')

#Preffered head direction for each conjunctive cell
pref_hd = jnp.repeat(jnp.linspace(0,2*jnp.pi,hd_modules,False),  N_conj_sqrt ** 2 ).reshape(-1,1)

#spatial visual input positions
x,y = jnp.meshgrid(jnp.linspace(0,L,N_vis_sqrt,False),
                   jnp.linspace(0,L,N_vis_sqrt,False))
pos = jnp.column_stack((x.ravel(),y.ravel()))

#Conjunctive grid phases
X_phase_conj = jnp.column_stack(
    generate_uniform_toroidal_phase_distribution(
        N_conj_sqrt,N_conj_sqrt,l_torus))

#Omnidirectional grid phases
X_phase_omni = jnp.column_stack(
    generate_uniform_toroidal_phase_distribution(
        N_omni_sqrt,N_omni_sqrt,l_torus))

X_phase_conj_dummy = X_phase_conj.copy()

for i in range(1,hd_modules):
    X_phase_conj = jnp.vstack((X_phase_conj,X_phase_conj_dummy))

#%% Connectivity matrices

# Visual feedforward input to conjunctive cells
Wvis_conj = build_feedforward_connectivity(pos, X_phase_conj,
                                           std, l_torus)

# Recurrent connectivity between conjunctive cells
Wrec_conj = build_torus_connectivity(X_phase_conj, X_phase_conj,
                                     std, l_torus, l_asym = 0)

# Feedforward input from conjunctive to omnidirectional cells
Wconj_omni = jnp.zeros((N_omni,N_conj))
for i in range(hd_modules):
    idx = jnp.arange(i*N_conj_sqrt**2,(i+1)*N_conj_sqrt**2)
    Wconj_omni = Wconj_omni.at[:,idx].set(build_torus_connectivity(X_phase_conj_dummy, X_phase_omni, std, l_torus, l_asym = l_asym, hd_pre=pref_hd[idx[0],0]))

#%% Initiate neural variables

key = random.PRNGKey(20)
init_pos = random.uniform(key,shape=(2,1)) * L

U_conj = (Wvis_conj @ gaussian2D(pos,init_pos,std,L)) * gaussian(pref_hd,jnp.pi/2,angle_std,jnp.pi*2)
V_conj = m * U_conj
fU_conj = jnp.maximum(U_conj,0)**2
fU_conj = fU_conj / ( 1 + k * fU_conj.sum())

U_omni = Wconj_omni @ fU_conj
V_omni = m * U_omni

#%% Run simulation
nfr = 30
thresh = .75
neural_params =( 1/tau, 1/tauv, m, k, 2, gain)
input_params = (dt, std, angle_std, A_vis, A_hd, A_vest, L)
U_conj, U_omni, V_conj, V_omni, fU_conj, rate_map_conj, rate_map_omni = run_basic_simulation((U_conj, U_omni), (V_conj, V_omni), 
                                                                                             (Wvis_conj, Wrec_conj, Wconj_omni), 
                                                                                             traj, neural_params,
                                                                                             input_params, pos, pref_hd, steps,
                                                                                             nfr, nfr, thresh)
#%%
# 2. Compute occupancy
occupancy = compute_occupancy_map(traj, L, nfr, nfr)

# 3. Prevent division by zero for unvisited spatial bins
safe_occupancy = jnp.maximum(occupancy, 1)

# 4. Compute true spatial firing rate maps
# rate_map_conj shape: (N_neurons, nx * ny)
# safe_occupancy shape: (nx * ny,) -> JAX handles the broadcasting automatically
true_rate_map_conj = (rate_map_conj * steps) / safe_occupancy
true_rate_map_omni = (rate_map_omni * steps) / safe_occupancy

np.save(f'true_rate_map_conj_incl_{inclination_ang:-2f}',true_rate_map_conj)
np.save(f'true_rate_map_omni{inclination_ang:-2f}',true_rate_map_omni)

plt.figure(figsize=(5,5))
xplot = jnp.linspace(0,L,30,False)
FRMAP = true_rate_map_omni[2,:].reshape((nfr,nfr))[:,::-1]
plt.pcolormesh(xplot,xplot[:-1],FRMAP[:-1,:])
plt.title(f'Incl ang. = {inclination_ang:.2f}, Max_fr = {FRMAP.max():.2f}Hz')
plt.show()
#%%
population_maps_2d = true_rate_map_conj.reshape((true_rate_map_conj.shape[0], nfr, nfr))[:,:,::-1][:,:-1,1:]

# 2. Apply Gaussian filter across the spatial axes (axes 1 and 2)
# By setting sigma=(0, sigma_bins, sigma_bins), we apply 0 smoothing across the neuron axis,
# and identical 2D smoothing across the spatial (Y, X) axes.
smoothed_population_maps = gaussian_filter(
    population_maps_2d, 
    sigma=(0, 1.5, 1.5)
)

plt.figure(figsize=(5,5))
xplot = jnp.linspace(0,L,30,False)[1:]
FRMAP = smoothed_population_maps[35]#.reshape((nfr,nfr))[:,::-1]
plt.pcolormesh(xplot,xplot,FRMAP)
plt.title(f'Incl ang. = {inclination_ang:.2f}, Max_fr = {FRMAP.max():.2f}Hz')