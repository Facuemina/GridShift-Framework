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
import argparse
import os
import pickle
from numbers import Number
#%% Set simulation and network parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path2save", type=str, default='GridShift-Trial-num0')#'GridShift-Minimal-num')
parser.add_argument("--firingrate", type=str, default='False')#'GridShift-Minimal-num')
parser.add_argument("--l_asym", type=float, default=4)
parser.add_argument("--dt", type=float, default=.01)
parser.add_argument("--inclination_angle", 
                    type=float, default=jnp.pi/3)
parser.add_argument("--L", type=float, default=50)
parser.add_argument("--l_torus", type=float, default=25)
parser.add_argument("--tau", type=float, default=.01)
parser.add_argument("--tauv", type=float, default=.1)
parser.add_argument("--steps", type=int, default=int(5*1e4))
parser.add_argument("--N_vis_sqrt", type=int, default=40)
parser.add_argument("--N_conj_sqrt", type=int, default=10)
parser.add_argument("--N_omni_sqrt", type=int, default=10)
parser.add_argument("--hd_modules", type=int, default=8)
parser.add_argument("--input_std", type=float, default=4)
parser.add_argument("--angle_std", type=float, default=1.2)
parser.add_argument("--inc_angle_std", type=float, default=1.2)
parser.add_argument("--k", type=float, default=.01)
parser.add_argument("--m", type=float, default=.5)
parser.add_argument("--gain", type=float, default=1.)
parser.add_argument("--v", type=float, default=20)
parser.add_argument("--A_vis", type=float, default=30)
parser.add_argument("--A_hd", type=float, default=1)
parser.add_argument("--A_vest", type=float, default=15)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nfr", type=int, default=30)
parser.add_argument("--thresh", type=float, nargs = 2, default= [0,150] )
parser.add_argument("--ACTIVATION_EXP", type=float, default = 2)
parser.add_argument("--superficial", type=str, default = 'True')
parser.add_argument("--inclination_dir", type=float, default = 'inclination_dir')
parser.add_argument("--load_traj", type=str, default="False")

args = parser.parse_args()

path2save = args.path2save
firingrate = args.firingrate
superficial = args.superficial
load_traj = args.load_traj

inclination_angle = args.inclination_angle
L = args.L #arena size
l_torus = args.l_torus #hexagonal periodicity
l_asym = args.l_asym
N_vis_sqrt = args.N_vis_sqrt
N_conj_sqrt = args.N_conj_sqrt
N_omni_sqrt = args.N_omni_sqrt
hd_modules = args.hd_modules
inclination_dir = args.inclination_dir


N_vis = N_vis_sqrt ** 2

N_conj_sqrt_x = 0 + N_conj_sqrt
N_conj_sqrt_y = int(N_conj_sqrt +0* jnp.sin(jnp.pi/3))
N_conj = N_conj_sqrt_x * N_conj_sqrt_y * hd_modules
N_omni_sqrt_x = 0 + N_omni_sqrt
N_omni_sqrt_y = int(N_omni_sqrt +0* jnp.sin(jnp.pi/3))
N_omni = N_omni_sqrt_x * N_omni_sqrt_y

input_std = args.input_std
angle_std = args.angle_std
inc_angle_std = args.inc_angle_std
A_vis = args.A_vis * jnp.sqrt(2*jnp.pi*args.input_std**2)
A_hd = args.A_hd * jnp.sqrt(2*jnp.pi*args.angle_std**2)
A_vest = args.A_vest #* jnp.sqrt(2*jnp.pi*angle_std**2) * jnp.sin(inclination_angle)
seed = args.seed
nfr = args.nfr
thresh = args.thresh

dt = args.dt
steps = args.steps

m = args.m
tau = args.tau
tauv = args.tauv
v = args.v
k = args.k
gain = args.gain

ACTIVATION_EXP = args.ACTIVATION_EXP
#%% Position and phase variables

# simulated rat trajectory and HD
if load_traj == 'False':
    traj = jnp.vstack(generate2D_pos(seed, steps, L, L, v, 0.8, dt)).T
else:
    import scipy.io as sio
    from scipy.interpolate import interp1d, CubicSpline
    path = r'C:\Users\Facundo\Desktop\Facu\Doctorado\PythonCodes\2D-CANN\Miao\Miao-files'
    path = os.path.join(path,'trajectory_60.mat')
    traj = sio.loadmat(path)
    
    t_orig = np.arange(len(traj['trajectory']['position_x'][0][0])) * traj['trajectory']['dt'][0][0][0][0]
    
    traj = np.hstack((traj['trajectory']['position_x'][0][0],
                      traj['trajectory']['position_y'][0][0],
                      traj['trajectory']['headDirection'][0][0]))
    # Create new time vector up to the last original time point
    t_new = np.arange(0, t_orig[-1], dt) 
    
    # --- Cubic Spline ---
    cs = CubicSpline(t_orig, traj, axis=0)
    traj = cs(t_new)
    steps = len(traj)
    
#Preffered head direction for each conjunctive cell
pref_hd = jnp.repeat(jnp.linspace(0,2*jnp.pi,hd_modules+1)[:-1],  N_conj_sqrt_x*N_conj_sqrt_y).reshape(-1,1)

#spatial visual input positions
x,y = jnp.meshgrid(jnp.linspace(0,L,N_vis_sqrt,False),
                   jnp.linspace(0,L,N_vis_sqrt,False))
pos = jnp.column_stack((x.ravel(),y.ravel()))

# # Rotate
# pos = jnp.column_stack((-x.ravel() * jnp.cos(jnp.pi/5) + y.ravel() * jnp.sin(jnp.pi/5),
#                         y.ravel() * jnp.cos(jnp.pi/5) + x.ravel() * jnp.sin(jnp.pi/5)))

# traj = traj.at[:,:-1].set(traj[:,:-1] @ jnp.array([[jnp.cos(jnp.pi/5),-jnp.sin(jnp.pi/5)],[jnp.sin(jnp.pi/5),jnp.cos(jnp.pi/5)]]))
# traj = traj.at[:,-1].set(traj[:,-1] + jnp.pi/5)


#Conjunctive grid phases
X_phase_conj = jnp.column_stack(
    generate_uniform_toroidal_phase_distribution(
        N_conj_sqrt_x,N_conj_sqrt_y,l_torus))

#Omnidirectional grid phases
X_phase_omni = jnp.column_stack(
    generate_uniform_toroidal_phase_distribution(
        N_omni_sqrt_x,N_omni_sqrt_y,l_torus))

X_phase_conj_dummy = X_phase_conj.copy()

for i in range(1,hd_modules):
    X_phase_conj = jnp.vstack((X_phase_conj,X_phase_conj_dummy))

#%% Connectivity matrices

# Visual feedforward input to conjunctive cells
Wvis_conj = build_feedforward_connectivity(pos, X_phase_conj,
                                           input_std, l_torus)

# Recurrent connectivity between conjunctive cells
Wrec_conj = build_torus_connectivity(X_phase_conj, X_phase_conj,
                                     input_std, l_torus, l_asym = 0) * gaussian(pref_hd,pref_hd.T,angle_std,jnp.pi*2)

# Feedforward input from conjunctive to omnidirectional cells
Wconj_omni = jnp.zeros((N_omni,N_conj))
for i in range(hd_modules):
    idx = jnp.arange(i*N_conj_sqrt_x*N_conj_sqrt_y,(i+1)*N_conj_sqrt_x*N_conj_sqrt_y)
    Wconj_omni = Wconj_omni.at[:,idx].set(build_torus_connectivity(X_phase_conj_dummy, X_phase_omni, input_std, l_torus, l_asym = l_asym, hd_pre=pref_hd[idx[0],0]))

# Wconj_omni = Wconj_omni**2
#%% Initiate neural variables
# # Anti Rotate
# pos = jnp.column_stack((-x.ravel() * jnp.cos(-jnp.pi/5) + y.ravel() * jnp.sin(-jnp.pi/5),
#                         y.ravel() * jnp.cos(-jnp.pi/5) + x.ravel() * jnp.sin(-jnp.pi/5)))


init_pos = traj[0:1,:-1].T

U_conj = (Wvis_conj @ gaussian2D(pos,init_pos,input_std,L)) * gaussian(pref_hd,jnp.pi/2,angle_std,jnp.pi*2)
V_conj = m * U_conj
if isinstance(thresh,Number):
    if thresh >0:
        fU_conj = jnp.maximum(U_conj - jnp.quantile(U_conj,thresh),0)**2
    else:
        fU_conj = jnp.maximum(U_conj,0)**2
elif len(thresh)>1:
    fU_conj = jnp.maximum(U_conj - thresh[0], 0)**2
else:
    fU_conj = jnp.maximum(U_conj - jnp.quantile(U_conj,thresh[0]),0)**2
    
fU_conj = fU_conj / ( 1 + k * fU_conj.sum())

U_omni = Wconj_omni @ fU_conj
V_omni = m * U_omni

#%% Run simulation
# nfr = 30
# ACTIVATION_EXP = 2
neural_params =( 1/tau, 1/tauv, m, k, ACTIVATION_EXP, gain)
input_params = (dt, input_std, (angle_std, inc_angle_std), inclination_angle, A_vis, A_hd, A_vest, L)

if firingrate == 'True':
    # 1. Run simulation
    U_conj, U_omni, V_conj, V_omni, fU_conj, rate_map_conj, rate_map_omni = run_rate_simulation((U_conj, U_omni), (V_conj, V_omni), 
                                                                                                 (Wvis_conj, Wrec_conj, Wconj_omni), 
                                                                                                 traj, neural_params, 
                                                                                                 input_params, pos, 
                                                                                                 pref_hd, steps,
                                                                                                 nfr, nfr, thresh)
    
    # 2. Compute occupancy
    occupancy = compute_occupancy_map(traj, L, nfr, nfr)
    
    # 3. Prevent division by zero for unvisited spatial bins
    safe_occupancy = jnp.maximum(occupancy, 1)
    
    # 4. Compute true spatial firing rate maps
    # rate_map_conj shape: (N_neurons, nx * ny)
    # safe_occupancy shape: (nx * ny,) -> JAX handles the broadcasting automatically
    true_rate_map_conj = np.array((rate_map_conj * steps) / safe_occupancy)
    true_rate_map_omni = np.array((rate_map_omni * steps) / safe_occupancy)
    
    np.save(os.path.join(path2save,'true_rate_map_conj'),true_rate_map_conj)
    np.save(os.path.join(path2save,'true_rate_map_omni'),true_rate_map_omni)
else:
    _, _, _, _, KEY = random.split(random.PRNGKey(seed),5)
    
    if superficial == 'True':
        print('2sup function')
        U_conj, U_omni, V_conj, V_omni, fU_conj, spikes_conj, spikes_omni = run_spiking_simulation_2sup(KEY, (U_conj, U_omni), (V_conj, V_omni), 
                                                                                                     (Wvis_conj, Wrec_conj, Wconj_omni), 
                                                                                                     traj, neural_params,                                                                                        input_params, pos, pref_hd, steps,
                                                                                                     thresh, inclination_dir)
    else:
        print('2deep function')
        U_conj, U_omni, V_conj, V_omni, fU_conj, spikes_conj, spikes_omni = run_spiking_simulation_2deep(KEY, (U_conj, U_omni), (V_conj, V_omni), 
                                                                                                     (Wvis_conj, Wrec_conj, Wconj_omni), 
                                                                                                     traj, neural_params,                                                                                        input_params, pos, pref_hd, steps,
                                                                                                     thresh, inclination_dir)
    
    # Convert JAX array to NumPy
    dense_spikes_omni = np.array(spikes_omni)
    dense_spikes_conj = np.array(spikes_conj)
    
    # Get the indices (time step, neuron index) where spikes > 0
    time_indices_omni, neuron_indices_omni = np.nonzero(dense_spikes_omni)
    time_indices_conj, neuron_indices_conj = np.nonzero(dense_spikes_conj)
    
    # Get the actual counts for those indices
    counts_omni = dense_spikes_omni[time_indices_omni, neuron_indices_omni]
    counts_conj = dense_spikes_conj[time_indices_conj, neuron_indices_conj]
    
    # Package them into a space-efficient dictionary
    sparse_spike_data_omni = {
        'time_idx': time_indices_omni,
        'neuron_idx': neuron_indices_omni,
        'counts': counts_omni
    }
    sparse_spike_data_conj = {
        'time_idx': time_indices_conj,
        'neuron_idx': neuron_indices_conj,
        'counts': counts_conj
    }
    with open(os.path.join(path2save,"sparse_spikes_omni.pkl"), "wb") as f:
        pickle.dump(sparse_spike_data_omni, f)
    with open(os.path.join(path2save,"sparse_spikes_conj.pkl"), "wb") as f:
        pickle.dump(sparse_spike_data_conj, f)
#%%
np.save(os.path.join(path2save,'traj'),traj)

parameters = {
    "d_asym":l_asym,
    "dt":dt,
    "inclination_angle": inclination_angle,
    "L":L,
    "l_torus":l_torus,
    "tau":tau, 
    "tauv":tauv,
    "steps":steps,
    "N_vis_sqrt": N_vis_sqrt, 
    "N_conj_sqrt":N_conj_sqrt, 
    "N_omni_sqrt":N_omni_sqrt,
    "hd_modules":hd_modules, 
    "input_std":input_std,
    "inc_angle_std": inc_angle_std,
    "angle_std":angle_std,
    "k":k,
    "m":m,
    "gain":gain, 
    "v":v,
    "A_vis":A_vis, 
    "A_hd":A_hd,
    "A_vest":A_vest,
    "seed":seed,
    "thresh":thresh,
    "inclination_dir":inclination_dir}

with open(os.path.join(path2save,'parameters_complete.pkl'),'wb') as file:
    pickle.dump(parameters,file)