# -*- coding: utf-8 -*-
"""
model.py
"""

import os
import pickle
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from numbers import Number

from src.engine import *
from src.utils import *

#%% Set simulation and network parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path2save", type=str, default='GridShift-Trial-num0')
parser.add_argument("--l_asym", type=float, default=8)
parser.add_argument("--dt", type=float, default=.005)
parser.add_argument("--inclination_angle", type=float, default=jnp.pi/3)
parser.add_argument("--L", type=float, default=50)
parser.add_argument("--l_torus", type=float, default=30)
parser.add_argument("--tau", type=float, default=.01)
parser.add_argument("--steps", type=int, default=int(5*1e4))
parser.add_argument("--N_vis_sqrt", type=int, default=40)
parser.add_argument("--N_conj_sqrt1", type=int, default=15)
parser.add_argument("--N_conj_sqrt2", type=int, default=9)
parser.add_argument("--N_omni_sqrt", type=int, default=9)
parser.add_argument("--hd_modules", type=int, default=8)
parser.add_argument("--input_std", type=float, default=6)
parser.add_argument("--angle_std", type=float, default=1.)
parser.add_argument("--inc_angle_std", type=float, default=1.)
parser.add_argument("--k", type=float, nargs = 3, default=.01)
parser.add_argument("--gain", type=float, nargs = 3, default=[12,12,1.2])
parser.add_argument("--v", type=float, default=6)
parser.add_argument("--A_vis", type=float, default=30)
parser.add_argument("--A_hd", type=float, default=1)
parser.add_argument("--A_vest", type=float, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--thresh", type=float, nargs=3, default=[0,0,150])
parser.add_argument("--ACTIVATION_EXP", type=float, default=2)
parser.add_argument("--inclination_dir", type=float, default=-np.pi/2)
parser.add_argument("--load_traj", type=str, default="False")
parser.add_argument("--thresh_weights", type=float, nargs=4, default=[0.3,0.3,0.3,0.3])

args = parser.parse_args()

path2save = args.path2save
load_traj = args.load_traj

inclination_angle = args.inclination_angle
L = args.L
l_torus = args.l_torus
l_asym = args.l_asym
N_vis_sqrt = args.N_vis_sqrt
N_conj_sqrt1 = args.N_conj_sqrt1
N_conj_sqrt2 = args.N_conj_sqrt2
N_omni_sqrt = args.N_omni_sqrt
hd_modules = args.hd_modules
inclination_dir = args.inclination_dir

N_vis = N_vis_sqrt ** 2
N_conj1 = N_conj_sqrt1 ** 2 * hd_modules
N_conj2 = N_conj_sqrt2 ** 2 * hd_modules
N_omni = N_omni_sqrt ** 2

input_std = args.input_std
angle_std = args.angle_std
inc_angle_std = args.inc_angle_std
A_vis = args.A_vis * jnp.sqrt(2*jnp.pi*args.input_std**2)
A_hd = args.A_hd * jnp.sqrt(2*jnp.pi*args.angle_std**2)
A_vest = args.A_vest
seed = args.seed
thresh = args.thresh

dt = args.dt
steps = args.steps

tau = args.tau
v = args.v
k = args.k
gain = args.gain

thresh_weights = args.thresh_weights
ACTIVATION_EXP = args.ACTIVATION_EXP

#%% Trajectory
if load_traj == 'False':
    traj = jnp.vstack(generate2D_pos(seed, steps, L, L, v, 0.8, dt)).T
else:
    import scipy.io as sio
    path = r'C:\Users\Facundo\Desktop\Facu\Doctorado\PythonCodes\2D-CANN\Miao\Miao-files'
    path = os.path.join(path,'trajectory_60.mat')
    traj_mat = sio.loadmat(path)
    
    t_orig = np.arange(len(traj_mat['trajectory']['position_x'][0][0])) * traj_mat['trajectory']['dt'][0][0][0][0]
    
    traj_mat = np.hstack((traj_mat['trajectory']['position_x'][0][0],
                          traj_mat['trajectory']['position_y'][0][0],
                          traj_mat['trajectory']['headDirection'][0][0]))
                          
    t_new = np.arange(0, t_orig[-1], dt) 
    hd_unwrapped = np.unwrap(np.mod(traj_mat[:,2], 2*np.pi))
    
    x_pos = np.interp(t_new, t_orig, traj_mat[:,0])
    y_pos = np.interp(t_new, t_orig, traj_mat[:,1])
    hd = np.mod(np.interp(t_new, t_orig, hd_unwrapped), 2*np.pi)
    traj = np.column_stack((x_pos, y_pos, hd))

#%% Position and phase variables
# Preferred head direction for each conjunctive cell
pref_hd1 = jnp.repeat(jnp.linspace(0, 2*jnp.pi, hd_modules+1)[:-1], N_conj_sqrt1**2).reshape(-1,1)
pref_hd2 = jnp.repeat(jnp.linspace(0, 2*jnp.pi, hd_modules+1)[:-1], N_conj_sqrt2**2).reshape(-1,1)

# Spatial visual input positions
l_buffer = L*.2
x, y = jnp.meshgrid(jnp.linspace(-l_buffer, L+l_buffer, N_vis_sqrt),
                    jnp.linspace(-l_buffer, L+l_buffer, N_vis_sqrt))
pos = jnp.column_stack((x.ravel(), y.ravel()))

# Conjunctive grid phases 1
X_phase_conj1 = jnp.column_stack(generate_uniform_toroidal_phase_distribution(N_conj_sqrt1, N_conj_sqrt1, l_torus))

# Conjunctive grid phases 2
X_phase_conj2 = jnp.column_stack(generate_uniform_toroidal_phase_distribution(N_conj_sqrt2, N_conj_sqrt2, l_torus))

# Omnidirectional grid phases
X_phase_omni = jnp.column_stack(generate_uniform_toroidal_phase_distribution(N_omni_sqrt, N_omni_sqrt, l_torus))

X_phase_conj1_dummy = X_phase_conj1.copy()
X_phase_conj2_dummy = X_phase_conj2.copy()

for i in range(1, hd_modules):
    X_phase_conj1 = jnp.vstack((X_phase_conj1, X_phase_conj1_dummy))
    X_phase_conj2 = jnp.vstack((X_phase_conj2, X_phase_conj2_dummy))

#%% Connectivity matrices
Wvis_conj1 = build_feedforward_connectivity(pos, X_phase_conj1, input_std, l_torus)
Wvis_conj1 = jnp.maximum(Wvis_conj1-thresh_weights[0], 0) / (1.0 - thresh_weights[0])

Wrec_conj1 = build_torus_connectivity(X_phase_conj1, X_phase_conj1, input_std,
                                      l_torus, l_asym=1.07*v*tau, hd_pre=pref_hd1.ravel()) 
Wrec_conj1 = jnp.maximum(Wrec_conj1-thresh_weights[1], 0) / (1.0 - thresh_weights[1])
Wrec_conj1 = Wrec_conj1 * gaussian(pref_hd1, pref_hd1.T, angle_std, jnp.pi*2) * angle_std*jnp.sqrt(jnp.pi*2)

Wconj1_conj2 = build_torus_connectivity(X_phase_conj1, X_phase_conj2, input_std,
                                        l_torus, l_asym=1.07*v*tau, hd_pre=pref_hd1.ravel())
Wconj1_conj2 = jnp.maximum(Wconj1_conj2-thresh_weights[2], 0) / (1.0 - thresh_weights[2])
Wconj1_conj2 = .01 * Wconj1_conj2 * gaussian(pref_hd2, pref_hd1.T, angle_std, jnp.pi*2) * angle_std*jnp.sqrt(jnp.pi*2)

Wrec_conj2 = build_torus_connectivity(X_phase_conj2, X_phase_conj2, input_std,
                                      l_torus, l_asym=1.07*v*tau, hd_pre=pref_hd2.ravel()) 
Wrec_conj2 = jnp.maximum(Wrec_conj2-thresh_weights[1], 0) / (1.0 - thresh_weights[2])
Wrec_conj2 = .1 * Wrec_conj2 * gaussian(pref_hd2, pref_hd2.T, angle_std, jnp.pi*2) * angle_std*jnp.sqrt(jnp.pi*2)


Wconj2_omni = build_torus_connectivity(X_phase_conj2, X_phase_omni, input_std, 
                                       l_torus, l_asym=l_asym, hd_pre=pref_hd2.ravel())
Wconj2_omni = jnp.maximum(Wconj2_omni-thresh_weights[3], 0) / (1.0 - thresh_weights[3])

#%% Initiate neural variables
init_pos = traj[0:1, :-1].T

U_conj1 = (Wvis_conj1 @ gaussian2D(pos, init_pos, input_std, L)) * gaussian(pref_hd1, jnp.pi/2, angle_std, jnp.pi*2)
fU_conj1 = jnp.maximum(U_conj1 - thresh[0], 0)**2    
fU_conj1 = fU_conj1 / (1 + k[0] * fU_conj1.sum())

U_conj2 = (Wconj1_conj2 @ fU_conj1) * gaussian(pref_hd2, jnp.pi/2, angle_std, jnp.pi*2)
fU_conj2 = jnp.maximum(U_conj2 - thresh[1], 0)**2
fU_conj2 = fU_conj2 / (1 + k[1] * fU_conj2.sum())

U_omni = Wconj2_omni @ fU_conj2

#%% Run simulation
neural_params = (1/tau, k, ACTIVATION_EXP, gain)
input_params = (dt, input_std, (angle_std, inc_angle_std), inclination_angle, A_vis, A_hd, A_vest)
weights = (Wvis_conj1, Wrec_conj1, Wconj1_conj2, Wconj2_omni)
initial_network_state = (U_conj1, U_conj2, U_omni)
pref_hd = (pref_hd1, pref_hd2)

_, _, _, _, KEY = random.split(random.PRNGKey(seed), 5)

a_vis = 1/(1 + jnp.exp((jnp.arange(steps)-steps*.1)*dt/5)) #slowly turn off visual input to anchor grid cells

final_network_state, rate_maps = run_spiking_simulation_2layers(KEY,
                                              initial_network_state,
                                              weights,
                                              traj, 
                                              neural_params, 
                                              input_params, 
                                              pos, pref_hd,
                                              steps, 
                                              thresh=thresh, a_vis=a_vis)

# save spikes into sparse arrays
spikes_conj1, spikes_conj2, spikes_omni = rate_maps

dense_spikes_conj1 = np.array(spikes_conj1)
dense_spikes_conj2 = np.array(spikes_conj2)
dense_spikes_omni = np.array(spikes_omni)

time_indices_conj1, neuron_indices_conj1 = np.nonzero(dense_spikes_conj1)
time_indices_conj2, neuron_indices_conj2 = np.nonzero(dense_spikes_conj2)
time_indices_omni, neuron_indices_omni = np.nonzero(dense_spikes_omni)

counts_omni = dense_spikes_omni[time_indices_omni, neuron_indices_omni]
counts_conj1 = dense_spikes_conj1[time_indices_conj1, neuron_indices_conj1]
counts_conj2 = dense_spikes_conj2[time_indices_conj2, neuron_indices_conj2]

sparse_spike_data_omni = {
    'time_idx': time_indices_omni,
    'neuron_idx': neuron_indices_omni,
    'counts': counts_omni
}
sparse_spike_data_conj1 = {
    'time_idx': time_indices_conj1,
    'neuron_idx': neuron_indices_conj1,
    'counts': counts_conj1
}
sparse_spike_data_conj2 = {
    'time_idx': time_indices_conj2,
    'neuron_idx': neuron_indices_conj2,
    'counts': counts_conj2
}

#%% Un-comment to plot data
# rm1 = compute_rate_maps_from_sparse_utils(sparse_spike_data_conj1, traj[i_0:i_f], jnp.arange(N_conj1), L, dt, nx=50, ny=50)
# rm2 = compute_rate_maps_from_sparse_utils(sparse_spike_data_conj2, traj[i_0:i_f], jnp.arange(N_conj2), L, dt, nx=50, ny=50)
# rmomni = compute_rate_maps_from_sparse_utils(sparse_spike_data_omni, traj[i_0:i_f], jnp.arange(N_omni), L, dt, nx=50, ny=50)

# plt.figure(figsize=(15,5))
# plt.subplot(131)
# plt.imshow(rm1[0])
# plt.subplot(132)
# plt.imshow(rm2[0])
# plt.subplot(133)
# plt.imshow(rmomni[0])

#%% Save data
with open(os.path.join(path2save,"sparse_spikes_omni.pkl"), "wb") as f:
    pickle.dump(sparse_spike_data_omni, f)
with open(os.path.join(path2save,"sparse_spikes_conj1.pkl"), "wb") as f:
    pickle.dump(sparse_spike_data_conj1, f)
with open(os.path.join(path2save,"sparse_spikes_conj2.pkl"), "wb") as f:
    pickle.dump(sparse_spike_data_conj2, f)

np.save(os.path.join(path2save,'traj'), traj)

parameters = {
    "path2save": path2save,
    "l_asym": l_asym,
    "dt": dt, 
    "inclination_angle": inclination_angle,
    "L": L,
    "l_torus": l_torus,
    "tau": tau,
    "steps": steps,
    "N_vis_sqrt": N_vis_sqrt,
    "N_conj_sqrt1": N_conj_sqrt1,
    "N_conj_sqrt2": N_conj_sqrt2,
    "N_omni_sqrt": N_omni_sqrt,
    "hd_modules": hd_modules,
    "input_std": input_std,
    "angle_std": angle_std,
    "inc_angle_std": inc_angle_std,
    "k": k,
    "gain": gain,
    "v": v,
    "A_vis": A_vis,
    "A_hd": A_hd,
    "A_vest": A_vest,
    "seed": seed,
    "thresh": thresh,
    "ACTIVATION_EXP": ACTIVATION_EXP,
    "inclination_dir": inclination_dir,
    "load_traj": load_traj,
    "thresh_weights": thresh_weights}

with open(os.path.join(path2save,'parameters_complete.pkl'),'wb') as file:
    pickle.dump(parameters,file)