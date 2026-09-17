# -*- coding: utf-8 -*-
"""
model_final.py
"""

import os
import pickle
import argparse
import numpy as np
import jax.numpy as jnp
from jax import random, jit, lax
from utils import (generate2D_pos, 
                    generate_uniform_toroidal_phase_distribution, 
                    map2torus_fn,
                    distance_torus_sq,
                    build_torus_connectivity,
                    gaussian)

#%% Set simulation and network parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path2save", type=str, default='GridShift-Trial-num0')
parser.add_argument("--path2load_traj", type=str, default='')
parser.add_argument("--l_asym", type=float, default=6)
parser.add_argument("--dt", type=float, default=.005)
parser.add_argument("--inclination_angle", type=float, default=jnp.pi/3*0)
parser.add_argument("--L", type=float, default=50)
parser.add_argument("--l_torus", type=float, default=30)
parser.add_argument("--tau", type=float, default=.01)
parser.add_argument("--steps", type=int, default=int(1e5))
parser.add_argument("--N_spat_sqrt", type=int, default=15)
parser.add_argument("--N_conj_sqrt", type=int, default=15)
parser.add_argument("--N_omni_sqrt", type=int, default=9)
parser.add_argument("--hd_modules", type=int, default=8)
parser.add_argument("--input_std", type=float, default=6)
parser.add_argument("--angle_std", type=float, default=1.5)
parser.add_argument("--inc_angle_std", type=float, default=.6)
parser.add_argument("--k", type=float, default=.1)
parser.add_argument("--gain", type=float, default=10)
parser.add_argument("--v", type=float, default=6)
parser.add_argument("--A_spat", type=float, default=1)
parser.add_argument("--A_conj", type=float, default=1)
parser.add_argument("--A_mod", type=float, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--inclination_dir", type=float, default=3*np.pi/2)
parser.add_argument("--load_traj", type=str, default="False")
parser.add_argument("--thresh_weights", type=float, default=0.)
parser.add_argument("--T0", type=float, default = 40)
parser.add_argument("--T1", type=float, default = 12)

args = parser.parse_args()

path2save = args.path2save
load_traj = args.load_traj
path2load_traj = args.path2load_traj

inclination_angle = args.inclination_angle
L = args.L
l_torus = args.l_torus
l_asym = args.l_asym
N_spat_sqrt = args.N_spat_sqrt
N_conj_sqrt = args.N_conj_sqrt
N_omni_sqrt = args.N_omni_sqrt
hd_modules = args.hd_modules
inclination_dir = args.inclination_dir

N_spat = N_spat_sqrt ** 2 * hd_modules
N_conj = N_conj_sqrt ** 2 * hd_modules
N_omni = N_omni_sqrt ** 2

input_std = args.input_std
angle_std = args.angle_std
inc_angle_std = args.inc_angle_std
A_spat = args.A_spat 
A_conj = args.A_conj 
A_mod = args.A_mod
seed = args.seed
T0 = args.T0
T1 = args.T1

dt = args.dt
steps = args.steps

tau = args.tau
v = args.v
k = args.k
gain = args.gain

thresh_weights = args.thresh_weights
#%% Trajectory
if load_traj == 'False':
    traj = jnp.vstack(generate2D_pos(steps, L, L, v, 0.8, dt)).T
else:
    import scipy.io as sio
    
    path = os.path.join(path2load_traj,'trajectory_60.mat')
    traj_mat = sio.loadmat(path)
    
    t_orig = jnp.arange(len(traj_mat['trajectory']['position_x'][0][0])) * traj_mat['trajectory']['dt'][0][0][0][0]
    
    traj_mat = jnp.hstack((traj_mat['trajectory']['position_x'][0][0],
                          traj_mat['trajectory']['position_y'][0][0],
                          traj_mat['trajectory']['headDirection'][0][0]))
                          
    t_new = jnp.arange(0, t_orig[-1], dt) 
    hd_unwrapped = jnp.unwrap(np.mod(traj_mat[:,2], 2*np.pi))
    
    x_pos = jnp.interp(t_new, t_orig, traj_mat[:,0])
    y_pos = jnp.interp(t_new, t_orig, traj_mat[:,1])
    hd = jnp.mod(np.interp(t_new, t_orig, hd_unwrapped), 2*np.pi)
    traj = jnp.column_stack((x_pos, y_pos, hd))
    steps = traj.shape[0]

#%% Position and phase variables
# Preferred head direction for each conjunctive cell
pref_hd_conj = jnp.repeat(jnp.linspace(0, 2*jnp.pi, hd_modules+1)[:-1], N_conj_sqrt**2).reshape(-1,1)
pref_hd_spat = jnp.repeat(jnp.linspace(0, 2*jnp.pi, hd_modules+1)[:-1], N_spat_sqrt**2).reshape(-1,1)

# Spatial visual input positions
l_buffer = L*.2
x, y = jnp.meshgrid(jnp.linspace(-l_buffer, L+l_buffer, N_spat_sqrt),
                    jnp.linspace(-l_buffer, L+l_buffer, N_spat_sqrt))
pos = jnp.column_stack((x.ravel(), y.ravel()))

# Conjunctive grid phases 1
X_phase_conj = jnp.column_stack(generate_uniform_toroidal_phase_distribution(N_conj_sqrt, N_conj_sqrt, l_torus))

# Omnidirectional grid phases
X_phase_omni = jnp.column_stack(generate_uniform_toroidal_phase_distribution(N_omni_sqrt, N_omni_sqrt, l_torus))
 
#%% Connectivity matrices

traj_torus =  np.vstack(map2torus_fn(traj[:,0],
                        traj[:,1],
                        l_torus)
                        ).T[None, :, :]
pos_torus =  np.vstack(map2torus_fn(pos[:,0],
                        pos[:,1],
                        l_torus)
                        ).T[:, None, :]
d_spat =  distance_torus_sq(pos_torus,
                           traj_torus, l_torus)

d_conj = distance_torus_sq(X_phase_conj[:,None,:],
                           traj_torus, l_torus)

W_conj_omni = []
W_spat_omni = []
for i in range(hd_modules):
    Mc = build_torus_connectivity(X_phase_conj, 
                                 X_phase_omni, input_std, 
                                 l_torus, 
                                 l_asym=l_asym, 
                                 hd_pre=pref_hd_conj[i*N_conj_sqrt**2:(i+1)*N_conj_sqrt**2].ravel())
    
    W_conj_omni.append(jnp.maximum(Mc - thresh_weights,0)/(1-thresh_weights))
    
    Ms = build_torus_connectivity(pos_torus[:,0,:], 
                                 X_phase_omni, input_std, 
                                 l_torus, 
                                 l_asym= l_asym,
                                 hd_pre=pref_hd_spat[i*N_spat_sqrt**2:(i+1)*N_spat_sqrt**2].ravel())
    
    W_spat_omni.append(jnp.maximum(Ms - thresh_weights,0)/(1-thresh_weights))


#%% Set neural currents
Iinh = (T0 + T1 * jnp.sin(inclination_angle) ) * jnp.ones(steps)

I_spat_list = jnp.zeros((hd_modules,N_omni,steps))
I_conj_list = jnp.zeros((hd_modules,N_omni,steps))


for i, w in enumerate(zip(W_spat_omni,W_conj_omni)):
    wspat, wconj = w
    gaussian_tuning = gaussian(inclination_dir,pref_hd_spat[i*N_spat_sqrt**2,0],inc_angle_std,2*jnp.pi) * inc_angle_std * jnp.sqrt(2*jnp.pi)
    spat_inc_modulation = 1 + A_mod * jnp.sin(inclination_angle) * gaussian_tuning
    
    hd_spat = gaussian(traj[:,2],pref_hd_spat[i*N_spat_sqrt**2],angle_std,2*jnp.pi) * angle_std * jnp.sqrt(2*jnp.pi)
    hd_conj = gaussian(traj[:,2],pref_hd_conj[i*N_conj_sqrt**2],angle_std,2*jnp.pi) * angle_std * jnp.sqrt(2*jnp.pi)
    
    I_spat = wspat @ jnp.exp(-.5 * (d_spat/input_std)**2) * hd_spat
    I_conj = wconj @ jnp.exp(-.5 * (d_conj/input_std)**2) * hd_conj
    
    I_spat_list = I_spat_list.at[i].set(spat_inc_modulation * I_spat)
    I_conj_list = I_conj_list.at[i].set(I_conj)
    
        
U_omni = A_spat * I_spat_list[:,:,0].sum(axis = 0) + A_conj * I_conj_list[:,:,0].sum(axis = 0) - Iinh[0]

#%% Run simulation
tau_inv = 1/tau
@jit
def step(U_omni, seq_vars):

    I_spat, I_conj, Iinh = seq_vars
    
    dU_omni = (-U_omni + I_spat + I_conj - Iinh) * tau_inv          
    U_omni_next = U_omni + dt * dU_omni
    U_pos = jnp.maximum(U_omni_next,0) ** 2
    fU_omni_next = gain * (U_pos) / (1 + k * U_pos.sum()) 
    
    return U_omni_next, fU_omni_next
    
final_U_omni, firing_rates = lax.scan(step, U_omni, (I_spat_list.sum(axis = 0).T, I_conj_list.sum(axis=0).T, Iinh))

#%% save spikes into sparse arrays
I0 = 2000
key = random.PRNGKey(seed)
spikes = random.poisson(key, firing_rates[I0:] * dt)

#%%
dense_spikes = np.array(spikes)

time_indices, neuron_indices = np.nonzero(dense_spikes)

counts = dense_spikes[time_indices, neuron_indices]

sparse_spike_data = {
    'time_idx': time_indices,
    'neuron_idx': neuron_indices,
    'counts': counts
}

#%% Save data
with open(os.path.join(path2save,"sparse_spikes.pkl"), "wb") as f:
    pickle.dump(sparse_spike_data, f)

np.save(os.path.join(path2save,'traj'), traj[I0:,:])

parameters = {
    "path2save": path2save,
    "l_asym": l_asym,
    "dt": dt, 
    "inclination_angle": inclination_angle,
    "L": L,
    "l_torus": l_torus,
    "tau": tau,
    "steps": steps,
    "N_spat_sqrt": N_spat_sqrt,
    "N_conj_sqrt": N_conj_sqrt,
    "N_omni_sqrt": N_omni_sqrt,
    "hd_modules": hd_modules,
    "input_std": input_std,
    "angle_std": angle_std,
    "inc_angle_std": inc_angle_std,
    "k": k,
    "gain": gain,
    "v": v,
    "A_spat": A_spat,
    "A_mod": A_mod,
    "A_conj": A_conj,
    "seed": seed,
    "inclination_dir": inclination_dir,
    "load_traj": load_traj,
    "thresh_weights": thresh_weights,
    "T0":T0, "T1":T1}

with open(os.path.join(path2save,'parameters_complete.pkl'),'wb') as file:
    pickle.dump(parameters,file)