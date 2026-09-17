# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:08:07 2026

@author: Facundo
"""

import sys
import subprocess
from multiprocessing import set_start_method
import jax.numpy as jnp
import os
from tqdm import tqdm
import pickle
import numpy as np

def run_job(path2save, l_asym, l_torus, inclination_angle, 
            L=50, dt=0.005, steps=int(5 * 1e4),
            N_spat_sqrt=15, N_conj_sqrt=15, N_omni_sqrt=9,
            k=.1, gain=1, v=6, 
            A_mod=3, A_conj=1, A_spat=1,
            seed=0,
            tau=0.01, 
            input_std=6, angle_std=1.5, inc_angle_std=1., 
            hd_modules=8,
            inclination_dir=1.5*np.pi, load_traj='True',
            thresh_weights=0.,T0=17,T1=20):
    
    path2load_traj = r'C:\Users\Facundo\Desktop\Facu\Doctorado\PythonCodes\2D-CANN\Miao\Miao-files' 
    
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "simulation.py",
        f"--path2save={path2save}",
        F"--path2load_traj={path2load_traj}",
        f"--l_asym={l_asym}",
        f"--l_torus={l_torus}",
        f"--inclination_angle={inclination_angle}",
        f"--dt={dt}",
        f"--L={L}",
        f"--tau={tau}", 
        f"--steps={steps}",
        f"--N_spat_sqrt={N_spat_sqrt}", 
        f"--N_conj_sqrt={N_conj_sqrt}",
        f"--N_omni_sqrt={N_omni_sqrt}",
        f"--hd_modules={hd_modules}", 
        f"--input_std={input_std}",
        f"--angle_std={angle_std}",
        f"--inc_angle_std={inc_angle_std}",
        f"--k={k}",
        f"--gain={gain}",
        f"--v={v}",
        f"--A_spat={A_spat}", 
        f"--A_conj={A_conj}",
        f"--A_mod={A_mod}",
        f"--seed={seed}",
        f"--inclination_dir={inclination_dir}",
        f"--load_traj={load_traj}",
        f"--thresh_weights={thresh_weights}",
        f"--T0={T0}",f"--T1={T1}"
    ]
    
        
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("\n--- SPIKING.PY CRASHED ---")
        print(e.stderr)
        raise

if __name__ == "__main__":
    set_start_method("spawn")

    num = 1
    load_traj= 'True'        
    l_torus = 30
    l_asym = 6.5
    # inclination_angs = [0, jnp.pi/6, jnp.pi/3, 0]
    inclination_angs = [0, jnp.pi/3]

    L = 50

    path2save0 = os.getcwd()
    path2save0 = os.path.join(path2save0,'data', f'Simulation-spatial-num{num}')
    
    if not os.path.exists(path2save0):
        os.mkdir(path2save0)
    print(path2save0)
    
        
    for i_incl, incl_ang in tqdm(enumerate(inclination_angs)):
            
        parameters = {'d_asym': l_asym,
                      'inclination_angle': incl_ang,
                      'l': l_torus, 'L': L}
        
        path2save = os.path.join(path2save0, f'incl_ang{i_incl}')
        if not os.path.exists(path2save):
            os.mkdir(path2save)
        with open(os.path.join(path2save, 'parameters_reduced.pkl'), 'wb') as file:
            pickle.dump(parameters, file)
        
        run_job(path2save = path2save,
                l_asym = l_asym,
                l_torus = l_torus, 
                inclination_angle = incl_ang,
                seed = i_incl,
                dt = 0.003,
                gain=1, 
                T0 = 11, 
                T1 = 13,
                A_mod = 7.1)                        
                            