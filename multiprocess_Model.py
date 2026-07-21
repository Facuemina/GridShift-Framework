#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Facundo Emina
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
            L = 50, dt = 0.01, steps = int(5 * 1e4),
            N_vis_sqrt = 40, N_conj_sqrt = 12, N_omni_sqrt = 10,
            k = 0.01, m = 0.5, gain = 1, v = 20, 
            A_vis = 30, A_hd = 1, A_vest = 15, seed = 0,
            tau = 0.01, tauv = 0.1, input_std = 3,
            angle_std = 1.2, hd_modules = 8, nfr = 30, 
            thresh=0, ACTIVATION_EXP=2, firingrate = False
            ):
    # Call the original script with extra args
    env = os.environ.copy()
    cmd = [
        sys.executable,  # python interpreter
        "Model.py",
        f"--path2save={path2save}",
        f"--firingrate={firingrate}",
        f"--l_asym={l_asym}",
        f"--dt={dt}",
        f"--inclination_angle={inclination_angle}",
        f"--L={L}",
        f"--l_torus={l_torus}",
        f"--tau={tau}", 
        f"--tauv={tauv}",
        f"--steps={steps}",
        f"--N_vis_sqrt={N_vis_sqrt}", 
        f"--N_conj_sqrt={N_conj_sqrt}", 
        f"--N_omni_sqrt={N_omni_sqrt}",
        f"--hd_modules={hd_modules}", 
        f"--input_std={input_std}",
        f"--angle_std={angle_std}",
        f"--k={k}",
        f"--m={m}",
        f"--gain={gain}", 
        f"--v={v}",
        f"--A_vis={A_vis}", 
        f"--A_hd={A_hd}",
        f"--A_vest={A_vest}",
        f"--seed={seed}",
        f"--nfr={nfr}",
        f"--ACTIVATION_EXP={ACTIVATION_EXP}"
    ]
    
    # Dynamically append thresh so argparse (nargs=2) reads it correctly
    if isinstance(thresh, (list, tuple)) and len(thresh) >= 2:
        cmd.extend(["--thresh", str(thresh[0]), str(thresh[1])])
    else:
        # If a single number is passed (like the default thresh=0), 
        # duplicate it to satisfy the nargs=2 requirement
        cmd.extend(["--thresh", str(thresh), str(thresh)])
        
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("\n--- RATEMODEL.PY CRASHED ---")
        print(e.stderr) # This prints the actual traceback from the child script
        raise

if __name__ == "__main__":
    set_start_method("spawn")

    # parameter grid
    num = 2
    firingrate = 'False'
    periodicities = [25]#,30,35,40]
    asymmetries = [4]
    inclination_angs = [0,jnp.pi/6,jnp.pi/3,0]
    L = 60

    path2save0 = os.path.split(os.getcwd())[0]    
    if firingrate == 'True':
        folder ='FiringRate'
    else:
        folder = 'Spiking'
    path2save0 = os.path.join(path2save0,f'Simulation{folder}-num{num}')
    
    if not os.path.exists(path2save0):
        os.mkdir(path2save0)
        
    for l_torus in periodicities:
        for d_asym in asymmetries:
            for i_incl, incl_ang in tqdm(enumerate(inclination_angs)):
        
                parameters = {'d_asym': d_asym,
                              'inclination_angle': i_incl,
                              'l': l_torus, 'L': L}
                
                path2save = os.path.join(path2save0,f'incl_ang{i_incl}')
                if not os.path.exists(path2save):
                    os.mkdir(path2save)
                with open(os.path.join(path2save,'parameters_reduced.pkl'),'wb') as file:
                    pickle.dump(parameters,file)
                
                run_job(path2save, d_asym, l_torus, incl_ang, 
                        seed = i_incl*5 + 80,
                        input_std =4, k=.01, gain = 1, nfr = 30, 
                        thresh=[0,300], hd_modules=8,
                        firingrate=firingrate, ACTIVATION_EXP=2)