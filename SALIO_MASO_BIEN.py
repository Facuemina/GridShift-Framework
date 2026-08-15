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
            N_vis_sqrt=40, N_conj_sqrt1=15, N_conj_sqrt2=9, N_omni_sqrt=9,
            k=0.01, gain=1, v=20, A_vis=30, A_hd=1, A_vest=15, seed=0,
            tau=0.01, input_std=3, angle_std=1.2, inc_angle_std=1.2, 
            hd_modules=8, thresh=0, ACTIVATION_EXP=2,
            inclination_dir=np.pi/2, load_traj='False',
            thresh_weights=[0.3,0.3,0.3,0]):
    
    env = os.environ.copy()
    cmd = [
        sys.executable,
        "model.py",
        f"--path2save={path2save}",
        f"--l_asym={l_asym}",
        f"--dt={dt}",
        f"--inclination_angle={inclination_angle}",
        f"--L={L}",
        f"--l_torus={l_torus}",
        f"--tau={tau}", 
        f"--steps={steps}",
        f"--N_vis_sqrt={N_vis_sqrt}", 
        f"--N_conj_sqrt1={N_conj_sqrt1}",
        f"--N_conj_sqrt2={N_conj_sqrt2}", 
        f"--N_omni_sqrt={N_omni_sqrt}",
        f"--hd_modules={hd_modules}", 
        f"--input_std={input_std}",
        f"--angle_std={angle_std}",
        f"--inc_angle_std={inc_angle_std}",
        f"--v={v}",
        f"--A_vis={A_vis}", 
        f"--A_hd={A_hd}",
        f"--A_vest={A_vest}",
        f"--seed={seed}",
        f"--ACTIVATION_EXP={ACTIVATION_EXP}",
        f"--inclination_dir={inclination_dir}",
        f"--load_traj={load_traj}"
    ]
    
    # 2. Add thresh gracefully (handles if it's passed as a list OR an integer)
    if isinstance(thresh, (list, tuple)) and len(thresh) >= 3:
        cmd.extend(["--thresh", str(thresh[0]), str(thresh[1]), str(thresh[2])])
    else:
        cmd.extend(["--thresh", str(thresh), str(thresh), str(thresh)])
        
    # 3. Add gain gracefully (handles if it's passed as a list OR an integer)
    if isinstance(gain, (list, tuple)) and len(gain) >= 3:
        cmd.extend(["--gain", str(gain[0]), str(gain[1]), str(gain[2])])
    else:
        cmd.extend(["--gain", str(gain), str(gain), str(gain)])
        
    # 4. Add inhibition strength
    if isinstance(k, (list, tuple)) and len(gain) >= 3:
        cmd.extend(["--k", str(k[0]), str(k[1]), str(k[2])])
    else:
        cmd.extend(["--k", str(k), str(k), str(k)])
    
    # 5. Add thresh_weights
    cmd.extend([
        "--thresh_weights", 
        str(thresh_weights[0]), 
        str(thresh_weights[1]), 
        str(thresh_weights[2]), 
        str(thresh_weights[3])
    ])    
    
    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("\n--- SPIKING.PY CRASHED ---")
        print(e.stderr)
        raise

if __name__ == "__main__":
    set_start_method("spawn")

    num = 2
    for num in [22]:#[1,2,3,4,5]:
        load_traj = 'False'
        periodicities = [30]
        asymmetries = [7.5]
        # inclination_angs = [0, jnp.pi/6, jnp.pi/3, 0]
        inclination_angs = [0, jnp.pi/3]
        L = 50
        dt = 0.01#3#5 * 2
        STEPS = int(50000 * 0.01 / dt)
        path2save0 = os.path.split(os.getcwd())[0]    
    
        path2save0 = os.path.join(path2save0, f'Simulation-2layer-num{num}')
        
        if not os.path.exists(path2save0):
            os.mkdir(path2save0)
        print(path2save0)
        
        for l_torus in periodicities:
            for l_asym in asymmetries:
                for i_incl, incl_ang in tqdm(enumerate(inclination_angs)):
            
                    parameters = {'d_asym': l_asym,
                                  'inclination_angle': i_incl,
                                  'l': l_torus, 'L': L}
                    
                    path2save = os.path.join(path2save0, f'incl_ang{i_incl}')
                    if not os.path.exists(path2save):
                        os.mkdir(path2save)
                    with open(os.path.join(path2save, 'parameters_reduced.pkl'), 'wb') as file:
                        pickle.dump(parameters, file)
                                                                    
                    pars = {'path2save': path2save, 'l_asym': l_asym, 'dt': dt, 
                            'inclination_angle': 0, 'L': L, 'l_torus': l_torus, 
                            'tau': 0.01, 'steps': STEPS, 'N_vis_sqrt': 25, 
                            'N_conj_sqrt1': 15, 'N_conj_sqrt2': 9, 'N_omni_sqrt': 9, 
                            'hd_modules': 8, 'input_std': 6, 'angle_std': 1.5, 
                            'inc_angle_std': 1.2, 'k': 0.001,
                            'gain': 1.2, 
                            'v': 6, 'A_vis': 30, 'A_hd': 1, 
                            'A_vest': 1, 'seed': 0, 'thresh': [0, 0, 150], 
                            'ACTIVATION_EXP': 2, 'inclination_dir': np.pi*3/2, 
                            'load_traj': 'False', 'thresh_weights': [0.3, 0.3, 0.3, .3]}
                    
                    conj_std = 0.3
                    
# =============================================================================
#                     pars['A_vest'] = .7 #1.2
#                     pars['inclination_angle'] = incl_ang
#                     pars['seed'] = i_incl + 10 * num
#                     pars['l_torus'] = l_torus
#                     pars['thresh'] = [0,
#                                       0, 
#                                       300 * (1 + 1 * pars['A_vest'] * np.sin(incl_ang))]#250]
#                                       # 200 * (1 + 1 * pars['A_vest'] * np.sin(incl_ang))]#300 * (np.cos(incl_ang) + 1)]
#                     
#                     pars['thresh_weights'] = [conj_std, conj_std,
#                                               0.4, 0.8]
#                     pars['hd_modules'] = 8
#                     pars['N_conj_sqrt1'] = 15
#                     pars['N_conj_sqrt2'] = 9
#                     # pars['N_omni_sqrt'] = 9
#                     # pars['gain'] = [10,.1,.1]
#                     # pars['k'] = [1e-3, 1e-6, 1e-8]
#                     
#                     pars['gain'] = [10,.1,.05]
#                     pars['k'] = [1e-3, 1e-6, 1e-4]
#                     # pars['gain'] = [1,1,1]
#                     # pars['k'] = [1e-4, 1e-4, 1e-5]
# =============================================================================
                    
                    pars['A_vest'] = .8 #.5
                    pars['inclination_angle'] = incl_ang
                    pars['seed'] = i_incl + 20 * num
                    pars['l_torus'] = l_torus
                    pars['thresh'] = [0,
                                      0 * pars['A_vest'] * np.sin(incl_ang), 
                                      150 * (1 + 1 * pars['A_vest'] * np.sin(incl_ang))]#300 * (np.cos(incl_ang) + 1)]
                                        # 200 * (1 + 1 * pars['A_vest'] * np.sin)
                    pars['thresh_weights'] = [conj_std, conj_std,
                                              0.5, 0.5]
                    pars['hd_modules'] = 8
                    pars['N_conj_sqrt1'] = 15
                    pars['N_conj_sqrt2'] = 9
                    
                    pars['gain'] = [10,.1,10]
                    pars['k'] = [1e-3, 1e-6, 1e-1]
                    
                    run_job(**pars)