# Grid-Cell Continuous Attractor Network on an Inclined Torus

This repository contains the simulation and analysis code for a continuous attractor network (CAN) model of grid cells, in which spatial and head-direction (HD) inputs drive an "omnidirectional" grid-cell population living on a twisted-torus manifold. The model includes a body/surface **inclination angle** that modulates the gain of spatially-tuned inputs as a function of the animal's heading, allowing the effect of locomotion on tilted/inclined surfaces on the grid pattern to be studied.

The `data/` folder contains the simulation outputs used to produce the figures and statistics reported in the associated paper.

## Repository structure

```
.
├── simulation.py                # Single-run simulation (command-line script)
├── multiprocess_simulation.py   # Orchestrates multiple simulation.py runs (one per inclination angle)
├── utils.py                     # Trajectory generation, connectivity, rate maps, cross-correlation utilities
├── analysis.py                  # Loads simulation output, computes rate maps/shifts, generates figures + stats
└── data/                        # Simulation results shown in the paper (see "Data" below)
```

## Model overview

- **Trajectory**: A 2D random-walk trajectory (correlated random walk with reflecting boundaries, `generate2D_pos` in `utils.py`) or a trajectory loaded from an external `.mat` file (`load_traj=True`) can be used to drive the network.
- **Populations**:
  - *Spatial* cells (`N_spat_sqrt² × hd_modules`), tuned to visual/allocentric position and to one of `hd_modules` preferred head directions.
  - *Conjunctive* cells (`N_conj_sqrt² × hd_modules`), with grid phases uniformly tiling a toroidal manifold and joint position × head-direction tuning.
  - *Omnidirectional* output cells (`N_omni_sqrt²`), which receive convergent input from the spatial and conjunctive populations through connectivity built on a twisted torus (`build_torus_connectivity`, `map2torus_fn`, `distance_torus_sq`).
- **Inclination modulation**: an `inclination_angle` parameter multiplicatively modulates the spatial input gain (`A_mod`, `inc_angle_std`) for the subset of head directions aligned with `inclination_dir`, and additively shifts global inhibition (`T0`, `T1`) — modeling how a change in the plane/slope of locomotion (e.g., pitch/tilt of the head or environment) reshapes grid-cell firing.
- **Dynamics**: The omnidirectional population is a leaky-integrator attractor network (`tau`, `k`, `gain`) integrated with `lax.scan`; spikes are drawn from a Poisson process on the resulting firing rates and stored as sparse `(time_idx, neuron_idx, counts)` triplets.

### The inclination-induced shift effect

Each omnidirectional cell pools input from spatial cells across all `hd_modules` head-direction preferences. Inclination breaks the symmetry of that pooling: spatial cells whose preferred heading points *uphill* are modulated differently from those pointing *downhill* (via `A_mod`, `inc_angle_std`, and `inclination_dir` in `simulation.py`), so the two sub-populations no longer contribute equally to the summed drive.

- **Uphill movement**: the "downhill-preferring" spatial input is boosted, pulling the peak of the summed input (and hence the rectified, normalized firing field) forward along the trajectory.
- **Downhill movement**: the "uphill-preferring" spatial input is boosted instead, pulling the peak backward relative to baseline.

Because the boosted sub-population differs between uphill and downhill running, the net effect is not a simple rescaling but a **direction-dependent spatial shift** of each cell's firing field relative to its flat-terrain (baseline) position — visible at the population level as a rotation/displacement of the averaged 2D HD × spatial contribution (e.g., the 60° vs. 0° inclination comparison), and at the single-cell level as a horizontal offset between the modulated and baseline input/firing-rate curves.

This is exactly what the `data` pipeline in `utils.py`/`analysis.py` measures: `compute_rate_maps_from_sparse` → `compute_cross_corrs` → `find_spatial_shift_subpixel` computes, for every cell, the sub-pixel offset between its baseline (0° inclination) rate map and its rate map at a given inclination angle, and `analysis.py` aggregates these offsets into the `Shift X` / `Shift Y` distributions and the corresponding Friedman/Wilcoxon/Dunn statistics, split further by running direction (`Up` vs. `Down`) to test whether the shift itself is direction-dependent.

## Requirements

- Python 3.9+
- [JAX](https://github.com/google/jax) (`jax`, `jax.numpy`) — CPU or GPU build
- `numpy`, `scipy`
- `pandas`, `seaborn`, `matplotlib`
- `scikit-posthocs`
- `tqdm`
- `scipy.io` is used to load `.mat` trajectory files when `load_traj=True`

Install with:

```bash
pip install jax jaxlib numpy scipy pandas seaborn matplotlib scikit-posthocs tqdm
```

(Use the JAX install instructions for your platform if you want GPU acceleration.)

## Usage

### 1. Run a single simulation

`simulation.py` is a command-line script driven by `argparse`. Example:

```bash
python simulation.py \
    --path2save ./Simulation-spatial-num0/incl_ang0 \
    --l_asym 6 \
    --l_torus 30 \
    --inclination_angle 0 \
    --seed 7 \
    --load_traj False
```

Key arguments (see `simulation.py` for the full list and defaults):

| Argument | Description |
|---|---|
| `--path2save` | Output directory (must exist) |
| `--path2load_traj` | Folder containing `trajectory_60.mat`, used when `--load_traj True` |
| `--l_asym` | Asymmetric (feed-forward) shift length between conjunctive/spatial inputs and the omnidirectional layer |
| `--l_torus` | Grid spacing / torus period |
| `--inclination_angle` | Inclination of the surface/heading modulating spatial gain |
| `--inclination_dir` | Preferred head direction affected by inclination |
| `--L` | Size of the (square) environment |
| `--N_spat_sqrt`, `--N_conj_sqrt`, `--N_omni_sqrt` | Grid dimensions (per side) of each population |
| `--hd_modules` | Number of discrete head-direction tuning modules |
| `--dt`, `--tau`, `--steps` | Simulation time step, network time constant, number of steps |
| `--k`, `--gain` | Attractor network nonlinearity / normalization parameters |
| `--seed` | RNG seed (trajectory + Poisson spiking) |
| `--T0`, `--T1` | Baseline and inclination-dependent inhibition |
| `--load_traj` | `"True"`/`"False"` — use an experimental trajectory instead of a simulated random walk |
| `--thresh_weights` | Threshold applied to normalize connectivity weights |

**Outputs** written to `path2save`:
- `sparse_spikes.pkl` — dict with `time_idx`, `neuron_idx`, `counts` (sparse spike counts)
- `traj.npy` — the `[x, y, head_direction]` trajectory used (after discarding the first `I0` burn-in steps)
- `parameters_complete.pkl` — dict of all parameters used for the run

### 2. Run a batch of simulations across inclination angles

`multiprocess_simulation.py` launches `simulation.py` as a subprocess once per inclination angle (`inclination_angs`), creating one `incl_angN` subfolder per run inside `Simulation-spatial-num{num}/`:

```bash
python multiprocess_simulation.py
```

Edit the parameters at the bottom of the file (`num`, `l_torus`, `l_asym`, `inclination_angs`, `L`, seeds, etc.) to configure a batch. Each run's reduced parameter set is also saved as `parameters_reduced.pkl` inside its subfolder before `simulation.py` is launched.

### 3. Analyze simulation output and reproduce figures

`analysis.py` expects the folder structure produced above (`Simulation-spatial-num{num}/incl_ang{i}/...`) and:

1. Loads spikes/trajectories and computes spatial rate maps for every session (`load_and_compute_maps_from_sparse`).
2. Computes cross-correlograms between the baseline session (`incl_ang0`) and every other session, and extracts the sub-pixel spatial shift of the grid pattern (`compute_cross_corrs`, `find_spatial_shift_subpixel`).
3. Produces figures (`Figures/` subfolder, saved as `.svg`):
   - `CrossCorrelograms_zoom.svg` — mean cross-correlograms per session
   - `SpatialMaps.svg` — trajectory overlays, rate maps, and single-cell autocorrelograms
   - `Shifts_Y.svg` — boxplots of the X/Y grid-pattern shift across sessions
   - `Mean_fr.svg` — mean firing rate per session
   - `UP_DOWN-Correlation.svg` — running-direction (up vs. down) shift comparison and population spatial-correlation analysis
4. Runs statistics: Friedman / Wilcoxon / Mann–Whitney / Kruskal–Wallis tests and Dunn's post-hoc test (Bonferroni-corrected) on the grid-shift and firing-rate distributions, and builds `df_shifts`, `df_firing_rates`, `df_friedman`, and `df_dunn_omni_y` dataframes summarizing the results.

Run it (as a script, or cell-by-cell — it is written with `#%%` Spyder/Jupyter-style cell markers) from inside the directory that is one level below `Simulation-spatial-num{num}/`:

```bash
python analysis.py
```

Adjust `num`, `NEURON_IDX`, and the smoothing/plotting parameters at the top of the `CONFIGURATION` section to select which simulation batch and example neuron to inspect.

## Data

The `data/` folder contains the pre-computed simulation results (spikes, trajectories, and parameters for each inclination-angle session) used to generate the figures and statistics reported in the paper. Point `path2load` in `analysis.py` (or the `sim_folder` argument of `load_and_compute_maps_from_sparse`) at the relevant `Simulation-spatial-num{num}` subfolder inside `data/` to reproduce the published results without re-running the simulations.

## Citation

If you use this code, please cite the associated paper. *(Add citation / DOI here.)*

## License

*(Add license here.)*
