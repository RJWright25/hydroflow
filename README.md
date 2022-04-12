# hydroflow -- gas flows in cosmological simulations
#### Ruby Wright (2021)

Tools for Lagrangian analysis of gas flows in cosmological simulations. 

## Code outline

### src_sims
hydroflow/src_sims includes routines for (i) reading particle data and (ii) processing structure finder outputs from various simulations (as per each directory).

Currently supported: EAGLE snapshot outputs, EAGLE snipshots outputs.

### src_physics
hydroflow/src_physics includes low-level functions for conversions and profile-fitting (utils.py), tools to analyse galaxies (galaxy.py), and routines to analyse gas flows between outputs (gasflow.py). 

### run
hydroflow/run contains the routines to separate cosmological boxes into sub-volumes, and to execute the gas flow algorithms as a job array over these sub-volumes.  

