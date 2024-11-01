# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
### *Ruby Wright (2024)*

Tools for Eulerian analysis of gas flows in hydrodynamical simulations. This repository contains code to generalise the analysis of gas flows to and from large samples of haloes and galaxies. 

## Eulerian calculations
This code includes routines to calculate (instantaneous) Eulerian gas flow rates at a given boundary. This does not require that any particle or element represents the same "parcel" of matter from snapshot to snapshot. Eulerian-based mass flow rates can be calculated at a given boundary by categorising relevant boundary particles/elements as being either outflow or inflow depending on the sign of their radial velocity -- where the radial velocity of a particle/element $i$ relative to a halo center $j$ can be calculated as $`v_{{\rm\,r},\,ij}=\vec{{v}_{ij}}\cdot\vec{{r}_{ij}}/\lvert\vec{{r}_{ij}}\rvert`$. Then, inflow or outflow rate at shell $r=R$ around halo $j$ for each of these subsets of boundary gas elements, $i\in k$, can be calculated follows:

$`\dot{M}_{k}(r=R)=\frac{1}{\Delta\,r}\times\Sigma_{i\in k}\left(m_{i}\frac{{\vec{v}_{ij}}\cdot\vec{{r}_{ij}}}{\lvert\vec{{r}_{ij}}\rvert}\right)`$

## Code outline

### src_sims
hydroflow/src_sims includes routines for (i) reading particle data and (ii) processing structure finder outputs from various simulations (as per each directory).

Currently supported: 
* COLIBRE snapshots

### src_physics
hydroflow/src_physics includes low-level functions for conversions (utils.py), and a function to perform the analysis of galaxies and surrounding gas flows (galaxy.py)

### run
hydroflow/run contains the routines to separate cosmological boxes into sub-volumes and run the analysis -- the key script is execute.py. There are also some routines to submit job arrays per different HPC system architectures. The execute.py script will run the analysis for galaxies in a given subvolume of a cosmological box (defined by nslice and ivol). Splitting the analysis this way makes the problem "embarrassingly parallel", though the best way to parellelise the operation for each subvolume (e.g. via job arrays or at a node level) will depend on the HPC scheduler and architecture. 


