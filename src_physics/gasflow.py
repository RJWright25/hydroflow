# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells). Both 
import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import constant_MpcpGyrtokmps


def calculate_flow_rate(masses,vrad,dR,vboundary=0):
    """
    calculate_flow_rate: Calculate the mass flow rate across a boundary for a set of particles in a shell.

    Input:
    -----------
    masses: np.array
        Array of shell masses (in Msun).
    vrad: np.array
        Array of radial velocities (in km/s).
    vboundary: float
        Radial velocity of the boundary (in km/s).
    dr: float
        Shell width (in pMpc).

    Output:
    -----------
    flow_rate: np.array
        Array of mass flow rates (in Msun/yr).

    """
    # Calculate the flow rate across the boundary
    vrad-=vboundary
    inflow_mask=vrad<0
    outflow_mask=vrad>0

    # Convert vrad to Mpc/Gyr
    vrad/=constant_MpcpGyrtokmps
    vrad=np.abs(vrad)

    # Calculate the flow rate
    inflow=np.sum(masses[inflow_mask]*vrad[inflow_mask])/dR
    outflow=np.sum(masses[outflow_mask]*vrad[outflow_mask])/dR

    return np.array([inflow,outflow])






    
