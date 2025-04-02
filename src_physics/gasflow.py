# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells). Both 
import numpy as np
import pandas as pd
import astropy.units as apy_units


def calculate_flow_rate(masses,vrad,dr,vboundary=0):
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
    flow_rates: np.array
        Array of mass flow rates (in Msun/yr) -- [inflow,outflow].

    """
    # Calculate the flow rate across the boundary
    vrad-=vboundary
    inflow_mask=vrad<0
    outflow_mask=vrad>0

    # Convert vrad to Mpc/Gyr
    vrad=vrad*apy_units.km/apy_units.s
    vrad=vrad.to(apy_units.Mpc/apy_units.Gyr)
    vrad=np.abs(vrad.value)

    # Calculate the flow rate
    inflow=np.sum(masses[inflow_mask]*vrad[inflow_mask])/dr
    outflow=np.sum(masses[outflow_mask]*vrad[outflow_mask])/dr

    return np.array([inflow,outflow])






    
