# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: Routines to analyse gas flow rates of particles in a given shell. Returns both inflow/outflow rates.
import numpy as np
import pandas as pd
import astropy.units as apy_units


def calculate_flow_rate(masses,vrad,dr,vboundary=0,vmin=[]):
    """
    calculate_flow_rate: Calculate the mass flow rate across a boundary for a set of particles in a shell.

    Input:
    -----------
    masses: np.array
        Array of shell masses (in Msun).
    vrad: np.array
        Array of radial velocities (in km/s).
    vboundary: float
        Radial velocity of the boundary (in km/s). Relevant for e.g. pseudo-evolution.
    vmin: list
        List of minimum radial velocities (in km/s) for outflow calculations. Each entry can be fixed or an array of values.
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
    outflow_masks=[vrad>0]
    if len(vmin)>0:
        for i in range(len(vmin)):
            outflow_masks.append(np.logical_and(outflow_masks[0],vrad>vmin[i]))

    # Convert vrad to Mpc/Gyr
    vrad=vrad*apy_units.km/apy_units.s
    vrad=vrad.to(apy_units.Mpc/apy_units.Gyr)
    vrad=np.abs(vrad.value)

    # Calculate the flow rate
    inflow=np.sum(masses[inflow_mask]*vrad[inflow_mask])/dr
    outflow=[np.sum(masses[outflow_masks[0]]*vrad[outflow_masks[0]])/dr]

    for i in range(1,len(outflow_masks)):
        outflow.append(np.sum(masses[outflow_masks[i]]*vrad[outflow_masks[i]])/dr)
    
    # Make list with inflow, all outflow rates
    outflow=np.array(outflow)
    flowrates=np.zeros((len(outflow)+1))
    flowrates[0]=inflow
    flowrates[1:]=outflow

    return flowrates






    
