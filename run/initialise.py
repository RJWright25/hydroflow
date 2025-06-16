# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/initialise.py: initialise simulation object with associated attributes -- snapshot lists, redshifts, and cosmological parameters.

import os
import time
import h5py
import pickle
import numpy as np
import pandas as pd
import astropy.cosmology as apy_cosmo
from datetime import datetime


# Define the Simulation metadata class
class simulation_metadata:

    """
    A class to represent a collection of snapshots.

    Parameters:
    -----------
    snapshot_files: list
        The list of paths to the snapshot files.

    snapshot_idxs: list
        The list of snapshot indices. Assumed to be in the same order as the snapshot_file_list. If not provided, the snapshot indices are inferred from the order of the snapshot files.

    simtype: str
        The type of simulation. Supported: 'eagle', 'tng', 'simba', 'colibre', 'swift-bosca'.

    

    Attributes:
    -----------
    snapshot_files: list
        The list of paths to the snapshot files.

    snapshot_idxs: list
        The list of snapshot indices.

    snapshot_zs: list
        The list of redshifts of the snapshots.
    
    snapshot_times: list
        The list of times of the snapshots. (in Gyr)
    
    hval: float 
        The Hubble constant in km/s/Mpc/100.
    
    omegam: float
        The matter density parameter at z=0.
    
    omegab: float
        The baryon density parameter at z=0.

    omegal: float
        The dark energy density parameter at z=0.
    
    cosmology: astropy.cosmology
        The cosmology object.


    """

    # Initialize the simulation object, take a list of snapshot files and create a list of snapshot objects
    def __init__(self, snapshots_flist, snapshots_idx=None, simtype=None, name=None):
    
        self.snapshots_flist = np.array(snapshots_flist)
        if snapshots_idx is None:
            snapshots_idx = list(range(len(snapshots_flist)))
        self.snapshots_idx = np.array(snapshots_idx)
        self.simtype = simtype
        self.name = name

        # Read the redshifts and times of the snapshots
        self.snapshots_z = np.zeros(len(snapshots_flist))
        self.snapshots_afac= np.zeros(len(snapshots_flist))
        self.snapshots_tlb = np.zeros(len(snapshots_flist))
        self.snapshots_tsim = np.zeros(len(snapshots_flist))

        # Read the cosmological parameters
        cosmology=read_cosmology(snapshots_flist[-1], simtype)
        self.cosmology = cosmology
        self.hval = cosmology.H0.value/100
        self.omegam = cosmology.Om0
        self.omegab = cosmology.Ob0
        self.omegade = cosmology.Ode0

        # Read the redshifts and times of the snapshots
        for iidx,(idx, snapshot_file) in enumerate(zip(self.snapshots_idx,self.snapshots_flist)):
            with h5py.File(snapshot_file, 'r') as f:
                self.snapshots_z[iidx] = f['Header'].attrs['Redshift']
                self.snapshots_afac[iidx] = self.cosmology.scale_factor(self.snapshots_z[iidx])
                self.snapshots_tlb[iidx] = self.cosmology.lookback_time(self.snapshots_z[iidx]).value #lookback time in Gyr
                self.snapshots_tsim[iidx] = self.cosmology.age(self.snapshots_z[iidx]).value #universal time in Gyr

        #order the snapshots by redshift (ascending)
        argsort = list(np.argsort(self.snapshots_z))
        self.snapshots_flist = self.snapshots_flist[argsort]
        self.snapshots_idx = self.snapshots_idx[argsort]
        self.snapshots_z = self.snapshots_z[argsort]
        self.snapshots_afac = self.snapshots_afac[argsort]
        self.snapshots_tlb = self.snapshots_tlb[argsort]
        self.snapshots_tsim = self.snapshots_tsim[argsort]

        # Read boxsize (in cMpc) from the last snapshot
        with h5py.File(snapshots_flist[-1], 'r') as f:
            if simtype == 'colibre' or simtype=='swift-bosca':
                self.boxsize = f['Header'].attrs['BoxSize'][0]
            elif simtype == 'eagle':
                self.boxsize = f['Header'].attrs['BoxSize']/self.hval
            elif simtype == 'tng':
                self.boxsize = f['Header'].attrs['BoxSize']/1e3/self.hval 
            elif simtype == 'simba':
                self.boxsize = f['Header'].attrs['BoxSize']/1e3/self.hval

        # Create directories for outputs
        if not os.path.exists('catalogues'):
            os.makedirs('catalogues')
        if not os.path.exists('jobs'):
            os.makedirs('jobs')

        #save the simulation metadata to a pickle file
        if name is None:
            name = 'simulation'
        self.name = name
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(self, f)

        print(f"Simulation metadata saved to {name}.")



# Read the cosmological parameters from the header of a snapshot file
def read_cosmology(fname,simtype):
    """
    read_cosmology: Read the redshift of a snapshot from its filename.
    
    Input:
    
    fname: str
        Filename of the snapshot.

    simtype: str
        Type of simulation. Supported: 'eagle', 'tng', 'simba', 'colibre', 'swift-bosca'.
    
    Output:
    -----------
    cosmology: astropy.cosmology
        The cosmology object.

    """
    h5file=h5py.File(fname,'r')

    if not simtype in ['eagle','tng','simba','colibre','swift-bosca']:
        raise ValueError('Invalid simulation type. Supported: eagle, tng, simba, colibre, swift-bosca.')

    if simtype == 'eagle' or simtype == 'tng' or simtype == 'simba':
        hval=h5file['Header'].attrs['HubbleParam']
        omegam=h5file['Header'].attrs['Omega0']
        omegal=h5file['Header'].attrs['OmegaLambda']
        Tcmb0=2.73
        if simtype == 'simba':
            omegab=0.048
        else:
            omegab=h5file['Header'].attrs['OmegaBaryon']

    elif simtype == 'colibre' or simtype == 'swift-bosca':
        cosmo=h5file['Cosmology']
        hval=cosmo.attrs['H0 [internal units]'][0]/100
        omegam=cosmo.attrs['Omega_m'][0]
        omegab=cosmo.attrs['Omega_b'][0]
        omegal=cosmo.attrs['Omega_lambda'][0]
        Tcmb0=cosmo.attrs['T_CMB_0 [K]'][0]

    cosmology=apy_cosmo.LambdaCDM(H0=hval*100,Om0=omegam,Ob0=omegab,Ode0=omegal,Tcmb0=Tcmb0)
    return cosmology


def load_metadata(name):
    """
    load_metadata: Load a simulation metadata object from a pickle file.
    
    Input:
    -----------
    name: str
        Name of the pickle file containing the simulation metadata.
    
    Output:
    -----------
    sim_metadata: simulation_metadata
        The simulation metadata object.
    """
    with open(name, 'rb') as f:
        sim_metadata = pickle.load(f)

    try:
        cosmotest= sim_metadata.cosmology.H0
    except:
        print(f"Cosmology not found in {name}. Generating hard-coded cosmo object...")
        if sim_metadata.simtype =='simba':
            sim_metadata.cosmology = apy_cosmo.LambdaCDM(H0=68.0,Om0=0.3,Ob0=0.048,Ode0=0.7,Tcmb0=2.73)
        elif sim_metadata.simtype == 'tng':
            sim_metadata.cosmology = apy_cosmo.LambdaCDM(H0=67.74,Om0=0.3089,Ob0=0.0486,Ode0=0.6911,Tcmb0=2.73)
        elif sim_metadata.simtype == 'eagle':
            sim_metadata.cosmology = apy_cosmo.LambdaCDM(H0=67.77,Om0=0.307,Ob0=0.04825,Ode0=0.693,Tcmb0=2.73)
        elif sim_metadata.simtype == 'colibre':
            sim_metadata.cosmology = apy_cosmo.LambdaCDM(H0=68.1,Om0=0.304611,Ob0=0.0486,Ode0=0.693922,Tcmb0=2.7255)
    return sim_metadata



### Example usage
# colibre_flist=[]
# colibre_sim_object=simulation_metadata(snapshot_flist=['/fred/oz009/rwright/gasflows/simulations/COLIBRE/L0050N0752_FIDUCIAL/snapshots/colibre_0088/colibre_0088.hdf5',
#                                                       '/fred/oz009/rwright/gasflows/simulations/COLIBRE/L0050N0752_FIDUCIAL/snapshots/colibre_0123/colibre_0123.hdf5'],
#                                                       simtype='colibre',name='COLIBRE_L0050N0752_FIDUCIAL')
