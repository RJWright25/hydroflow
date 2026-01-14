# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/eaglesnap/particle.py: routines to read and convert particle data from EAGLE snapshot outputs – uses read_eagle package.

import numpy as np
import pandas as pd
import h5py 
import logging

import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from pyread_eagle import EagleSnapshot

from hydroflow.src_physics.utils import get_limits, partition_neutral_gas, constant_gpmsun, constant_spyr, constant_cmpkpc

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,metadata,logfile=None):
    """
    read_subvol: Read particle data for a subvolume from an EAGLE simulation snapshot. Uses the EagleSnapshot class from read_eagle.

    Input:
    -----------
    path: str
        Path to the simulation snapshot.
    ivol: int
        Subvolume index.
    nslice: int
        Number of subvolumes in each dimension.
    metadata: object
        Metadata object containing the simulation parameters.
    logfile: str
        Path to the logfile.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the particle data for the subvolume.
    
        
    """

    # Open the snapshot file
    file=h5py.File(path,'r')

    # Set up logging
    if logfile is not None:
        logging.basicConfig(filename=logfile,level=logging.INFO)
        logging.info(f"Reading subvolume {ivol} from {path}...")

    # Retrieve metadata
    boxsize=metadata.boxsize
    hval=metadata.hval

    # Get the scale factor
    snap_idx_in_metadata=np.where(metadata.snapshots_flist==path)[0][0]
    afac=metadata.snapshots_afac[snap_idx_in_metadata]
    zval=metadata.snapshots_z[snap_idx_in_metadata]

    # Get limits for the subvolume -- these are in cMpc
    lims=get_limits(ivol,nslice,boxsize)

    # Define the fields to read for each particle type (always include ParticleIDs, Masses, Coordinates, Velocity)
    ptypes={0:['Temperature',
                'Metallicity',
                'Density',
                'StarFormationRate',
                'Velocities'],
            1:['Velocities'],
            4:['Metallicity',
               'Velocities']}
    
    ptype_subset={0:1, 1:2, 4:2} 
    
    # Use the EagleSnapshot class to read the particle data
    snapshot=EagleSnapshot(path)

    # Select the region of interest -- need to convert these limits to cMpc/h
    snapshot.select_region(xmin=lims[0]*hval,xmax=lims[1]*hval,ymin=lims[2]*hval,ymax=lims[3]*hval,zmin=lims[4]*hval,zmax=lims[5]*hval) 
    pdata={}

    # Loop over particle types
    for iptype,ptype in enumerate(ptypes):

        logging.info(f"Reading {ptype} particle IDs, coordinates & velocities...")
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs')[::ptype_subset[ptype]],columns=['ParticleIDs'])
        pdata[ptype]['ParticleType']=np.ones(pdata[ptype].shape[0])*ptype
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')[::ptype_subset[ptype]]/hval #comoving position in Mpc
        pdata[ptype].loc[:,[f'Velocities_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Velocity')[::ptype_subset[ptype],:]*np.sqrt(afac) #peculiar velocity in km/s

        # Get masses (use the mass table value for DM particles)
        if ptype==1:
            pdata[ptype].loc[:,'Masses']=file['Header'].attrs['MassTable'][1]*1e10/hval*ptype_subset[ptype] #mass in Msun
        else:
            pdata[ptype]['Masses']=snapshot.read_dataset(ptype,'Mass')[::ptype_subset[ptype]]*1e10/hval*ptype_subset[ptype] #mass in Msun
        
        # Convert other properties to physical units
        logging.info(f"Reading extra baryonic properties...")
        for field in ptypes[ptype]:
            hexp=file[f'PartType{ptype}/{field}'].attrs['h-scale-exponent']
            aexp=file[f'PartType{ptype}/{field}'].attrs['aexp-scale-exponent']
            cgs=file[f'PartType{ptype}/{field}'].attrs['CGSConversionFactor']
            pdata[ptype][field]=snapshot.read_dataset(ptype,field)[::ptype_subset[ptype]]*(hval**hexp)*(afac**aexp)*cgs


    # Convert SFR to Msun/yr from g/s
    pdata[0]['StarFormationRate']=pdata[0]['StarFormationRate']*(1/constant_gpmsun)*constant_spyr


    #Convert cm/s to km/s
    for ptype in pdata:
        pdata[ptype].loc[:,[f'Velocities_{x}' for x in 'xyz']]=pdata[ptype].loc[:,[f'Velocities_{x}' for x in 'xyz']].values/1e5 #convert to km/s

    # Add missing fields to star particles
    npart_star=pdata[4].shape[0]
    for field in ptypes[0]:
        if not field in ptypes[4]:
            pdata[4][field]=np.ones(npart_star)+np.nan

    # Combine the particle data
    logging.info(f"Concatenating particle data...")
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    # Add hydrogen partitions into HI, H2, HII from Rahmati (2013) and Blitz & Rosolowsky (2006)
    logging.info(f"Adding hydrogen partitioning...")
    gas=pdata['ParticleType'].values==0
    fHI,fH2,fHII=partition_neutral_gas(pdata,redshift=zval,sfonly=True)
    logging.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    logging.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")
    pdata.loc[:,['mfrac_HI_BR06','mfrac_H2_BR06']]=np.nan
    pdata.loc[gas,'mfrac_HI_BR06']=fHI
    pdata.loc[gas,'mfrac_H2_BR06']=fH2


    # Print fraction of particles that are gas
    print(f"Fraction of gas particles: {np.sum(pdata['ParticleType'].values==0)/pdata.shape[0]:.2e}")
    logging.info(f"Fraction of gas particles: {np.sum(pdata['ParticleType'].values==0)/pdata.shape[0]:.2e}")  


    # Create a spatial KDTree for the particle data
    logging.info(f"Creating KDTree for particle data...")
    pdata_kdtree=KDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

