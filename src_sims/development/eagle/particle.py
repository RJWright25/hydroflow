# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/eaglesnap/particle.py: routines to read and convert particle data from EAGLE snapshot outputs – uses read_eagle package.

import numpy as np
import pandas as pd
import h5py 

from scipy.spatial import cKDTree
from read_eagle import EagleSnapshot

from hydroflow.src_physics.utils import get_limits
from hydroflow.src_physics.utils import msun,sec_in_Gyr

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,metadata,logfile=None,verbose=False):
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

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the particle data for the subvolume.
    
        
    """

    # Open the snapshot file
    file=h5py.File(path,'r')

    # Retrieve metadata
    boxsize=metadata.boxsize
    hval=metadata.hval
    afac=metadata.afac

    # Get limits for the subvolume
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)

    # Define the fields to read for each particle type (always include ParticleIDs, Masses, Coordinates, Velocity)
    ptypes={0:['Temperature',
                'Metallicity',
                'Density',
                'StarFormationRate'],
            4:['Metallicity']}
    
    # Use the EagleSnapshot class to read the particle data
    snapshot=EagleSnapshot(path)
    snapshot.select_region(*lims)
    pdata={}

    # Loop over particle types
    for iptype,ptype in enumerate(ptypes):
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs'),columns=['ParticleIDs'])
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')/hval #comoving position in Mpc
        pdata[ptype].loc[:,[f'Velocity_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Velocity')*np.sqrt(afac) #peculiar velocity in km/s
        pdata[ptype].loc[:,'ParticleType']=ptype
        
        # Get masses (use the mass table value for DM particles)
        if ptype==1:
            pdata[ptype].loc[:,'Masses']=file['Header'].attrs['MassTable'][1]*10**10/hval #mass in Msun
        else:
            pdata[ptype]['Masses']=snapshot.read_dataset(ptype,'Mass')*10**10/hval #mass in Msun
        
        # Convert other properties to physical units
        for field in ptypes[ptype]:
                hexp=file[f'PartType{ptype}/{field}'].attrs['h-scale-exponent']
                aexp=file[f'PartType{ptype}/{field}'].attrs['aexp-scale-exponent']
                cgs=file[f'PartType{ptype}/{field}'].attrs['CGSConversionFactor']
                pdata[ptype][field]=snapshot.read_dataset(ptype,field)*(hval**hexp)*(afac**aexp)*cgs

    snapshot.close()

    # Convert SFR to Msun/yr from g/s
    pdata[0]['StarFormationRate']=pdata[0]['StarFormationRate']*(1/msun)*sec_in_Gyr

    # Add missing fields to star particles
    npart_star=pdata[4].shape[0]
    for field in ptypes[0]:
        if not field in ptypes[4]:
            pdata[4][field]=np.ones(npart_star)+np.nan

    # Combine the particle data
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    # Create a spatial KDTree for the particle data
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

