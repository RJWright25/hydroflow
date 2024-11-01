# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/eaglesnap/particle.py: routines to read and convert particle data from EAGLE (SUBFIND) snapshot outputs – uses read_eagle.

import numpy as np
import pandas as pd
import h5py 

from scipy.spatial import cKDTree
from read_eagle import EagleSnapshot

from hydroflow.src_physics.utils import get_limits

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

    file=h5py.File(path,'r')
    
    boxsize=metadata.boxsize
    hval=metadata.hval
    afac=metadata.afac

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    if not ptypes:
        ptypes={0:['Temperature',
                   'Metallicity',
                   'Density',
                   'StarFormationRate'],
                4:['Metallicity']}
    
    snapshot=EagleSnapshot(path)
    snapshot.select_region(*lims)
    pdata={}

    for iptype,ptype in enumerate(ptypes):
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs'),columns=['ParticleIDs'])
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')/hval
        pdata[ptype].loc[:,[f'Velocity_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Velocity')*np.sqrt(afac)
        pdata[ptype].loc[:,'ParticleType']=ptype
        
        if ptype==1:
            pdata[ptype].loc[:,'Masses']=file['Header'].attrs['MassTable'][1]*10**10/hval
        else:
            pdata[ptype]['Masses']=snapshot.read_dataset(ptype,'Mass')*10**10/hval
        
        #convert other properties to physical units
        for field in ptypes[ptype]:
                hexp=file[f'PartType{ptype}/{field}'].attrs['h-scale-exponent']
                aexp=file[f'PartType{ptype}/{field}'].attrs['aexp-scale-exponent']
                cgs=file[f'PartType{ptype}/{field}'].attrs['CGSConversionFactor']
                pdata[ptype][field]=snapshot.read_dataset(ptype,field)*(hval**hexp)*(afac**aexp)*cgs

    snapshot.close()

    npart_star=pdata[4].shape[0]
    for field in ptypes[0]:
        if not field in ptypes[4]:
            pdata[4][field]=np.ones(npart_star)+np.nan

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

