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
def read_subvol(path,ivol,nslice,ptypes=None):
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
    ptypes: dict
        Dictionary containing the particle types to load. The keys are the particle types, and the values are lists of the fields to load.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the particle data for the subvolume.
    
        
    """


    file=h5py.File(path,'r')
    boxsize=file['Header'].attrs['BoxSize']
    hfac=file['Header'].attrs['HubbleParam']
    afac=file['Header'].attrs['ExpansionFactor']

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    if not ptypes:
        ptypes={0:['Temperature','Metallicity','StarFormationRate','MaximumTemperature'],
                4:['Metallicity']}
    
    snapshot=EagleSnapshot(path)
    snapshot.select_region(*lims)
    pdata={}

    for iptype,ptype in enumerate(ptypes):
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs'),columns=['ParticleIDs'])
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')
        pdata[ptype].loc[:,[f'Velocity_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Velocity')
        pdata[ptype].loc[:,'ParticleType']=ptype
        
        if ptype==1:
            pdata[ptype].loc[:,'Mass']=file['Header'].attrs['MassTable'][1]*10**10/hfac
        else:
            pdata[ptype]['Mass']=snapshot.read_dataset(ptype,'Mass')*10**10/hfac

        for field in ptypes[ptype]:
                hexp=file[f'PartType{ptype}/{field}'].attrs['h-scale-exponent']
                aexp=file[f'PartType{ptype}/{field}'].attrs['aexp-scale-exponent']
                cgs=file[f'PartType{ptype}/{field}'].attrs['CGSConversionFactor']
                pdata[ptype][field]=snapshot.read_dataset(ptype,field)*(hfac**hexp)*(afac**aexp)*cgs

    snapshot.close()

    #for star particles assign a crazy temp, density
    npart_star=pdata[4].shape[0]

    for field in ptypes[0]:
        if not field in ptypes[4]:
            pdata[4][field]=np.ones(npart_star)+np.nan

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #conversions & SFRs
    pdata=convert_pdata(path,pdata)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

##### PARTICLE CONVERSIONS
def convert_pdata(path,pdata):
    """
    convert_pdata: Convert particle data from EAGLE simulation snapshots to physical units.

    Input:
    -----------
    path: str
        Path to the simulation snapshot.
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    
    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume, with the particle properties converted to physical units.

    """
    
    # density in nH/cm^3; mass in Msun; SFR in msun/yr (grams per second)
    snapshot=h5py.File(path,'r')
    msun=snapshot[f'Constants'].attrs['SOLAR_MASS']
    secperyear=snapshot[f'Constants'].attrs['SEC_PER_YEAR']
    snapshot.close()
    conversions={'Mass':1,
                 'StarFormationRate':secperyear/msun}
    for field,conversion in conversions.items():
        pdata[field]=pdata[field].values*conversion
        
    return pdata



