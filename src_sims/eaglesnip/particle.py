# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/eaglesnip/particle.py: routines to read and convert particle data from EAGLE (SUBFIND) snipshot outputs – uses read_eagle. 

import numpy as np
import pandas as pd
import h5py 

from scipy.spatial import cKDTree
from read_eagle import EagleSnapshot

from hydroflow.src_physics.utils import get_limits

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,ptypes=None):
    file=h5py.File(path,'r')
    boxsize=file['Header'].attrs['BoxSize']
    hfac=file['Header'].attrs['HubbleParam']
    afac=file['Header'].attrs['ExpansionFactor']

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    if not ptypes:
        ptypes={0:['Temperature','Metallicity'],
                1:[],
                4:['Metallicity']}

    snapshot=EagleSnapshot(path)
    snapshot.select_region(*lims)
    pdata={}

    for iptype,ptype in enumerate(ptypes):
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs'),columns=['ParticleIDs'])
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')
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

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree





