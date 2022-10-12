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
def read_subvol(path,ivol,nslice,ptypes=None,idm=False):
    file=h5py.File(path,'r')
    boxsize=file['Header'].attrs['BoxSize']
    hfac=file['Header'].attrs['HubbleParam']
    afac=file['Header'].attrs['ExpansionFactor']

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    if not ptypes:
        ptypes={0:['Temperature','Metallicity','Density','Entropy'],
                4:['Metallicity']}

    if idm:
        ptypes[1]=[]

    snapshot=EagleSnapshot(path)
    snapshot.select_region(*lims)
    pdata={}

    for iptype,ptype in enumerate(ptypes):
        pdata[ptype]=pd.DataFrame(data=snapshot.read_dataset(ptype,'ParticleIDs'),columns=['ParticleIDs'])
        pdata[ptype].loc[:,[f'Coordinates_{x}' for x in 'xyz']]=snapshot.read_dataset(ptype,'Coordinates')
        if not ptype==1:
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

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)


    #conversions & SFRs
    pdata=convert_pdata(path,pdata)
    pdata=convert_sfr(pdata)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

##### PARTICLE CONVERSIONS
def convert_pdata(path,pdata):
    # density in nH/cm^3; mass in Msun; SFR in msun/yr (grams per second)
    snapshot=h5py.File(path,'r')
    msun=snapshot[f'Constants'].attrs['SOLAR_MASS']
    mproton=snapshot[f'Constants'].attrs['PROTONMASS']
    snapshot.close()

    molecular_weight=1.2285
    conversions={'Mass':1/msun,
                 'Density':1/(mproton*molecular_weight)}
                 
    for field,conversion in conversions.items():
        pdata[field]=pdata[field].values*conversion
        
    return pdata

##### ADD SFRS
def convert_sfr(pdata):
    gamma=5/3;n=1.4
    fac=1.7184561445175171e-15
    keos=17235.4775202551
    
    gas=pdata['ParticleType'].values==0
    pdata.loc[gas,'StarFormationRate']=0
    
    densitythresh=pdata['Density'].values>0.1*((pdata['Metallicity'].values)/0.002)**(-0.64)
    tempthresh=pdata['Temperature'].values/(keos*pdata['Density'].values**(1/3))<=10**0.5
    sfthresh=np.logical_and.reduce([densitythresh,tempthresh,gas])

    mass=pdata.loc[sfthresh,'Mass'].values
    entropy=pdata.loc[sfthresh,'Entropy'].values
    density=pdata.loc[sfthresh,'Density'].values

    pdata.loc[sfthresh,'StarFormationRate']=fac*mass*(entropy*density**gamma)**((n-1)/2)

    return pdata

