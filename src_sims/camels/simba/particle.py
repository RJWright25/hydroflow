# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/camels/particle.py: routines to read and convert particle data from camels (SUBFIND) snapshot outputs.

import numpy as np
import pandas as pd
import h5py 

from scipy.spatial import cKDTree

from hydroflow.src_physics.utils import get_limits

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,ptypes=None):
    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    if not ptypes:
        ptypes={0:['Masses','Density','InternalEnergy','ElectronAbundance','Metallicity','StarFormationRate'],
                1:['Masses'],
                4:['Masses','Metallicity']}
    
    pdata={}
    for iptype,ptype in enumerate(ptypes):
        print(f'Loading data for ptype {ptype}')

        #mask for subvolume
        npart_itype=pdata_file['Header'].attrs['NumPart_Total'][ptype]
        subvol_mask=np.ones(npart_itype)
    
        for idim,dim in enumerate('xyz'):
            print(f'Masking subvolume for dim {dim}')

            lims_idim=lims[2*idim:(2*idim+2)]
            pdata_itype_idim=pdata_file[f'PartType{ptype}']['Coordinates'][:,idim]
            idim_mask=np.logical_and(pdata_itype_idim>=lims_idim[0],pdata_itype_idim<=lims_idim[1])
            subvol_mask=np.logical_and(subvol_mask,idim_mask)

        print('Loading IDs')
        pdata[ptype]=pd.DataFrame(data=pdata_file[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
        pdata[ptype].loc[:,'ParticleType']=ptype

        for idim,dim in enumerate('xyz'):
            pdata[ptype].loc[:,f'Coordinates_{dim}']=pdata_file[f'PartType{ptype}']['Coordinates'][:,idim][subvol_mask]*1e-3

        for field in ptypes[ptype]:
            print(f'Loading {field}')

            if not field=='Metallicity':
                pdata[ptype][field]=pdata_file[f'PartType{ptype}'][field][:][subvol_mask]
            else:
                pdata[ptype][field]=pdata_file[f'PartType{ptype}'][field][:,0][subvol_mask]
        
        pdata[ptype]['Mass']=pdata[ptype]['Masses']
        del pdata[ptype]['Masses']
    
    pdata_file.close()


    # for star & DM particles assign a nan temp, density
    npart_dm=pdata[1].shape[0]
    npart_star=pdata[4].shape[0]
    for field in ptypes[0]:
        if not field in ptypes[4]:
            pdata[4][field]=np.ones(npart_star)*np.nan
        if not field in ptypes[1]:
            pdata[1][field]=np.ones(npart_dm)*np.nan

    

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)
    
    # #conversions
    pdata=convert_pdata(path,pdata)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree

##### PARTICLE CONVERSIONS
def convert_pdata(path,pdata):
    #coordinates to comoving h^-1 Mpc
    pdata_file=h5py.File(path,'r')
    hfac=pdata_file['Header'].attrs['HubbleParam']
    pdata_file.close()

    conversions={'Mass':1/hfac*10**10}

    for field,conversion in conversions.items():
        pdata[field]=pdata[field].values*conversion

    #temperature
    ne     = pdata['ElectronAbundance'].values
    energy = pdata['InternalEnergy'].values
    yhelium = 0.0789
    T = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
    T *= (1.67262178e-24/ 1.38065e-16  )
    pdata['Temperature']=T
    del pdata['InternalEnergy']
    del pdata['ElectronAbundance']

        
    return pdata



