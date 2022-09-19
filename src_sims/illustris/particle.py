# src_sims/illustris/particle.py: routines to read and convert particle data from TNG snapshot outputs.

from weakref import ProxyType
import numpy as np
import pandas as pd
import h5py 
import os

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits
import illustris_python


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    pdata_file.close()

    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname])[:3]
    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    snapnum=int(path.split('snapdir_')[-1][:3])
    ptype_fields={0:['Masses','Density','InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:['Potential'],
                  4:['Masses','GFM_Metallicity'],
                  5:['Masses']}

    pdata=[{ptype:[] for ptype in ptype_fields} for ifile in range(numfiles)]
    pdata_tracers=[ifile for ifile in range(numfiles)]

    for ifile,ifname in enumerate(flist):
        pdata_ifile=h5py.File(ifname,'r')

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            print(f'Loading data for ptype {ptype}')

            #mask for subvolume
            npart_itype=pdata_ifile['Header'].attrs['NumPart_ThisFile'][ptype]
            subvol_mask=np.ones(npart_itype)
            
            for idim,dim in enumerate('xyz'):
                # print(f'Masking subvolume for dim {dim}')
                lims_idim=lims[2*idim:(2*idim+2)]
                pdata_itype_idim=pdata_ifile[f'PartType{ptype}']['Coordinates'][:,idim]
                idim_mask=np.logical_and(pdata_itype_idim>=lims_idim[0],pdata_itype_idim<=lims_idim[1])
                subvol_mask=np.logical_and(subvol_mask,idim_mask)

            subvol_mask=np.where(subvol_mask)

            # print('Loading IDs')
            pdata[ifile][ptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
            pdata[ifile][ptype].loc[:,'ifile']=ifile
            pdata[ifile][ptype].loc[:,'ParticleType']=ptype
            
            for idim,dim in enumerate('xyz'):
                pdata[ifile][ptype].loc[:,f'Coordinates_{dim}']=pdata_ifile[f'PartType{ptype}']['Coordinates'][:,idim][subvol_mask]*1e-3

            # print('Loading masses')
            if not ptype==1:
                pdata[ifile][ptype]['Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][subvol_mask]*10**10/hval
            else:
                pdata[ifile][ptype].loc[:,'Mass']=masstable[ptype]            

            for field in ptype_fields[ptype]:
                # print(f'Loading {field}')
                pdata[ifile][ptype][field]=pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask]

            ################# tracers if needed #################
            if ptype==0:
                # print('Loading tracers')
                pdata_tracers[ifile]=pd.DataFrame(np.column_stack([pdata_ifile[f'PartType3']['ParentID'][:],pdata_ifile[f'PartType3']['TracerID'][:]]),columns=['ParentID','TracerID'])
                pdata_tracers[ifile].loc[:,'ifile']=ifile

            pdata_ifile.close()

        pdata[ifile]=pd.concat(pdata[ifile])
        pdata[ifile].sort_values(by="ParticleIDs",inplace=True)
        pdata[ifile].reset_index(inplace=True,drop=True)

    print('Successfully loaded')

    pdata_tracers=pd.concat(pdata_tracers)
    pdata_tracers.sort_values(by="ParentID",inplace=True)
    pdata_tracers.reset_index(inplace=True,drop=True)

    # for star & DM particles assign a nan temp, density
    npart_dm=pdata[1].shape[0]
    npart_star=pdata[4].shape[0]
    for field in ptype_fields[0]:
        if not field in ptype_fields[4]:
            pdata[4][field]=np.ones(npart_star)*np.nan
        if not field in ptype_fields[1]:
            pdata[1][field]=np.ones(npart_dm)*np.nan

    #concat all pdata into one df
    pdata=pd.concat([pdata[ptype] for ptype in pdata],ignore_index=True,)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #temperature
    ne     = pdata['ElectronAbundance'].values
    energy = pdata['InternalEnergy'].values
    yhelium = 0.0789
    Temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
    Temp *= (1.67262178e-24/ 1.38065e-16  )
    pdata['Temperature']=Temp
    del pdata['InternalEnergy']
    del pdata['ElectronAbundance']

    pdata['Metallicity']=pdata['GFM_Metallicity'].values
    del pdata['GFM_Metallicity']

    #generate KDtree
    # pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_tracers



