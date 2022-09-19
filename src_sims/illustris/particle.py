# src_sims/illustris/particle.py: routines to read and convert particle data from TNG snapshot outputs.

import numpy as np
import pandas as pd
import h5py 
import os

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits
import illustris_python


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,ptypes=None):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    nparttable=pdata_file['Header'].attrs['NumPart_Total']
    pdata_file.close()

    flist=[path+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname]
    numfiles=len(flist)
    print(f'Loading from {numfiles}')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    snapnum=int(path.split('snapdir_')[-1][:3])


    ptype_fields={0:['Masses','Density','InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:[],
                  4:['Masses','GFM_Metallicity']}

    pdata={ptype:[ifile for ifile in range(numfiles)] for ptype in ptype_fields}
    pdata_tracers=[ifile for ifile in range(numfiles)]

    for iptype,ptype in enumerate(ptypes):
        print(f'Loading data for ptype {ptype}')
        for ifile,ifname in enumerate(flist):
            pdata_ifile=h5py.File(ifname,'r')
            print(f'Loading data for ifile {ifile+1}/{numfiles}')

            #mask for subvolume
            npart_itype=pdata_ifile['Header'].attrs['NumPart_ThisFile'][ptype]
            subvol_mask=np.ones(npart_itype)
            
            for idim,dim in enumerate('xyz'):
                print(f'Masking subvolume for dim {dim}')
                lims_idim=lims[2*idim:(2*idim+2)]
                pdata_itype_idim=pdata_ifile[f'PartType{ptype}']['Coordinates'][:,idim]
                idim_mask=np.logical_and(pdata_itype_idim>=lims_idim[0],pdata_itype_idim<=lims_idim[1])
                subvol_mask=np.logical_and(subvol_mask,idim_mask)

            subvol_mask=np.where(subvol_mask)

            print('Loading IDs')
            pdata[ptype][ifile]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
            pdata[ptype][ifile].loc[:,'ifile']=ifile

            print('Loading coordinates')
            for idim,dim in enumerate('xyz'):
                pdata[ptype][ifile].loc[:,f'Coordinates_{dim}']=pdata_ifile[f'PartType{ptype}']['Coordinates'][:,idim][subvol_mask]*1e-3

            print('Loading masses')
            if not ptype==1:
                pdata[ptype][ifile]['Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][subvol_mask]*10**10/hval
            else:
                pdata[ptype][ifile].loc[:,'Mass']=masstable[ptype]            

            for field in ptypes[ptype]:
                print(f'Loading {field}')
                pdata[ptype][ifile][field]=pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask]


            ################# tracers if needed #################
            if ptype==0:
                print('Loading tracers')
                pdata_tracers[ifile]=pd.DataFrame(np.column_stack([pdata_ifile[f'PartType3']['ParentID'][:],pdata_ifile[f'PartType{ptype}']['TracerID'][:]]),['ParentID','TracerID'])
                pdata_tracers[ifile].sort_values(by='ParentID',inplace=True);pdata_tracers[ifile].reset_index(inplace=True,drop=True)


                pdata_ifile.close()

            # ### step 1 - mask out the tracers with parents not in region
            # parentcell_expected_idx_if_present=np.searchsorted(pdata_pids,tracer_df['ParentID'].values)
            # tester=np.concatenate([pdata_pids,[-1]])
            # parentcell_expected_ID_if_present=tester[parentcell_expected_idx_if_present]
            # present=tracer_df['ParentID'].values==parentcell_expected_ID_if_present

            # print(np.nanmean(present))
            # print(np.nansum(present))
            # tracer_df=tracer_df.loc[present,:].copy();tracer_df.reset_index(inplace=True,drop=True)

            # ### step 2 -- collect parent info for tracers
            # parentcell_expected_idx=parentcell_expected_idx_if_present[present]
            # for field in list(pdata[ptype].keys()):
            #     if not field=='ParticleIDs':
            #         tracer_df[field]=pdata[ptype][field].values[(parentcell_expected_idx,)]
            
            # tracer_df['ParticleIDs']=tracer_df['TracerID'].values
            # tracer_df['CellIDs']=tracer_df['ParentID'].values

            # pdata[ptype]=tracer_df;del tracer_df
            # pdata[ptype].loc[:,'ParticleType']=ptype


        pdata[ptype]=pd.concat(pdata[ptype])
        pdata[ptype].loc[:,'ParticleType']=ptype
        pdata[ptype].sort_values(by="ParticleIDs",inplace=True)
        pdata[ptype].reset_index(inplace=True,drop=True)

    #temperature
    ne     = pdata[0]['ElectronAbundance'].values
    energy = pdata[0]['InternalEnergy'].values
    yhelium = 0.0789
    Temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
    Temp *= (1.67262178e-24/ 1.38065e-16  )
    pdata[0]['Temperature']=Temp
    del pdata[0]['InternalEnergy']
    del pdata[0]['ElectronAbundance']


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

    #generate KDtree
    # pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_tracers



