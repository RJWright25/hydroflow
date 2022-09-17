# src_sims/illustris/particle.py: routines to read and convert particle data from TNG snapshot outputs.

from turtle import tracer
import numpy as np
import pandas as pd
import h5py 

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits
import illustris_python


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,ptypes=None):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']
    hval=pdata_file['Header'].attrs['HubbleParam']
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    snapnum=int(path.split('snapdir_')[-1][:3])

    if not ptypes:
        ptypes={0:['Masses','Density','InternalEnergy','ElectronAbundance','Metallicity','StarFormationRate'],
                1:['Masses'],
                4:['Masses','Metallicity']}

    ptype_keys={0:'gas',1:'dm',4:'stars'}
 
    pdata={}
    for iptype,ptype in enumerate(ptypes):
        print(f'Loading data for ptype {ptype}')

        #mask for subvolume
        npart_itype=pdata_file['Header'].attrs['NumPart_Total'][ptype]
        subvol_mask=np.ones(npart_itype)
        
        for idim,dim in enumerate('xyz'):
            print(f'Masking subvolume for dim {dim}')
            lims_idim=lims[2*idim:(2*idim+2)]
            pdata_itype_idim=illustris_python.snapshot.loadSubset(path.split('snapdir')[0],snapnum,ptype_keys[ptype],['Coordinates'],mdi=[idim])*1e-3
            idim_mask=np.logical_and(pdata_itype_idim>=lims_idim[0],pdata_itype_idim<=lims_idim[1])
            subvol_mask=np.logical_and(subvol_mask,idim_mask)

        print('Loading IDs')
        pdata_pids=illustris_python.snapshot.loadSubset(path.split('snapdir')[0],snapnum,ptype_keys[ptype],['ParticleIDs'])[subvol_mask]
        pdata[ptype]=pd.DataFrame(data=pdata_pids,columns=['ParticleIDs'])

        for idim,dim in enumerate('xyz'):
            pdata_itype_idim=illustris_python.snapshot.loadSubset(path.split('snapdir')[0],snapnum,ptype_keys[ptype],['Coordinates'],mdi=[idim])[subvol_mask]*1e-3
            pdata[ptype].loc[:,f'Coordinates_{dim}']=pdata_itype_idim

        for field in ptypes[ptype]:
            print(f'Loading {field}')
            pdata[ptype][field]=illustris_python.snapshot.loadSubset(path.split('snapdir')[0],snapnum,ptype_keys[ptype],[field])[subvol_mask]
        
        pdata[ptype]['Mass']=pdata[ptype]['Masses']*10**10/hval;del pdata[ptype]['Masses']
        pdata[ptype]['Metallicity']=pdata[ptype]['GFM_Metallicity'];del pdata[ptype]['GFM_Metallicity']
        
        pdata[ptype].sort_values(by="ParticleIDs",inplace=True)
        pdata[ptype].reset_index(inplace=True,drop=True)

        if ptype==0:
            print('Loading tracers')
            tracer_df=pd.DataFrame(illustris_python.snapshot.loadSubset(path.split('snapdir')[0],snapnum,3,['ParentID','TracerID']))
            tracer_df.sort_values(by='ParentIDs',inplace=True);tracer_df.reset_index(inplace=True,drop=True)

            ### step 1 - mask out the tracers with parents not in region
            parentcell_expected_idx_if_present=pdata[ptype]['ParticleIDs'].searchsorted(tracer_df['ParentIDs'].values)
            parentcell_expected_ID_if_present=pdata[ptype]['ParticleIDs'].values[parentcell_expected_idx_if_present]
            present=tracer_df['ParentIDs'].values==parentcell_expected_ID_if_present

            print(np.nanmean(present))
            print(np.nansum(present))
            tracer_df=tracer_df.loc[present,:].copy();tracer_df.reset_index(inplace=True,drop=True)
            
            ### step 2 -- collect parent info for tracers
            parentcell_expected_idx=parentcell_expected_idx_if_present[present]
            for field in list(pdata[ptype].keys()):
                if not field=='ParticleIDs':
                    tracer_df[field]=pdata[ptype][field].values[(parentcell_expected_idx,)]
            tracer_df['ParticleIDs']=tracer_df['TracerIDs'].values
            tracer_df['CellIDs']=tracer_df['ParentIDs'].values

            pdata[0]=tracer_df;del tracer_df
        
        [ptype].loc[:,'ParticleType']=ptype

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

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree



