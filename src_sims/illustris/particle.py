# src_sims/illustris/particle.py: routines to read and convert particle data from TNG snapshot outputs.

import numpy as np
import pandas as pd
import h5py 
import os
import time

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

    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname])
    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['Masses','Density','InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:[],
                  4:['Masses','GFM_Metallicity'],
                  5:['Masses']}
    
    allkeys=ptype_fields[0]

    pdata=[{ptype:[] for ptype in ptype_fields} for ifile in range(numfiles)]

    for ifile,ifname in enumerate(flist):
        pdata_ifile=h5py.File(ifname,'r')
        npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            t0=time.time()

            #mask for subvolume
            subvol_mask=np.ones(npart_ifile[ptype])
            coordinates=pdata_ifile[f'PartType{ptype}']['Coordinates'][:]
            
            for idim,dim in enumerate('xyz'):
                # print(f'Masking subvolume for dim {dim}')
                lims_idim=lims[2*idim:(2*idim+2)]
                idim_mask=np.logical_and(coordinates[:,idim]>=lims_idim[0],coordinates[:,idim]<=lims_idim[1])
                subvol_mask=np.logical_and(subvol_mask,idim_mask)

            if np.nansum(subvol_mask):
                print(f'There are {np.nansum(subvol_mask)} ivol ptype {ptype} particles in this file')
                subvol_mask=np.where(subvol_mask)

                # print('Loading IDs')
                pdata[ifile][ptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])

                # print('Loading IDs')
                pdata[ifile][ptype].loc[:,[f'Coordinates_{dim}' for dim in 'xyz']]=coordinates[subvol_mask]*1e-3

                # print('Loading masses')
                if not ptype==1:
                    pdata[ifile][ptype]['Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*10**10/hval
                else:
                    pdata[ifile][ptype].loc[:,'Mass']=masstable[ptype]*10**10/hval        

                for field in ptype_fields[ptype]:
                    # print(f'Loading {field}')
                    pdata[ifile][ptype][field]=pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask]

                pdata[ifile][ptype].loc[:,'ParticleType']=ptype
            

            else:
                print(f'No ivol ptype {ptype} particles in this file!')
                pdata[ifile][ptype]=pd.DataFrame([])

            print(f'Loaded itype {ptype} for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')


        ################# tracers #################
        numgas=pdata[ifile][0].shape[0]
        numbar=np.nansum([pdata[ifile][ptype].shape[0] for ptype in [0,4,5]])

        if numgas and numbar:
            t0=time.time()
            pdata_tracers_ifile=pd.DataFrame(np.column_stack([pdata_ifile[f'PartType3']['ParentID'][:],pdata_ifile[f'PartType3']['TracerID'][:]]),columns=['ParentID','TracerID'])
            pdata_tracers_ifile.sort_values(by='ParentID',inplace=True)
            pdata_tracers_ifile.reset_index(inplace=True,drop=True)
            pdata_ifile.close()#housekeeping

            #baryons in the volume for this ifile
            pdata_ifile_baryons=pd.concat(pdata[ifile][ptype] for ptype in [0,4,5] if not pdata[ifile][ptype].shape[0]==0)

            pdata_ifile_baryons.sort_values(by='ParticleIDs',inplace=True)
            pdata_ifile_baryons.reset_index(inplace=True,drop=True)
            pdata_ifile_baryons_IDs=pdata_ifile_baryons['ParticleIDs'].values

            #all tracers in this file
            pdata_tracer_IDs=pdata_tracers_ifile['TracerID'].values
            pdata_tracer_parentIDs=pdata_tracers_ifile['ParentID'].values

            expected_idx_of_tracer_in_pdata=np.searchsorted(pdata_ifile_baryons_IDs,pdata_tracer_parentIDs)
            tracer_match_1=pdata_tracer_parentIDs==np.concatenate([pdata_ifile_baryons_IDs,[np.nan]])[(expected_idx_of_tracer_in_pdata,)]
            pdata_tracer_IDs_invol=pdata_tracer_IDs[tracer_match_1]
            expected_idx_of_tracer_in_pdata=expected_idx_of_tracer_in_pdata[tracer_match_1]

            parent_data=pdata_ifile_baryons.loc[expected_idx_of_tracer_in_pdata,:]
            parent_data['ParentID']=parent_data['ParticleIDs'].values
            parent_data['ParticleIDs']=pdata_tracer_IDs_invol #set particle IDs as the tracer IDs
            parent_data.loc[:,'TracerType']=parent_data['ParticleType'] #record the tracer type
            parent_data.loc[:,'ParticleType']=0 # set tracers as gas
            parent_data.reset_index(drop=True,inplace=True)
            #save the matched tracers as the gas data

            pdata[ifile][0]=parent_data
            print(f'Matched tracers for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec ({np.nanmean(tracer_match_1)*100:.2f}% of the tracers in this file were in the desired ivol {ivol+1}/{nslice**3})')
        else:
            print('No baryons in ifile for desired volume, will not match tracers')

        numdm=pdata[ifile][1].shape[0]

        if numbar and numdm:
            pdata[ifile]=pd.concat(pdata[ifile][ptype] for ptype in [0,1,4] if not pdata[ifile][ptype].shape[0]==0)
            pdata[ifile].sort_values(by="ParticleIDs",inplace=True)
            pdata[ifile].reset_index(inplace=True,drop=True)
            pdata[ifile].loc[:,'ifile']=ifile
        else:
            print('No particles in ifile for desired volume')
            pdata[ifile]=pd.DataFrame([])

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    tracermask=pdata.ParticleType==0
    print(f"Tracer breakdown: {np.nanmean(pdata.loc[tracermask,'TracerType'].values==0)*100:.2f}% in gas cells, {np.nanmean(pdata.loc[tracermask,'TracerType'].values==4)*100:.2f}% in stars or wind, {np.nanmean(pdata.loc[tracermask,'TracerType'].values==5)*100:.2f}% in BH")

    #temperature
    gas_mask=pdata['ParticleType'].values==0
    ne     = pdata.loc[gas_mask,'ElectronAbundance'].values
    energy = pdata.loc[gas_mask,'InternalEnergy'].values
    yhelium = 0.0789
    Temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
    Temp *= (1.67262178e-24/ 1.38065e-16  )
    pdata.loc[gas_mask,'Temperature']=Temp
    del pdata['InternalEnergy']
    del pdata['ElectronAbundance']

    pdata['Metallicity']=pdata['GFM_Metallicity'].values
    del pdata['GFM_Metallicity']

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize)

    return pdata, pdata_kdtree



