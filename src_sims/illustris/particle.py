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
def read_subvol(path,ivol,nslice,nchunks=None):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    pdata_file.close()
    
    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0])])
    if nchunks:
        flist=flist[:nchunks]

    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['Masses','InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:[],
                  4:['Masses','GFM_Metallicity'],
                  5:['Masses']}
    
    pdata=[{ptype:pd.DataFrame([]) for ptype in ptype_fields} for ifile in range(numfiles)]

    for ifile,ifname in enumerate(flist):
        pdata_ifile=h5py.File(ifname,'r')
        npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            t0=time.time()
            if npart_ifile[ptype]:
                #mask for subvolume
                subvol_mask=np.ones(npart_ifile[ptype])
                coordinates=pdata_ifile[f'PartType{ptype}']['Coordinates'][:]
                for idim,dim in enumerate('xyz'):
                    # print(f'Masking subvolume for dim {dim}')
                    lims_idim=lims[2*idim:(2*idim+2)]
                    if lims_idim[0]<0 and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]>=boxsize+lims_idim[0]
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]-boxsize
                    if lims_idim[1]>boxsize and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]<=(lims_idim[1]-boxsize)
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]+boxsize
                    idim_mask=np.logical_and(coordinates[:,idim]>=lims_idim[0],coordinates[:,idim]<=lims_idim[1])
                    subvol_mask=np.logical_and(subvol_mask,idim_mask)

                coordinates=coordinates[subvol_mask]*1e-3
                npart=np.nansum(subvol_mask)

                if np.nansum(subvol_mask):
                    print(f'There are {np.nansum(subvol_mask)} ivol ptype {ptype} particles in this file')
                    
                    for idim,dim in enumerate('xyz'):
                        pdata[ifile][ptype][f'Coordinates_{dim}']=coordinates[:,idim]
            
                    subvol_mask=np.where(subvol_mask)

                    #ptypes
                    pdata[ifile][ptype][f'ParticleType']=np.ones(npart)*ptype

                    #pids
                    pdata[ifile][ptype][f'ParticleIDs']=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask]
                    
                    #masses
                    if not ptype==1:
                        pdata[ifile][ptype][f'Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*10**10/hval
                    else:
                        pdata[ifile][ptype][f'Mass']=np.ones(npart)*masstable[ptype]*1e10/hval        

                    #rest
                    for field in ptype_fields[ptype]:
                        pdata[ifile][ptype][field]=pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask]
                
                else:
                    print(f'No ivol ptype {ptype} particles in this file!')
            else:
                print(f'No ptype {ptype} particles in this file!')

            print(f'Loaded itype {ptype} for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')

        pdata[ifile][0]=pd.concat([pdata[ifile][ptype] for ptype in [0,4,5]])
        pdata[ifile][0].reset_index(inplace=True,drop=True)
        pdata[ifile][0].sort_values(by=['ParticleIDs'],inplace=True)
        pdata[ifile][0].reset_index(inplace=True,drop=True)

        del pdata[ifile][4]; del pdata[ifile][5]

        ################# tracers #################

        numbar=pdata[ifile][0].shape[0]
        numtcr=pdata_ifile[f'PartType3']['ParentID'].shape[0]
        baryon_pids=pdata[ifile][0]['ParticleIDs'].values
        baryon_pids=np.concatenate([baryon_pids,[np.nan]])
        if numbar and numtcr:
            t0=time.time()
            pdata_parid_ifile=pdata_ifile[f'PartType3']['ParentID'][:]
            expected_idx_of_tcr_in_pdata=np.searchsorted(baryon_pids,pdata_parid_ifile)
            tracer_match_1=pdata_parid_ifile==baryon_pids[(expected_idx_of_tcr_in_pdata,)]
            pdata_tcrid_ifile=pdata_ifile[f'PartType3']['TracerID'][:][tracer_match_1]
            expected_idx_of_tcr_in_pdata=expected_idx_of_tcr_in_pdata[tracer_match_1]
            del tracer_match_1

            pdata[ifile][0]=pdata[ifile][0].loc[expected_idx_of_tcr_in_pdata,:]
            pdata[ifile][0]['ParticleIDs']=pdata_tcrid_ifile
            pdata[ifile][0].reset_index(inplace=True,drop=True)

            print(f'Matched tracers for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')
        
        else:
            print('No baryons in ifile for desired volume, will not match tracers')

        numdm=pdata[ifile][1].shape[0]

        if numbar or numdm:
            pdata[ifile]=pd.concat(pdata[ifile][ptype] for ptype in [0,1] if not pdata[ifile][ptype].shape[0]==0)
            pdata[ifile].sort_values(by="ParticleIDs",inplace=True)
            pdata[ifile].reset_index(inplace=True,drop=True)
            pdata[ifile].loc[:,'ifile']=ifile
        else:
            print('No particles in ifile for desired volume')
            pdata[ifile]=pd.DataFrame([])

    pdata=pd.concat(pdata)
    pdata.reset_index(inplace=True,drop=True)
    print('Successfully loaded')

    tracermask=np.logical_not(pdata.ParticleType==1)
    print(f"Tracer breakdown: {np.nanmean(pdata.loc[tracermask,'ParticleType'].values==0)*100:.2f}% in gas cells, {np.nanmean(pdata.loc[tracermask,'ParticleType'].values==4)*100:.2f}% in stars or wind, {np.nanmean(pdata.loc[tracermask,'ParticleType'].values==5)*100:.2f}% in BH")

    #temperature
    gas_mask=pdata['ParticleType'].values==0
    ne     = pdata.loc[gas_mask,'ElectronAbundance'].values; del pdata['ElectronAbundance']
    energy = pdata.loc[gas_mask,'InternalEnergy'].values; del pdata['InternalEnergy']
    yhelium = 0.0789
    Temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
    Temp *= (1.67262178e-24/ 1.38065e-16  )
    pdata.loc[gas_mask,'Temperature']=Temp

    pdata['Metallicity']=pdata['GFM_Metallicity'].values; del pdata['GFM_Metallicity']

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values)
    
    return pdata, pdata_kdtree



