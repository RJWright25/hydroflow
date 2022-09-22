
import numpy as np
import pandas as pd
import h5py 
import os
import time

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,nchunks=None):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']*1e-3
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    pdata_file.close()
    
    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname])
    if nchunks:
        flist=flist[:nchunks]

    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:[],
                  4:['GFM_Metallicity'],
                  5:[]}
    
    pdata=[{ptype:[] for ptype in ptype_fields} for ifile in range(numfiles)]

    for ifile,ifname in enumerate(flist):
        pdata_ifile=h5py.File(ifname,'r')
        npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            t0=time.time()

            if npart_ifile[ptype]:

                #mask for subvolume
                subvol_mask=np.ones(npart_ifile[ptype])
                coordinates=np.float16(pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3)
                
                for idim,dim in enumerate('xyz'):
                    lims_idim=lims[2*idim:(2*idim+2)]
                    if lims_idim[0]<0 and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]>=boxsize+lims_idim[0]
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]-boxsize
                    if lims_idim[1]>boxsize and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]<=(lims_idim[1]-boxsize)
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]+boxsize

                    idim_mask=np.logical_and(coordinates[:,idim]>=lims_idim[0],coordinates[:,idim]<=lims_idim[1])
                    subvol_mask=np.logical_and(subvol_mask,idim_mask)
                    npart_ifile_invol=np.nansum(subvol_mask)

                if npart_ifile_invol:
                    print(f'There are {npart_ifile_invol} ivol ptype {ptype} particles in this file')
                    subvol_mask=np.where(subvol_mask)
                    coordinates=coordinates[subvol_mask]
                    
                    # print('Loading IDs,ptypes')
                    pdata[ifile][ptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
                    pdata[ifile][ptype]['ParticleType']=np.uint16(np.ones(npart_ifile_invol)*ptype)

                    # print('Loading')
                    for idim,dim in enumerate('xyz'):
                        pdata[ifile][ptype][f'Coordinates_{dim}']=coordinates[:,idim]

                    # print('Loading masses')
                    if not ptype==1:
                        pdata[ifile][ptype]['Mass']=np.float32(pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*10**10/hval)
                    else:
                        pdata[ifile][ptype]['Mass']=np.float32(np.ones(npart_ifile_invol)*masstable[ptype]*10**10/hval)      

                    # print('Loading rest')
                    for field in ptype_fields[ptype]:
                        if not field=='GFM_Metallicity':
                            pdata[ifile][ptype][field]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])
                        else:
                            pdata[ifile][ptype]['Metallicity']=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])

                    #if gas, do temp clc
                    if ptype==0:
                        ne     = pdata[ifile][ptype].ElectronAbundance; del pdata[ifile][ptype]['ElectronAbundance']
                        energy =  pdata[ifile][ptype].InternalEnergy; del pdata[ifile][ptype]['InternalEnergy']
                        yhelium = 0.0789
                        temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
                        temp *= (1.67262178e-24/ 1.38065e-16  )
                        pdata[ifile][ptype]['Temperature']=np.float32(temp)
        
                else:
                    print(f'No ivol ptype {ptype} particles in this file!')
                    pdata[ifile][ptype]=pd.DataFrame([])
            else:
                print(f'No ptype {ptype} particles in this file!')
                pdata[ifile][ptype]=pd.DataFrame([])

            print(f'Loaded itype {ptype} for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')


        ################# tracers #################
        numbar_thisvol=np.nansum([pdata[ifile][ptype].shape[0] for ptype in [0,4,5]])
        numdm_thisvol=pdata[ifile][1].shape[0]
        numtcr=pdata_ifile[f'PartType3']['ParentID'].shape[0]
        
        if numbar_thisvol and numtcr:
            pdata[ifile][0]=pd.concat([pdata[ifile][ptype] for ptype in [0,4,5] if not pdata[ifile][ptype].shape[0]==0])
            pdata[ifile][0].sort_values(by='ParticleIDs',inplace=True)
            pdata[ifile][0].reset_index(inplace=True,drop=True)
            pdata_ifile_baryons_IDs=pdata[ifile][0].ParticleIDs

            t0=time.time()
            pdata_tcr_parent_IDs=np.uint64(pdata_ifile[f'PartType3']['ParentID'][:])
            expected_idx_of_tracer_in_pdata=np.int32(np.searchsorted(pdata_ifile_baryons_IDs,pdata_tcr_parent_IDs))
            tracer_match_1=np.int8(pdata_tcr_parent_IDs==np.concatenate([pdata_ifile_baryons_IDs,[np.nan]])[(expected_idx_of_tracer_in_pdata,)])
            pdata_tcr_tracer_IDs_invol=np.uint64(pdata_ifile[f'PartType3']['TracerID'][:])[tracer_match_1]
            expected_idx_of_tracer_in_pdata=expected_idx_of_tracer_in_pdata[tracer_match_1];del tracer_match_1    

            pdata[ifile][0]=pdata[ifile][0].loc[expected_idx_of_tracer_in_pdata,:]# reindexing to tracer based
            pdata[ifile][0]['ParticleIDs']=pdata_tcr_tracer_IDs_invol #set particle IDs as the tracer IDs
            pdata[ifile][0].reset_index(drop=True,inplace=True)

            print(f'Matched tracers for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')
            pdata_ifile.close()#housekeeping

        else:
            print('No baryons in ifile for desired volume, will not match tracers')

        if numbar_thisvol or numdm_thisvol:
            pdata[ifile]=pd.concat(pdata[ifile][ptype] for ptype in [0,1] if not pdata[ifile][ptype].shape[0]==0)

        else:
            print('No particles in ifile for desired volume')
            pdata[ifile]=pd.DataFrame([])

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']])
    
    return pdata, pdata_kdtree