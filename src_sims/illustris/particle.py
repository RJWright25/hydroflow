
import numpy as np
import pandas as pd
import h5py 
import os
import time

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,nchunks=1e3):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']*1e-3
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    afac=1/(1+pdata_file['Header'].attrs['Redshift'])
    pdata_file.close()
    
    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname])
    if nchunks and len(flist)>nchunks:
        flist=flist[:nchunks]

    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  4:['GFM_Metallicity','GFM_StellarFormationTime'],
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
                coordinates=np.float32(pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3)
                
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
                    pdata[ifile][ptype].loc[:,[f'Coordinates_{dim}' for dim in 'xyz']]=coordinates;del coordinates
                    if not ptype==1:
                        pdata[ifile][ptype].loc[:,[f'Velocity_{dim}' for dim in 'xyz']]=pdata_ifile[f'PartType{ptype}']['Velocities'][:][subvol_mask]*afac**(1/2)

                    # print('Loading masses')
                    if not ptype==1:
                        pdata[ifile][ptype]['Mass']=np.float32(pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval)
                    else:
                        pdata[ifile][ptype]['Mass']=np.float32(np.ones(npart_ifile_invol)*masstable[ptype]*10**10/hval)      

                    # print('Loading rest')
                    for field in ptype_fields[ptype]:
                        if not 'GFM' in field:
                            pdata[ifile][ptype][field]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])
                        else:
                            field_out=field[4:]
                            pdata[ifile][ptype][field_out]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])

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

        ########### match the tracers to the baryonic particles ###########
        ################################################################## 

        numbar_thisvol=np.nansum([pdata[ifile][ptype].shape[0] for ptype in [0,4,5]])
        numtcr=pdata_ifile[f'PartType3']['ParentID'].shape[0]
        
        if numbar_thisvol and numtcr:
            pdata[ifile][0]=pd.concat([pdata[ifile][ptype] for ptype in [0,4,5] if not pdata[ifile][ptype].shape[0]==0])
            pdata[ifile][0].sort_values(by='ParticleIDs',inplace=True)
            pdata[ifile][0].reset_index(inplace=True,drop=True)
            pdata_ifile_baryons_IDs=pdata[ifile][0].ParticleIDs

            t0=time.time()
            pdata_tcr_parent_IDs=np.uint64(pdata_ifile[f'PartType3']['ParentID'][:])
            expected_idx_of_tracer_in_pdata=np.searchsorted(pdata_ifile_baryons_IDs,pdata_tcr_parent_IDs)
            tracer_match_1=pdata_tcr_parent_IDs==np.concatenate([pdata_ifile_baryons_IDs,[np.nan]])[(expected_idx_of_tracer_in_pdata,)]
            pdata_tcr_tracer_IDs_invol=np.uint64(pdata_ifile[f'PartType3']['TracerID'][:])[tracer_match_1]
            expected_idx_of_tracer_in_pdata=expected_idx_of_tracer_in_pdata[tracer_match_1];del tracer_match_1    

            pdata[ifile][3]=pdata[ifile][0].loc[expected_idx_of_tracer_in_pdata,:].copy()# reindexing to tracer based
            pdata[ifile][3]['ParticleIDs']=pdata_tcr_tracer_IDs_invol #set particle IDs as the tracer IDs
            pdata[ifile][3].reset_index(drop=True,inplace=True)
            pdata[ifile][3]['Flag_Tracer']=np.ones(pdata[ifile][3].shape[0],dtype=np.int8) #
            pdata[ifile][3]['Mass']=np.float32(np.ones(pdata[ifile][3].shape[0])*masstable[3]*10**10/hval)  
            numtcr_thisvol=pdata[ifile][3].shape[0]

            pdata[ifile][0]['Flag_Tracer']=np.zeros(pdata[ifile][0].shape[0],dtype=np.int8)            

            print(f'Matched tracers for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')
            pdata_ifile.close()#housekeeping

        else:
            numtcr_thisvol=0
            print('No baryons in ifile for desired volume, will not match tracers')
        if numtcr_thisvol:
            try:
                pdata[ifile]=pd.concat(pdata[ifile][ptype] for ptype in [0,3] if not pdata[ifile][ptype].shape[0]==0)
            except:
                print('No particles in ifile for desired volume')
                pdata[ifile]=pd.DataFrame([])

        else:
            print('No particles in ifile for desired volume')
            pdata[ifile]=pd.DataFrame([])

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)

    pdata_tracers=pdata.loc[pdata.Flag_Tracer==1,:].copy()
    pdata_baryons=pdata.loc[pdata.Flag_Tracer==0,:].copy()

    pdata_tracers.sort_values(by="ParticleIDs",inplace=True)
    pdata_tracers.reset_index(inplace=True,drop=True)

    pdata_baryons.sort_values(by="ParticleIDs",inplace=True)
    pdata_baryons.reset_index(inplace=True,drop=True)

    print(pdata_baryons)
    print(pdata_tracers)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata_tracers.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    pdata_kdtree_cells=cKDTree(pdata_baryons.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata_tracers, pdata_kdtree, pdata_baryons, pdata_kdtree_cells