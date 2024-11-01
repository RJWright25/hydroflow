
import numpy as np
import pandas as pd
import h5py 
import os
import time

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits


##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,nchunks=None):
    """
    read_subvol: Read particle data for a subvolume from an Illustris simulation snapshot.

    Input:
    -----------
    path: str
        Path to the simulation snapshot.
    ivol: int
        Subvolume index.
    nslice: int
        Number of subvolumes in each dimension.
    nchunks: int
        Optional cap on the number of files to load.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the cell & normal baryonic particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the cell & normal baryonic particle data for the subvolume.

    """
    
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
    ptype_fields={0:['InternalEnergy',
                     'ElectronAbundance',
                     'Density',
                     ''
                     'GFM_Metallicity',
                     'StarFormationRate'],
                  4:['GFM_Metallicity',
                     'GFM_StellarFormationTime'],
                  5:[]}

    pdata=[{ptype:[] for ptype in ptype_fields} for ifile in range(numfiles)]

    for ifile,ifname in enumerate(flist):
        try:
            pdata_ifile=h5py.File(ifname,'r')
        except:
            print(f'Error reading file {ifname} -- skipping')
            continue
        npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            t0=time.time()

            if npart_ifile[ptype]:

                #mask for subvolume
                subvol_mask=np.ones(npart_ifile[ptype])
                coordinates=pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3/hval
                
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
                    pdata[ifile][ptype].loc[:,[f'Velocity_{dim}' for dim in 'xyz']]=pdata_ifile[f'PartType{ptype}']['Velocities'][:][subvol_mask]*np.sqrt(afac)#peculiar

                    # print('Loading masses')
                    if not ptype==1:
                        pdata[ifile][ptype]['Mass']=np.float32(pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval)
                    else:
                        pdata[ifile][ptype]['Mass']=np.float32(np.ones(npart_ifile_invol)*masstable[ptype]*1e10/hval)      

                    # print('Loading rest')
                    for field in ptype_fields[ptype]:
                        if not 'GFM' in field:
                            pdata[ifile][ptype][field]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])
                        else:
                            field_out=field[4:]
                            pdata[ifile][ptype][field_out]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])

                    #if gas, do temp clc
                    if ptype==0:
                        ne     = pdata[iptype].ElectronAbundance; del pdata[iptype]['ElectronAbundance']
                        energy =  pdata[iptype].InternalEnergy;del pdata[iptype]['InternalEnergy']
                        energy*=(1e10/hval/(1.67262178e-24))#convert to grams from 1e10Msun/h
                        energy*=3.086e21*3.086e21/(3.1536e16*3.1536e16) #convert to cm^2/s^2
                        mu=4.0/(1.0 + 3.0*0.76 + 4.0*0.76*ne)*1.67262178e-24
                        temp = energy*(5/3-1)*mu/1.38065e-16
                        pdata[iptype]['Temperature']=np.float32(temp)
            
                else:
                    print(f'No ivol ptype {ptype} particles in this file!')
                    pdata[ifile][ptype]=pd.DataFrame([])
            else:
                print(f'No ptype {ptype} particles in this file!')
                pdata[ifile][ptype]=pd.DataFrame([])

            print(f'Loaded itype {ptype} for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata, pdata_kdtree