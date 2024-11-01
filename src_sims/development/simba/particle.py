import numpy as np
import pandas as pd
import h5py 
from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,metadata,logfile=None,verbose=False):
    """
    read_subvol: Read particle data for a subvolume from a CAMELS simulation snapshot.

    Input:
    -----------
    path: str
        Path to the simulation snapshot.
    ivol: int
        Subvolume index.
    nslice: int
        Number of subvolumes in each dimension.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the particle data for the subvolume.

    """

    pdata_file=h5py.File(path,'r')
    boxsize=metadata.boxsize
    hval=metadata.hval
    afac=metadata.afac
    pdata_file.close()
    
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['InternalEnergy',
                     'ElectronAbundance',
                     'Density',
                     'Metallicity',
                     'StarFormationRate'],
                  4:['Metallicity'],
                  5:[]}

    pdata=[{} for iptype in range(len(ptype_fields))]

    pdata_ifile=h5py.File(path,'r')
    npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

    for iptype,ptype in enumerate(ptype_fields):
        if npart_ifile[ptype]:

            #mask for subvolume
            subvol_mask=np.ones(npart_ifile[ptype])
            coordinates=np.float32(pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3/hval)
            
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
                pdata[iptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
                pdata[iptype]['ParticleType']=np.uint16(np.ones(npart_ifile_invol)*ptype)

                # print('Loading')
                pdata[iptype].loc[:,[f'Coordinates_{dim}' for dim in 'xyz']]=coordinates
                if not ptype==1:
                    pdata[iptype].loc[:,[f'Velocity_{dim}' for dim in 'xyz']]=pdata_ifile[f'PartType{ptype}']['Velocities'][:][subvol_mask]*np.sqrt(afac)#peculiar

                # print('Loading masses')
                pdata[iptype]['Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval      


                # print('Loading rest')
                for field in ptype_fields[ptype]:
                    if not field=='Metallicity':
                        pdata[iptype][field]=pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask]
                    else:
                        pdata[iptype][field]=pdata_ifile[f'PartType{ptype}'][field][:,0][subvol_mask]

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
                pdata[iptype]=pd.DataFrame([])
        else:
            print(f'No ptype {ptype} particles in this file!')
            pdata[iptype]=pd.DataFrame([])

    pdata_ifile.close()

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata, pdata_kdtree