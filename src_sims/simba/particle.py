import numpy as np
import pandas as pd
import h5py 
from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice):

    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']*1e-3
    hval=pdata_file['Header'].attrs['HubbleParam']
    pdata_file.close()
    
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)
    ptype_fields={0:['InternalEnergy','ElectronAbundance','Metallicity','StarFormationRate'],
                  1:[],
                  4:['Metallicity'],
                  5:[]}
    
    pdata=[{} for iptype in range(len(ptype_fields))]

    pdata_ifile=h5py.File(path,'r')
    npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

    for iptype,ptype in enumerate(ptype_fields):
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
                pdata[iptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
                pdata[iptype]['ParticleType']=np.uint16(np.ones(npart_ifile_invol)*ptype)

                # print('Loading')
                for idim,dim in enumerate('xyz'):
                    pdata[iptype][f'Coordinates_{dim}']=coordinates[:,idim]

                # print('Loading masses')
                pdata[iptype]['Mass']=pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval      


                # print('Loading rest')
                for field in ptype_fields[ptype]:
                    pdata[iptype][field]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])


                #if gas, do temp clc
                if ptype==0:
                    ne     = pdata[iptype].ElectronAbundance; del pdata[iptype]['ElectronAbundance']
                    energy =  pdata[iptype].InternalEnergy; del pdata[iptype]['InternalEnergy']
                    yhelium = 0.0789
                    temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
                    temp *= (1.67262178e-24/ 1.38065e-16  )
                    pdata[iptype]['Temperature']=np.float32(temp)
    
            else:
                print(f'No ivol ptype {ptype} particles in this file!')
                pdata[iptype]=pd.DataFrame([])
        else:
            print(f'No ptype {ptype} particles in this file!')
            pdata[iptype]=pd.DataFrame([])

    print('Successfully loaded')

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #generate KDtree
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata, pdata_kdtree
