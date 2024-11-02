
import os
import h5py 
import time
import logging
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits, calc_temperature

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,metadata,logfile=None,verbose=False):

    """
    read_subvol: Read particle data for a subvolume from a TNG simulation snapshot.

    Input:
    -----------
    path: str
        Path to the simulation snapshot. This is the path to first snapshot chunk, not the containing folder.
    ivol: int
        Subvolume index.
    nslice: int
        Number of subvolumes in each dimension.
    metadata: object
        Metadata object containing the simulation parameters.
    logfile: str
        Path to the logfile.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the cell & normal baryonic particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the cell & normal baryonic particle data for the subvolume.

    """

    # Set up logging
    if logfile is not None:
        logging.basicConfig(filename=logfile,level=logging.INFO)
        logging.info(f"Reading subvolume {ivol} from {path}...")

    # Retrieve metadata
    boxsize=metadata.boxsize
    hval=metadata.hval

    # Get the scale factor
    snap_idx_in_metadata=np.where(metadata.snapshots_flist==path)[0][0]
    afac=metadata.snapshots_afac[snap_idx_in_metadata]

    # Get file list
    isnap_folder=path.split('snap_')[0]
    isnap_flist=sorted([isnap_folder+fname for fname in os.listdir(isnap_folder) if '.hdf5' in fname])
    numfiles=len(isnap_flist)
    logging.info(f"Reading {numfiles} files from {isnap_folder}...")

    # Get limits for the subvolume
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)

    ptype_fields={0:['InternalEnergy',
                     'ElectronAbundance',
                     'Metallicity',
                     'Density',
                     'StarFormationRate'],
                  1:[],
                  4:['GFM_Metallicity']}

    # Initialize the particle data dictionary
    pdata=[{ptype:[] for ptype in ptype_fields} for ifile in range(numfiles)]

    # Loop over all the snapshot chunks
    t0=time.time()
    for ifile,ifname in enumerate(isnap_flist):
        pdata_ifile=h5py.File(ifname,'r')
        npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']
        mass_table=pdata_ifile['Header'].attrs['MassTable']

        print(f'Loading data for ifile {ifile+1}/{numfiles}')
        for iptype,ptype in enumerate(ptype_fields):
            
            # Check if the particle type exists in the file
            if npart_ifile[ptype]:
                
                # Generate a mask for the particles in the subvolume
                subvol_mask=np.ones(npart_ifile[ptype])
                coordinates=pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3/hval #convert to cMpc
                
                # Print max/min coordinates
                logging.info(f"Coordinates: {np.nanmin(coordinates,axis=0)} {np.nanmax(coordinates,axis=0)}")
                
                # Mask for each dimension and check for periodicity
                for idim,dim in enumerate('xyz'):
                    lims_idim=lims[2*idim:(2*idim+2)]
                    if lims_idim[0]<0 and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]>=boxsize+lims_idim[0]
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]-boxsize
                    if lims_idim[1]>boxsize and nslice>1:#check for periodic
                        otherside=coordinates[:,idim]<=(lims_idim[1]-boxsize)
                        coordinates[:,idim][otherside]=coordinates[:,idim][otherside]+boxsize

                    # Mask for the subvolume
                    idim_mask=np.logical_and(coordinates[:,idim]>=lims_idim[0],coordinates[:,idim]<=lims_idim[1])
                    subvol_mask=np.logical_and(subvol_mask,idim_mask)
                    npart_ifile_invol=np.nansum(subvol_mask)

                # Check if there are particles in the subvolume
                if npart_ifile_invol:
                    print(f'There are {npart_ifile_invol} ivol ptype {ptype} particles in this file')
                    subvol_mask=np.where(subvol_mask)
                    coordinates=coordinates[subvol_mask]
                    
                    # Mask and load basic properties -- ParticleIDs, Masses, Velocities are always included
                    logging.info(f"Reading {ptype} particle IDs, masses, coordinates & velocities...")
                    pdata[ifile][ptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
                    pdata[ifile][ptype]['ParticleType']=np.uint16(np.ones(npart_ifile_invol)*ptype)
                    pdata[ifile][ptype].loc[:,[f'Coordinates_{dim}' for dim in 'xyz']]=coordinates;del coordinates
                    pdata[ifile][ptype].loc[:,[f'Velocity_{dim}' for dim in 'xyz']]=pdata_ifile[f'PartType{ptype}']['Velocities'][:][subvol_mask]*np.sqrt(afac) #peculiar velocity in km/s

                    # Get masses (use the mass table value for DM particles)
                    if not ptype==1:
                        pdata[ifile][ptype]['Mass']=np.float32(pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval)
                    else:
                        pdata[ifile][ptype]['Mass']=np.float32(np.ones(npart_ifile_invol)*mass_table[ptype]*1e10/hval)      

                    # Mask and load rest of the properties
                    for field in ptype_fields[ptype]:
                        if not 'GFM' in field:
                            pdata[ifile][ptype][field]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])
                        else:
                            field_out=field[4:]
                            pdata[ifile][ptype][field_out]=np.float32(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])

                    # If gas, do temp calculation
                    if ptype==0:
                        pdata[ifile][ptype]['Temperature']=calc_temperature(pdata[ifile][ptype],XH=0.76,gamma=5/3)
                        del pdata[ifile][ptype]['InternalEnergy']
                        del pdata[ifile][ptype]['ElectronAbundance']
    
                else:
                    print(f'No ivol ptype {ptype} particles in this file!')
                    pdata[ifile][ptype]=pd.DataFrame([])
            else:
                print(f'No ptype {ptype} particles in this file!')
                pdata[ifile][ptype]=pd.DataFrame([])

            print(f'Loaded itype {ptype} for ifile {ifile+1}/{numfiles} in {time.time()-t0:.3f} sec')
        
        # Concatenate the dataframes for this file
        pdata[ifile]=pd.concat([pdata[ifile][ptype] for ptype in ptype_fields])
        pdata[ifile].reset_index(inplace=True,drop=True)

    # Concatenate the particle dataframes
    print('Concatenating all particle data...')
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    #Print max/min coordinates
    print(f"Coordinates: {np.nanmin(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']],axis=0)} {np.nanmax(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']],axis=0)}")

    # Create a spatial KDTree for the particle data
    print('Creating KDTree for particle data...')
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata, pdata_kdtree