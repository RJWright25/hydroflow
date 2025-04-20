import logging
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits, calc_temperature, partition_neutral_gas, constant_gpmsun, constant_cmpkpc

##### READ PARTICLE DATA
def read_subvol(path,ivol,nslice,metadata,logfile=None,verbose=False):
    """
    read_subvol: Read particle data for a subvolume from a SIMBA simulation snapshot.

    Input:
    -----------
    path: str
        Path to the simulation snapshot. 
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
    zval=metadata.snapshots_z[snap_idx_in_metadata]
    
    # Get limits for the subvolume
    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)

    # Particle type fields -- will always read ParticleIDs, ParticleType, Coordinates, Velocities, Masses
    ptype_fields={0:['InternalEnergy',
                     'ElectronAbundance',
                     'Density',
                     'Metallicity',
                     'StarFormationRate'],
                  1:[],
                  4:['Metallicity'],
                  5:[]}
    
    # Initialize particle data
    pdata=[None for iptype in range(len(ptype_fields))]

    # Open the snapshot file
    pdata_ifile=h5py.File(path,'r')
    npart_ifile=pdata_ifile['Header'].attrs['NumPart_ThisFile']

    # Loop over particle types
    for iptype,ptype in enumerate(ptype_fields):
        logging.info(f"Reading ptype {ptype}...")
        subset=ptype_subset[ptype]
        
        if npart_ifile[ptype]:
            # Generate a mask for the particles in the subvolume
            subvol_mask=np.ones(npart_ifile[ptype])
            coordinates=np.float32(pdata_ifile[f'PartType{ptype}']['Coordinates'][:]*1e-3/hval)

            # logging.info(f"min/max coordinates: {np.min(coordinates)}/{np.max(coordinates)}")

            # Check for periodicity
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

            # Check if there are particles in the subvolume
            if npart_ifile_invol:
                logging.info(f'There are {npart_ifile_invol} ivol ptype {ptype} particles in this file')
                subvol_mask=np.where(subvol_mask)

                # Save basic particle data 
                logging.info(f"Reading IDs, coordinates, velocities and masses for ptype {ptype}...")
                pdata[iptype]=pd.DataFrame(data=pdata_ifile[f'PartType{ptype}']['ParticleIDs'][:][subvol_mask],columns=['ParticleIDs'])
                pdata[iptype]['ParticleType']=np.uint16(np.ones(npart_ifile_invol)*ptype)
                pdata[iptype].loc[:,[f'Coordinates_{dim}' for dim in 'xyz']]=coordinates[subvol_mask];del coordinates
                pdata[iptype].loc[:,[f'Velocities_{dim}' for dim in 'xyz']]=pdata_ifile[f'PartType{ptype}']['Velocities'][:][subvol_mask]*np.sqrt(afac)#peculiar velocity in km/s
                pdata[iptype]['Masses']=pdata_ifile[f'PartType{ptype}']['Masses'][:][subvol_mask]*1e10/hval #mass in Msun

                # Load extra baryonic properties
                for field in ptype_fields[ptype]:
                    if not field=='Metallicity':
                        pdata[iptype][field]=np.float128(pdata_ifile[f'PartType{ptype}'][field][:][subvol_mask])
                    else:
                        pdata[iptype][field]=pdata_ifile[f'PartType{ptype}'][field][:,0][subvol_mask]

                # Convert density to g/cm^3
                if ptype==0:
                    # Raw data are in 1e10/h (ckpc/h)^-3
                    pdata[ptype]['Density']=pdata[ptype]['Density'].values*1e10*hval**2/afac**3 #Msun/pkpc^3
                    pdata[ptype]['Density']=pdata[ptype]['Density'].values*np.float128(constant_gpmsun)/np.float128(constant_cmpkpc)**3 #g/cm^3

                # If gas, do temp calculation
                if ptype==0:
                    logging.info(f"Calculating temperature for {ptype} particles...")
                    pdata[iptype]['Temperature']=calc_temperature(pdata[ptype],XH=0.76,gamma=5/3)
                    del pdata[iptype]['InternalEnergy']
                    del pdata[iptype]['ElectronAbundance']    
            else:
                logging.info(f'No ivol ptype {ptype} particles in this file!')
                pdata[iptype]=pd.DataFrame([])
        else:
            logging.info(f'No ptype {ptype} particles in this file!')
            pdata[iptype]=pd.DataFrame([])

    pdata_ifile.close()

    # Combine the particle data
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)      

    # Add hydrogen partitions into HI, H2, HII from Rahmati (2013) and Blitz & Rosolowsky (2006)
    logging.info(f"Adding hydrogen partitioning...")
    gas=pdata['ParticleType'].values==0
    fHI,fH2,fHII=partition_neutral_gas(pdata,redshift=zval,sfonly=True)
    logging.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    logging.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")
    pdata.loc[:,['mfrac_HI','mfrac_H2','mfrac']]=np.nan
    pdata.loc[gas,'mfrac_HI']=fHI
    pdata.loc[gas,'mfrac_H2']=fH2
    
    # Create a spatial KDTree for the particle data
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}'for x in 'xyz']].values)
    
    return pdata, pdata_kdtree