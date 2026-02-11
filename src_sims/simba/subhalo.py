
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time

from hydroflow.run.tools_catalogue import dump_hdf
from hydroflow.run.initialise import load_metadata

def extract_subhaloes(path,mcut=1e10,metadata=None):

    """
    extract_subhaloes: Read the subhalo catalogue from a SIMBA caesar output file. 
                       Currently only reads central galaxies.

    Input:
    -----------
    path: str or list of str.
        Path(s) to the simulation caesar catalogue(s).
    mcut: float
        Minimum mass of subhaloes to include [log10(M/Msun)].
    metadata: str
        Path to the metadata file.

    Output:
    -----------
    subcat: pd.DataFrame
        DataFrame containing the subhalo catalogue.

    """
    # Check if just one path is given
    if type(path)==str:
        path=[path]
    
    # Grab metadata from the metadata file
    if metadata is not None:
        metadata_path=metadata
        metadata=load_metadata(metadata)
    else:
        simflist=os.listdir(os.getcwd())
        for metadata_path in simflist:
            if '.pkl' in metadata_path:
                metadata_path=metadata_path
                metadata=load_metadata(metadata_path)
                print(f"Metadata file found: {metadata_path}")
                break

    # Ensure that some catalogues exist
    if len(path)==0:
        print("No catalogue paths given. Exiting...")
        return None

    # Extract snapshot numbers from the paths and metadata
    snapnums=[]
    afacs=[]
    hval=metadata.hval
    for ipath in path:
        snapnum=int(ipath.split('.hdf5')[0][-3:]);snapnums.append(snapnum)
        mask=np.where(metadata.snapshots_idx==snapnum)[0][0]
        afac=metadata.snapshots_afac[mask];afacs.append(afac)

    # Base output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'

    # Conversion factors
    mconv=1 #convert to Msun
    dconv=1e-3 #convert to cMpc

    # Initialize the subcat list
    subhalo_dfs=[]

    # Iterate over the snapshots
    for isnapnum,snapnum in enumerate(snapnums):
        print (f"Loading snapshot {snapnum}...")

        # Load the caesar file
        caesarfile=h5py.File(path[isnapnum],mode='r')
        zval=caesarfile['simulation_attributes'].attrs['redshift']
        afac=1/(1+zval)

        # Initialize the group data structure
        group_df=pd.DataFrame()

        # Load the group data
        numgroups=caesarfile['/halo_data/GroupID'].shape[0]
        group_df['SnapNum']=np.ones(numgroups)*snapnum
        group_df['Redshift']=np.ones(numgroups)*zval
        group_df['GroupNumber']=caesarfile['/halo_data/GroupID'][:]
        group_df['SubGroupNumber']=np.zeros(numgroups)
        group_df['GalaxyID']=np.int64(snapnum*1e12+group_df.loc[:,'GroupNumber'])
        group_df['Mass']=caesarfile['/halo_data/dicts/masses.total'][:]*mconv
        group_df['GroupMass']=caesarfile['/halo_data/dicts/masses.total'][:]*mconv
        group_df['Group_M_Crit200']=caesarfile['/halo_data/dicts/virial_quantities.m200c'][:]*mconv
        group_df['Group_R_Crit200']=caesarfile['/halo_data/dicts/virial_quantities.r200c'][:]*dconv 
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=caesarfile['/halo_data/minpotpos'][:]*dconv

        # Remove groups with mass below the cut and reindex
        group_df=group_df[group_df.Group_M_Crit200>=mcut]
        group_df.sort_values(['Mass'],inplace=True,ascending=False)
        group_df.reset_index(drop=True,inplace=True)

        # Append the group data to the list
        subhalo_dfs.append(group_df)

    # Concatenate the subhalo dataframes
    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]

    # Sort the subcat by snapshot number and mass
    subcat.sort_values(by=['SnapNum','Group_M_Crit200','SubGroupNumber'],ascending=[False,False,True],inplace=True)
    subcat.reset_index(inplace=True,drop=True)

    # Dump the subhalo catalogue
    dump_hdf(outpath,subcat)

    # Add path to metadata in hdf5
    if metadata is not None:
        with h5py.File(outpath, 'r+') as subcatfile:
            header= subcatfile.create_group("Header")
            header.attrs['metadata'] = metadata_path
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    return subcat