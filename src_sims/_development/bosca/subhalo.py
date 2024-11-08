

import os
import numpy as np
import pandas as pd
import h5py

from hydroflow.run.tools_catalog import dump_hdf
from hydroflow.run.initialise import load_metadata

def extract_subhaloes(path,mcut=1e11,metadata=None):

    """
    read_subcat: Read the subhalo catalogue from an Illustris simulation snapshot. Uses the illustris_python package.
    
    Input:
    -----------
    path: str or list of str
        Path(s) to the halo catalogues.
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
    

        # Ensure that some catalogues exist
    if len(path)==0:
        print("No catalogue paths given. Exiting...")
        return None
    
    # Extract snapshot numbers from the paths and metadata
    snapnums=[]
    afacs=[]
    hval=metadata.hval
    for ipath in path:
        snapnum=int(ipath.split('snap_')[-1][:4]);snapnums.append(snapnum)
        mask=np.where(metadata.snapshots_idx==snapnum)[0][0]
        afac=metadata.snapshots_afac[mask];afacs.append(afac)


    # Base output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'

    # Initialize the subhalo data structure
    subhalo_dfs=[]

    # Iterate over the snapshots
    for isnapnum,snapnum in enumerate(snapnums):
        print (f"Loading snapshot {snapnum}...")

        # Units for loads
        mconv=1e10 #convert to Msun
        dconv=1/afacs[isnapnum] #convert to cMpc

        # Load the group data
        fname=path[isnapnum]
        h5file=h5py.File(fname,'r')
        
        # Initialize the group data structure
        subhalo_df=pd.DataFrame()
    
        # Load the subhalo data
        numgroups=h5file['ID'][:].shape[0]
        subhalo_df['SnapNum']=np.ones(numgroups)*snapnum
        subhalo_df['Redshift']=np.ones(numgroups)*1/afacs[isnapnum]-1
        subhalo_df['GalaxyID']=h5file['ID'][:]
        subhalo_df['Mass']=h5file['Mass_tot'][:]*mconv
        subhalo_df['GroupMass']=h5file['Mass_FOF'][:]*mconv
        subhalo_df['Group_M_Crit200']=h5file['Mass_200crit'][:]*mconv
        subhalo_df['Group_R_Crit200']=h5file['R_200crit'][:]*dconv #convert to cMpc
        subhalo_df['CentreOfPotential_x']=h5file['Xcmbp'][:]*dconv #convert to cMpc
        subhalo_df['CentreOfPotential_y']=h5file['Ycmbp'][:]*dconv #convert to cMpc
        subhalo_df['CentreOfPotential_z']=h5file['Zcmbp'][:]*dconv #convert to cMpc

        # Assign unique group number to haloes with hostHaloID==-1
        hostHaloID=h5file['hostHaloID'][:]
        subhalo_df['GroupNumber']=np.zeros(numgroups)
        subhalo_df.loc[hostHaloID==-1,'GroupNumber']=np.arange(np.sum(hostHaloID==-1))
        # Find the group number for the subhaloes
        for ihalo,halo in enumerate(subhalo_df['GalaxyID']):
            if hostHaloID[ihalo]!=-1:
                subhalo_df.loc[ihalo,'GroupNumber']=subhalo_df.loc[subhalo_df['GalaxyID']==hostHaloID[ihalo],'GroupNumber'].values[0]
        
        # Add the subgroupnumber
        subhalo_df['SubGroupNumber']=np.zeros(numgroups)
        for igroup,group in enumerate(subhalo_df['GroupNumber'].unique()):
            mask=subhalo_df['GroupNumber'].values==group
            subhalo_df.loc[mask,'SubGroupNumber']=np.arange(np.sum(mask))
        

        # Append the subhalo data to the list
        subhalo_dfs.append(subhalo_df)

    # Concatenate the subhalo dataframes
    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]

    # Sort the subcat by snapshot number and mass
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False],inplace=True)
    subcat.reset_index(inplace=True,drop=True)

    # Mask masses
    subcat=subcat[subcat['Mass'].values>=mcut].copy()

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



        
        

