
import os
import numpy as np
import pandas as pd
import h5py

from hydroflow.run.tools_catalog import dump_hdf
from hydroflow.run.initialise import load_metadata

import illustris_python as tng_tools

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
        snapnum=int(ipath.split('groups_')[-1][:3]);snapnums.append(snapnum)
        mask=np.where(metadata.snapshots_idx==snapnum)[0][0]
        afac=metadata.snapshots_afac[mask];afacs.append(afac)

    # Units for loads
    mconv=1e10/hval #convert to Msun
    dconv=1e-3/hval #convert to cMpc

    # Base output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'

    # Input base path
    basepath=path[0].split('/groups')[0]

    # Initialize the subhalo data structure
    subhalo_dfs=[]

    # Iterate over the snapshots
    for isnapnum,snapnum in enumerate(snapnums):
        print (f"Loading snapshot {snapnum}...")
        subfind_raw=tng_tools.groupcat.load(basepath,snapNum=snapnum)
        groupcat=subfind_raw['halos']
        subcat=subfind_raw['subhalos']

        # Get the redshift
        afac=afacs[isnapnum]
        zval=1/afac-1

        # Initialize the group and subhalo dataframes
        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        # Extract group data
        print('Extracting group data...')
        numgroups=groupcat['GroupMass'][:].shape[0]
        group_df['SnapNum']=np.ones(numgroups)*snapnum
        group_df['Redshift']=np.ones(numgroups)*zval
        group_df['GroupNumber']=np.float64(list(range(group_df.shape[0])))
        group_df['SubGroupNumber']=np.zeros(numgroups)
        group_df['GroupMass']=groupcat['GroupMass'][:]*mconv
        group_df['Group_M_Crit200']=groupcat['Group_M_Crit200'][:]*mconv
        group_df['Group_R_Crit200']=groupcat['Group_R_Crit200'][:]*dconv #convert to cMpc
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=groupcat['GroupPos'][:]*dconv #convert to cMpc
        group_df.sort_values(by='GroupNumber',inplace=True,ascending=True)
        group_df=group_df.loc[group_df['GroupMass'].values>=mcut,:].copy()
        group_df.reset_index(drop=True,inplace=True)

        # Extract subhalo data
        print('Extracting subhalo data...')
        numsubhaloes=subcat['SubhaloMass'][:].shape[0]
        subhalo_df['SnapNum']=np.ones(numsubhaloes)*snapnum
        subhalo_df['Redshift']=np.ones(numsubhaloes)*zval
        subhalo_df['GroupNumber']=np.float64(subcat['SubhaloGrNr'][:])
        subhalo_df['GalaxyID']=np.array(range(subhalo_df.shape[0]))
        subhalo_df['StarFormationRate']=subcat['SubhaloSFR'][:] #Msun/yr
        subhalo_df['StellarMass']=subcat['SubhaloMassType'][:,4]*mconv
        subhalo_df['Mass']=subcat['SubhaloMass'][:]*mconv 
        subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat['SubhaloPos'][:]*dconv
        subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat['SubhaloVel'][:,:]*np.sqrt(afac) #peculiar velocity in km/s

        # Initialize group data in subhalo data
        keys_groups=['GroupMass','Group_M_Crit200','Group_R_Crit200','Group_CentreOfPotential_x','Group_CentreOfPotential_y','Group_CentreOfPotential_z']
        for key in keys_groups:
            subhalo_df[key]=np.zeros(subhalo_df.shape[0])+np.nan
        
        # Sort subhalo data
        subhalo_df.sort_values(by=['GroupNumber','Mass'],inplace=True,ascending=[True,False])
        subhalo_df=subhalo_df.loc[subhalo_df['Mass'].values>=mcut,:] #apply mass cut
        subhalo_df.reset_index(inplace=True,drop=True)

        # Match group data to subhalo data
        print('Matching group data to subhalo data...')
        unique_groups=subhalo_df['GroupNumber'].unique()
        for igroup,group in enumerate(unique_groups):
            if igroup%1000==0:
                print(f'Group {igroup+1}/{unique_groups.shape[0]}...')
                print(f"Searching for {group} in {group_df['GroupNumber'].values} ...")
            
            # Using the searchsorted method to find the group index
            group_idx=np.searchsorted(group_df['GroupNumber'].values,group)
            if group!=group_df['GroupNumber'].values[group_idx]:
                print(f"Group {group} does not match {group_df['GroupNumber'].values[group_idx]}...")
                continue
            
            # Find the subhalo indices range for the group
            subhalo_idx_1=np.searchsorted(subhalo_df['GroupNumber'].values,group)
            subhalo_idx_2=np.searchsorted(subhalo_df['GroupNumber'].values,subhalo_df['GroupNumber'].values[igroup+1])
            if subhalo_idx_2-subhalo_idx_1==0:
                print(f'No subhalos in group {group}...')
                continue
            
            # Assign group data to subhalo data
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'SubGroupNumber']=np.array(range(np.sum(subhalo_idx_2-subhalo_idx_1)))
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'GroupMass']=group_df.loc[group_idx,'GroupMass']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'Group_M_Crit200']=group_df.loc[group_idx,'Group_M_Crit200']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'Group_R_Crit200']=group_df.loc[group_idx,'Group_R_Crit200']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,[f'Group_CentreOfPotential_{x}' for x in 'xyz']]=group_df.loc[group_idx,[f'CentreOfPotential_{x}' for x in 'xyz']]

            # Add relative distance to group centre
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'Group_Rrel']=np.sqrt((subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'CentreOfPotential_x'].values-group_df.loc[group_idx,'CentreOfPotential_x'])**2+(subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'CentreOfPotential_y'].values-group_df.loc[group_idx,'CentreOfPotential_y'])**2+(subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'CentreOfPotential_z'].values-group_df.loc[group_idx,'CentreOfPotential_z'])**2)

        # Append the group and subhalo dataframes to the subhalo data structure
        subhalo_dfs.append(subhalo_df)

    # Concatenate the subhalo dataframes
    if len(subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False],inplace=True)
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