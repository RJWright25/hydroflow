
import os
import numpy as np
import pandas as pd
import logging
import time
import illustris_python as tng_tools

def read_subcat(basepath,snapnums=None):

    """
    read_subcat: Read the subhalo catalogue from an Illustris simulation snapshot. Uses the illustris_python package.
    
    Input:
    -----------
    basepath: str   
        Path to the simulation snapshot.
    snapnums: list
        List of snapshot indices to read.

    Output:
    -----------
    subcat: pd.DataFrame
        DataFrame containing the subhalo catalogue.

    """
    mcut=3.16e10
    mstarcut=3.16e8

    files=os.listdir(basepath)
    numfiles=len(files)
    if snapnums is None:
        snapnums=list(range(numfiles))

    if len(snapnums)==0:
        logging.info(f'No snapshots found in {basepath}')
        return
    
    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('jobs/logs'):
        os.mkdir('jobs/logs')

    snapm1=snapnums[-1]
    if os.path.exists(f'jobs/logs/read_subcat_{snapm1}.log'):
        os.remove(f'jobs/logs/read_subcat_{snapm1}.log')

        
    t0=time.time()
    logging.basicConfig(filename=f'jobs/logs/read_subcat_{snapm1}.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for {len(snapnums)} snaps ending at {snapnums[-1]} ...')

    subhalo_dfs=[]

    for snapnum in snapnums:
        logging.info(f'')
        logging.info(f'***********************************************************************')
        logging.info(f'Processing snapnum {snapnum} [runtime {time.time()-t0:.2f} sec]')
        logging.info(f'***********************************************************************')

        subfind_raw=tng_tools.groupcat.load(basepath,snapNum=snapnum)
        groupcat=subfind_raw['halos']
        subcat=subfind_raw['subhalos']

        hfac=subfind_raw['header']['HubbleParam']
        zval=subfind_raw['header']['Redshift']
        afac=1/(1+zval)

        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        ### group data
        group_df.loc[:,'Mass']=groupcat['GroupMass'][:]*10**10/hfac
        group_df.loc[:,'GroupMass']=groupcat['GroupMass'][:]*10**10/hfac
        group_df.loc[:,'Group_M_Crit200']=groupcat['Group_M_Crit200'][:]*10**10/hfac
        group_df.loc[:,'Group_R_Crit200']=groupcat['Group_R_Crit200'][:]*1e-3
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=groupcat['GroupPos'][:]*1e-3
        group_df.loc[:,'GroupNumber']=np.float64(list(range(group_df.shape[0])))
        group_df.loc[:,'SubGroupNumber']=0
        group_df.loc[:,'SnapNum']=snapnum
        group_df.loc[:,'Redshift']=zval
        group_df.sort_values(by='GroupNumber',inplace=True,ascending=False)
        logging.info(f'Loaded group data {snapnum} [runtime {time.time()-t0:.2f} sec]')

        group_df=group_df.loc[group_df['Mass'].values>=mcut,:].copy()
        group_df.reset_index(drop=True,inplace=True)

        ### subhalo data
        subhalo_df['GroupNumber']=np.float64(subcat['SubhaloGrNr'][:])
        subhalo_df['SubfindID']=np.array(range(subhalo_df.shape[0]))
        subhalo_df['StarFormationRate']=subcat['SubhaloSFR'][:]
        subhalo_df['StellarMass']=subcat['SubhaloMassType'][:,4]*10**10/hfac
        subhalo_df['Mass']=subcat['SubhaloMass'][:]*10**10/hfac
        subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat['SubhaloPos'][:]*1e-3
        subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat['SubhaloVel'][:,:]*np.sqrt(afac)
        subhalo_df.loc[:,'SnapNum']=snapnum
        subhalo_df.loc[:,'Redshift']=zval

        keys_groups=['GroupMass','Group_M_Crit200','Group_R_Crit200','Group_CentreOfPotential_x','Group_CentreOfPotential_y','Group_CentreOfPotential_z']
        for key in keys_groups:
            subhalo_df[key]=np.zeros(subhalo_df.shape[0])+np.nan
        
        #sort subhalo data
        subhalo_df.sort_values(by=['GroupNumber','Mass'],inplace=True,ascending=[False,True])
        subhalo_df=subhalo_df.loc[np.logical_and(subhalo_df['Mass'].values>=mcut,subhalo_df['StellarMass'].values>=mstarcut),:].copy()
        subhalo_df.reset_index(inplace=True,drop=True)

        print(subhalo_df.loc[:,["Mass","GroupNumber"]])

        #match subhalo data to group data
        print('Matching group data to subhalo data...')
        unique_groups=subhalo_df['GroupNumber'].unique()
        for igroup,group in enumerate(unique_groups):
            print(group)
            if igroup%1000==0:
                print(f'Group {igroup+1}/{unique_groups.shape[0]}...')
                print(f"Searching for {group} in {group_df['GroupNumber'].values} ...")
            group_idx=np.searchsorted(group_df['GroupNumber'].values,group)
            if group!=group_df['GroupNumber'].values[group_idx]:
                print(f"Group {group} does not match {group_df['GroupNumber'].values[group_idx]}...")
                continue
            subhalo_idx_1=np.searchsorted(subhalo_df['GroupNumber'].values,group)
            subhalo_idx_2=np.searchsorted(subhalo_df['GroupNumber'].values,subhalo_df['GroupNumber'].values[igroup+1])
            
            if subhalo_idx_2-subhalo_idx_1==0:
                print(f'No subhalos in group {group}...')
                continue

            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'SubGroupNumber']=np.array(range(np.sum(subhalo_idx_2-subhalo_idx_1)))
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'GroupMass']=group_df.loc[group_idx,'GroupMass']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'Group_M_Crit200']=group_df.loc[group_idx,'Group_M_Crit200']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,'Group_R_Crit200']=group_df.loc[group_idx,'Group_R_Crit200']
            subhalo_df.loc[subhalo_idx_1:subhalo_idx_2,[f'Group_CentreOfPotential_{x}' for x in 'xyz']]=group_df.loc[group_idx,[f'CentreOfPotential_{x}' for x in 'xyz']]

        subhalo_dfs.append(subhalo_df)

        logging.info(f'Matched group data {snapnum} [runtime {time.time()-t0:.2f} sec]')

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final subhalo data structure...')
    logging.info(f'*********************************************')

    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False],inplace=True)
    subcat.reset_index(inplace=True,drop=True)

    if len(snapnums)==1:
        outname=f'catalogues/catalogue_subhalo_{str(int(snapnums[0])).zfill(3)}.hdf5'
    else:
        outname=f'catalogues/catalogue_subhalo_{str(int(snapnums[0])).zfill(3)}_to_{str(int(snapnums[-1])).zfill(3)}.hdf5'    
    
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure to {outname}...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subcat.to_hdf(f'{outname}',key='Subhalo')

    return subcat