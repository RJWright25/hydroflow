
import os
import numpy as np
import pandas as pd
import logging
import time
import illustris_python as tng_tools

def read_subcat(basepath,snapnums=None):
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

        subfind_raw= tng_tools.groupcat.load(basepath,snapNum=snapnum)
        groupcat=subfind_raw['halos']
        subcat=subfind_raw['subhalos']

        hfac=subfind_raw['header']['HubbleParam']
        zval=subfind_raw['header']['Redshift']
        afac=1/(1+zval)

        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        mcut=1e10

        ### group data
        group_df.loc[:,'Mass']=groupcat['GroupMass'][:]*10**10/hfac
        group_df.loc[:,'GroupMass']=groupcat['GroupMass'][:]*10**10/hfac
        group_df.loc[:,'Group_M_Crit200']=groupcat['Group_M_Crit200'][:]*10**10/hfac
        group_df.loc[:,'Group_R_Crit200']=groupcat['Group_R_Crit200'][:]*1e-3
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=groupcat['GroupPos'][:]*1e-3
        group_df.loc[:,'GroupNumber']=np.array(list(range(group_df.shape[0]))).astype(np.uint64)
        group_df.loc[:,'SubGroupNumber']=0

        group_df.loc[:,'SnapNum']=snapnum
        group_df.loc[:,'Redshift']=zval

        logging.info(f'Loaded group data {snapnum} [runtime {time.time()-t0:.2f} sec]')

        group_df=group_df.loc[group_df['Mass'].values>=mcut,:].copy()
        group_df.reset_index(drop=True,inplace=True)
        group_df.append(group_df)

        subhalo_df['GroupNumber']=subcat['SubhaloGrNr'][:]
        subhalo_df['SubfindID']=np.array(range(subhalo_df.shape[0]))
        subhalo_df['StarFormationRate']=subcat['SubhaloSFR'][:]
        subhalo_df['StellarMass']=subcat['SubhaloMassType'][:,4]*10**10/hfac
        subhalo_df['Mass']=subcat['SubhaloMass'][:]*10**10/hfac
        subhalo_df.sort_values(by='Mass',inplace=True,ascending=False);subhalo_df.reset_index(inplace=True,drop=True)
        subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat['SubhaloVel'][:,:]*np.sqrt(afac)

        subhalo_uniquegroupnums,subhalo_unique_indices=np.unique(subhalo_df['GroupNumber'].values,return_index=True)
        subhalo_mostmassive_indices=subhalo_df['SubfindID'].values[subhalo_unique_indices]
        subhalo_mostmassive_mass=subhalo_df['Mass'].values[subhalo_unique_indices]
        subhalo_mostmassive_SFR=subhalo_df['StarFormationRate'].values[subhalo_unique_indices]
        subhalo_mostmassive_StellarMass=subhalo_df['StellarMass'].values[subhalo_unique_indices]
        subhalo_mostmassive_Velocities=np.column_stack([subhalo_df[f'Velocity_{x}'].values[subhalo_unique_indices] for x in 'xyz'])*afac**(0.5)

        subhalo_df=pd.DataFrame({'GroupNumber':subhalo_uniquegroupnums,'SubfindID':subhalo_mostmassive_indices,'Mass':subhalo_mostmassive_mass,'StarFormationRate':subhalo_mostmassive_SFR,'StellarMass':subhalo_mostmassive_StellarMass})
        for idim,dim in enumerate('xyz'):
            subhalo_df[f'Velocity_{dim}']=subhalo_mostmassive_Velocities[:,idim]

        subhalo_df.sort_values(by='GroupNumber',inplace=True);subhalo_df.reset_index(inplace=True,drop=True)

        idx_of_igroup_in_subcat=subhalo_df['GroupNumber'].searchsorted(group_df['GroupNumber'].values)
        groupmatch=group_df['GroupNumber'].values==subhalo_df['GroupNumber'].values[(idx_of_igroup_in_subcat,)]
        idx_of_igroup_in_subcat=idx_of_igroup_in_subcat[np.where(groupmatch)]
        group_df.loc[groupmatch,'SubfindID']=subhalo_df['SubfindID'].values[(idx_of_igroup_in_subcat,)]
        group_df.loc[groupmatch,'StarFormationRate']=subhalo_df['StarFormationRate'].values[(idx_of_igroup_in_subcat,)]
        group_df.loc[groupmatch,'StellarMass']=subhalo_df['StellarMass'].values[(idx_of_igroup_in_subcat,)]
        group_df.loc[groupmatch,[f'Velocity_{x}' for x in 'xyz']]=np.column_stack([subhalo_df[f'Velocity_{x}'].values[(idx_of_igroup_in_subcat,)] for x in 'xyz'])
        group_df.loc[groupmatch,'GalaxyID']=np.int64(snapnum*1e12+group_df.loc[groupmatch,'SubfindID'].values)

        subhalo_dfs.append(group_df)

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

