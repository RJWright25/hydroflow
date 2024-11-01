
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time

def read_subcat(basepath,prefix='m50n512_',snapnums=None):
    """
    read_subcat: Read the subhalo catalogue from a SIMBA simulation snapshot.

    Input:
    -----------
    basepath: str   
        Path to the simulation snapshot.
    prefix: str
        Prefix for the snapshot files.
    snapnums: list
        List of snapshot indices to read.

    Output:
    -----------
    subcat: pd.DataFrame
        DataFrame containing the subhalo catalogue.


    """
    
    files=os.listdir(basepath)
    numfiles=len(files)
    if snapnums is None:
        snapnums=list(range(numfiles))

    snapnums=sorted(snapnums)

    if len(snapnums)==0:
        logging.info(f'No snapshots found in {basepath}')
        return
    

    if not os.path.exists('catalogues'):
        os.mkdir('catalogues')

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

        rockstarfile=h5py.File(basepath+'/'+f'{prefix}{str(snapnum).zfill(3)}.hdf5')
        zval=rockstarfile['simulation_attributes'].attrs['redshift']

        group_df=pd.DataFrame()
        mcut=1e10

        ### group data
        group_df.loc[:,'Mass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'GroupMass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'Group_M_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.m200c'][:]
        group_df.loc[:,'Group_R_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.r200c'][:]*1e-3
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=rockstarfile['/halo_data/minpotpos'][:]*1e-3
        group_df.loc[:,'GroupNumber']=rockstarfile['/halo_data/GroupID'][:]
        group_df.loc[:,'SubGroupNumber']=0
        group_df.loc[:,'StellarMass']=np.nan
        group_df.loc[:,'StarFormationRate']=np.nan
        group_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=np.nan
        group_df.loc[:,'SnapNum']=snapnum
        group_df.loc[:,'Redshift']=zval
        group_df.loc[:,'GalaxyID']=np.int64(snapnum*1e12+group_df.loc[:,'GroupNumber'])

        group_df.sort_values('GroupNumber',inplace=True)
        group_df.reset_index(drop=True,inplace=True)

        subhalo_dfs.append(group_df)

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final subhalo data structure...')
    logging.info(f'*********************************************')

    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]

    subcat=subcat.loc[subcat.Mass>=mcut,:].copy()
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