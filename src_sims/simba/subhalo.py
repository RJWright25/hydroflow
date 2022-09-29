
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time

def read_subcat(basepath,snapnums=None):
    snapm1=snapnums[-1]
    if os.path.exists(f'logs/extract_subhalo_{snapm1}.log'):
        os.remove(f'logs/extract_subhalo_{snapm1}.log')
        
    t0=time.time()
    logging.basicConfig(filename=f'logs/extract_subhalo_{snapm1}.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for {len(snapnums)} snaps ending at {snapnums[-1]} ...')

    subhalo_dfs=[]

    for snapnum in snapnums:
        logging.info(f'')
        logging.info(f'***********************************************************************')
        logging.info(f'Processing snapnum {snapnum} [runtime {time.time()-t0:.2f} sec]')
        logging.info(f'***********************************************************************')

        rockstarfile=h5py.File(basepath+f'_{str(snapnum).zfill(3)}.hdf5')

        hfac=rockstarfile['simulation_attributes'].attrs['hubble_constant']
        zval=rockstarfile['simulation_attributes'].attrs['redshift']

        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        mcut=1e10

        ### group data
        group_df.loc[:,'Mass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'GroupMass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'Group_M_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.m200c'][:]
        group_df.loc[:,'Group_R_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.r200c'][:]*1e-3*hfac
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=rockstarfile['/halo_data/minpotpos'][:]*1e-3*hfac
        group_df.loc[:,'GroupNumber']=np.array(list(range(group_df.shape[0]))).astype(np.uint64)
        group_df.loc[:,'SubGroupNumber']=0
        group_df['GalaxyID']=np.uint64(group_df['GroupNumber'].values+1e12*snapnum)

        group_df.loc[:,'SnapNum']=snapnum
        group_df.loc[:,'Redshift']=zval

        group_df=group_df.loc[group_df['Mass'].values>=mcut,:].copy()
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
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False],inplace=True)
    subcat.reset_index(inplace=True,drop=True)

    outname=f'catalogues/catalogue_subhalo_{str(int(snapnums[0])).zfill(3)}_to_{str(int(snapnums[-1])).zfill(3)}.hdf5'
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure to {outname}...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subcat.to_hdf(f'{outname}',key='Subhalo')

    return subcat
