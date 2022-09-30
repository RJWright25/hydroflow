
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
        group_df.loc[:,[f'GroupCentreOfPotential_{x}' for x in 'xyz']]=rockstarfile['/halo_data/minpotpos'][:]*1e-3*hfac
        group_df.loc[:,'GroupNumber']=rockstarfile['/halo_data/GroupID'][:]

        # group_df=group_df.loc[group_df['Mass'].values>=mcut,:].copy()
        group_df.sort_values('GroupNumber',inplace=True)
        group_df.reset_index(drop=True,inplace=True)

        subhalo_df['GalaxyIndex']=np.array(rockstarfile['/galaxy_data/GroupID'][:])
        subhalo_df['GalaxyID']=np.uint64(subhalo_df['GalaxyIndex'].values+snapnum*1e12)
        if 'descend_galaxy_star' in list(rockstarfile['tree_data'].keys()):
            subhalo_df['DescendantIndex']=np.array(rockstarfile['/tree_data/descend_galaxy_star'][:])
            subhalo_df['DescendantID']=np.uint64(subhalo_df['DescendantIndex'].values+(snapnum+1)*1e12)
        else:
            subhalo_df.loc[:,'DescendantIndex']=-1
            subhalo_df.loc[:,'DescendantID']=-1

        subhalo_df['GroupNumber']=rockstarfile['/galaxy_data/parent_halo_index'][:]
        subhalo_df['SubGroupNumber']=np.logical_not(rockstarfile['/galaxy_data/central'][:]).astype(np.uint16)
        subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=rockstarfile['/galaxy_data/minpotpos'][:]*1e-3*hfac
        subhalo_df=subhalo_df.loc[subhalo_df['Mass'].values>=mcut,:].copy()
        subhalo_df.reset_index(inplace=True,drop=True)
        subhalo_df.loc[:,'SnapNum']=snapnum
        subhalo_df.loc[:,'Redshift']=zval

        ##add groups
        totransfer=['GroupMass','Group_M_Crit200','Group_R_Crit200','GroupCentreOfPotential_x','GroupCentreOfPotential_y','GroupCentreOfPotential_z']
        idx_subhalo_in_group=group_df['GroupNumber'].searchsorted(subhalo_df['GroupNumber'].values)
        subhalo_df.loc[:,totransfer]=group_df.loc[idx_subhalo_in_group,totransfer].values
        subhalo_df['Mass']=subhalo_df['GroupMass']
        subhalo_dfs.append(subhalo_df)

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final subhalo data structure...')
    logging.info(f'*********************************************')

    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]

    subcat.sort_values(by=['SnapNum','Mass'],ascending=[True,False],inplace=True)
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
