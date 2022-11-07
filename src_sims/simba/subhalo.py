
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time

def read_subcat(basepath,snapnums=None):
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

        rockstarfile=h5py.File(basepath+f'_{str(snapnum).zfill(3)}.hdf5')

        hfac=rockstarfile['simulation_attributes'].attrs['hubble_constant']
        zval=rockstarfile['simulation_attributes'].attrs['redshift']

        group_df=pd.DataFrame()

        mcut=1e10

        ### group data
        group_df.loc[:,'Mass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'GroupMass']=rockstarfile['/halo_data/dicts/masses.total'][:]
        group_df.loc[:,'Group_M_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.m200c'][:]
        group_df.loc[:,'Group_R_Crit200']=rockstarfile['/halo_data/dicts/virial_quantities.r200c'][:]*1e-3*hfac
        group_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=rockstarfile['/halo_data/minpotpos'][:]*1e-3*hfac
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

    subcat=subcat.loc[subcat.Mass>=mcut,:]
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


def gen_btree(path,snapidxmin=0):
    path_out=path.split('.h')[0]+'_btree.hdf5'

    t0=time.time()
    if os.path.exists('jobs/logs/gen_btree.log'):
        os.remove('jobs/logs/gen_btree.log')

    logging.basicConfig(filename='jobs/logs/gen_btree.log', level=logging.INFO)
    logging.info(f'Loading subhalo catalogue from {path} [runtime {time.time()-t0:.2f} sec]')

    subcat=pd.read_hdf(path,key='Subhalo')
    subcat.reset_index(inplace=True,drop=True)

    subcat.loc[:,'DescendantID']=-1

    ### basic matching
    snapnums=sorted(np.unique(subcat['SnapNum'].values))

    logging.info(f'')
    logging.info(f'Running btree for subhaloes after (and including) snapidx {snapidxmin} ...')

    for snap in snapnums:

        nowmask=np.logical_and(subcat.SnapNum==snap,subcat.SubGroupNumber==0)
        nextmask=np.logical_and(subcat.SnapNum==(snap+1),subcat.SubGroupNumber==0)

        desc_ids=np.zeros(np.nansum(nowmask))-1

        if snap>=snapidxmin:

            logging.info(f'')
            logging.info(f'***********************************************************************')
            logging.info(f'Processing snapnum {snap} [runtime {time.time()-t0:.2f} sec]')
            logging.info(f'***********************************************************************')

            if np.nansum(nextmask):
                subcat_now=subcat.loc[nowmask,:].copy();subcat_now.reset_index(drop=True,inplace=True)
                subcat_next=subcat.loc[nextmask,:].copy();subcat_next.reset_index(drop=True,inplace=True)

                mass_next=subcat_next['Group_M_Crit200'].values
                positions_next=subcat_next.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']].values
                positions_next_x=positions_next[:,0]
                positions_next_y=positions_next[:,1]
                positions_next_z=positions_next[:,2]

                for isub,subhalo in subcat_now.iterrows():
                    loc_x_match=np.abs(positions_next_x-subhalo[f'CentreOfPotential_x'])<=0.5
                    mass_offset=np.abs(np.log10(mass_next/subhalo['Group_M_Crit200']))
                    mass_match=mass_offset<=0.5
                    match=np.logical_and(loc_x_match,mass_match)
                    
                    if not np.nansum(match):
                        continue
                    elif np.nansum(match)==1:
                        desc_ids[isub]=subcat_next.loc[match,'GalaxyID'].values[0]
                    else:
                        match=np.logical_and.reduce([match,np.abs(positions_next_y-subhalo[f'CentreOfPotential_y'])<=0.5,np.abs(positions_next_z-subhalo[f'CentreOfPotential_z'])<=0.5])
                        if np.nansum(match)==1:
                            desc_ids[isub]=subcat_next.loc[match,'GalaxyID'].values[0]
                        elif np.nansum(match)>1:
                            mass_offset_min=np.where(np.nanmin(mass_offset[match])==mass_offset[match])[0][0]
                            desc_ids[isub]=subcat_next.loc[match,'GalaxyID'].values[mass_offset_min]

            logging.info(f'')
            logging.info(f'Match rate = {np.nansum(desc_ids>0)/len(desc_ids)*100:.1f} %')

        subcat.loc[nowmask,'DescendantID']=desc_ids
    
    subcat['DescendantID']=subcat['DescendantID'].values.astype(np.int64)
    
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure to {path_out}...')
    logging.info(f'*********************************************')

    if os.path.exists(f'{path_out}'):
        os.remove(f'{path_out}')
    subcat.to_hdf(f'{path_out}',key='Subhalo')