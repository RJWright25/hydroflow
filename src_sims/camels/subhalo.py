
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time

#read gadget output
def read_subcat(path,snapidxmin=0):
    group_fields=['GroupMass',
                  'Group_M_Crit200',
                  'Group_R_Crit200',
                  'GroupPos']

    subcat_fnames=sorted([path+fname for fname in os.listdir(path) if 'hdf5' in fname])
    subhalo_dfs=[]

    if os.path.exists('logs/extract_subhalo.log'):
        os.remove('logs/extract_subhalo.log')

    t0=time.time()

    logging.basicConfig(filename='jobs/logs/extract_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for subhaloes after (and including) snapidx {snapidxmin} ...')


    for subcat_snapnum,subcat_fname in enumerate(subcat_fnames):

        if subcat_snapnum>snapidxmin:
            logging.info(f'')
            logging.info(f'***********************************************************************')
            logging.info(f'Processing snapnum {subcat_snapnum} [runtime {time.time()-t0:.2f} sec]')
            logging.info(f'***********************************************************************')

            subcat_file=h5py.File(subcat_fname,'r')

            hfac=subcat_file['Header'].attrs['HubbleParam']
            zval=subcat_file['Header'].attrs['Redshift']
            
            group_df=pd.DataFrame()
            subhalo_df=pd.DataFrame()

            for field in group_fields: 
                if ('Potential' in field) or ('Pos' in field):
                    group_df.loc[:,[f'GroupCentreOfPotential_{x}' for x in 'xyz']]=subcat_file['Group/'+field][:]*1e-3
                else:
                    group_df[field]=subcat_file['Group'][field][:]

                if ('Mass' in field) or ('_M_' in field):
                    group_df[field]=group_df[field]*10**10/hfac

            subhalo_df['GroupNumber']=subcat_file['Subhalo/SubhaloGrNr'][:]
            subhalo_df['Vmax']=subcat_file['Subhalo/SubhaloVmax'][:]
            subhalo_df['VmaxRadius']=subcat_file['Subhalo/SubhaloVmaxRad'][:]*1e-3
            subhalo_df['HalfMassRad']=subcat_file['Subhalo/SubhaloHalfmassRad'][:]*1e-3
            
            subhalo_df['Mass']=np.nansum(subcat_file['Subhalo/SubhaloMassType'][:],axis=1)*10**10/hfac
            subhalo_df.loc[:,[f'MassType_{itype}' for itype in [0,1,2,3,4,5]]]=subcat_file['Subhalo/SubhaloMassType'][:]*10**10/hfac
            subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloPos'][:]*1e-3
            subhalo_df.loc[:,[f'CentreOfMass_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloCM'][:]*1e-3
            subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloVel'][:]
            subhalo_df.loc[:,[f'Spin_{x}' for x in 'xyz']]=s=subcat_file['Subhalo/SubhaloSpin'][:]

            subhalo_df.loc[:,'SnapNum']=subcat_snapnum
            subhalo_df.loc[:,'Redshift']=zval
            subhalo_df.loc[:,'SubfindIndex']=np.uint64(list(range(subhalo_df.shape[0])))
            subhalo_df.loc[:,'GalaxyID']=np.uint64(10**10*subcat_snapnum+subhalo_df.loc[:,'SubfindIndex'].values)

            subcat_file.close()

            subhalo_df=subhalo_df.loc[subhalo_df['Mass'].values>=5e9,:].copy()
            subhalo_df.reset_index(drop=True,inplace=True)

            for groupnum in list(range(group_df.shape[0])):
                groupmatch=subhalo_df['GroupNumber']==groupnum
                subhalo_df.loc[groupmatch,list(group_df.keys())[1:]]=group_df.iloc[groupnum].to_numpy()[1:]
                subhalo_df.loc[groupmatch,'SubGroupNumber']=np.argsort(np.argsort(-subhalo_df.loc[groupmatch,'Mass'].values)).astype(int)

            subhalo_dfs.append(subhalo_df)

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final subhalo data structure...')
    logging.info(f'*********************************************')


    subcat=pd.concat(subhalo_dfs)
    subcat.sort_values(by=['Mass','SnapNum'],ascending=[False,False])
    subcat.reset_index(inplace=True,drop=True)


    outname='catalogues/catalogue_subhalo.hdf5'
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure to {outname}...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subcat.to_hdf(f'{outname}',key='Subhalo')

    return subcat






    