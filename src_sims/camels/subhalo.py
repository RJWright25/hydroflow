
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

    logging.basicConfig(filename='logs/extract_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for subhaloes after (and including) snapidx {snapidxmin} ...')


    for subcat_snapnum,subcat_fname in enumerate(subcat_fnames):

        if subcat_snapnum>=snapidxmin:
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
                if ('_R_' in field):
                    group_df[field]=group_df[field]*1e-3


            subhalo_df['GroupNumber']=subcat_file['Subhalo/SubhaloGrNr'][:]
            subhalo_df['Vmax']=subcat_file['Subhalo/SubhaloVmax'][:]
            subhalo_df['VmaxRadius']=subcat_file['Subhalo/SubhaloVmaxRad'][:]*1e-3
            subhalo_df['HalfMassRad']=subcat_file['Subhalo/SubhaloHalfmassRad'][:]*1e-3
            
            subhalo_df['Mass']=np.nansum(subcat_file['Subhalo/SubhaloMassType'][:],axis=1)*10**10/hfac
            subhalo_df.loc[:,[f'MassType_{itype}' for itype in [0,1,2,3,4,5]]]=subcat_file['Subhalo/SubhaloMassType'][:]*10**10/hfac
            subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloPos'][:]*1e-3
            subhalo_df.loc[:,[f'CentreOfMass_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloCM'][:]*1e-3
            subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloVel'][:]
            subhalo_df.loc[:,[f'Spin_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloSpin'][:]

            subhalo_df.loc[:,'SnapNum']=subcat_snapnum
            subhalo_df.loc[:,'Redshift']=zval
            subhalo_df.loc[:,'SubfindIndex']=np.uint64(list(range(subhalo_df.shape[0])))
            subhalo_df.loc[:,'GalaxyID']=np.uint64(10**10*subcat_snapnum+subhalo_df.loc[:,'SubfindIndex'].values)

            subcat_file.close()

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

    if len (subhalo_dfs)>1:
        subcat=pd.concat(subhalo_dfs)
    else:
        subcat=subhalo_dfs[0]
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False])
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


def gen_btree(path,snapidxmin=0):

    t0=time.time()
    if os.path.exists('logs/gen_btree.log'):
        os.remove('logs/gen_btree.log')

    logging.basicConfig(filename='logs/gen_btree.log', level=logging.INFO)
    logging.info(f'Loading subhalo catalogue from {path} [runtime {time.time()-t0:.2f} sec]')

    subcat=pd.read_hdf(path,key='Subhalo')
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False])
    subcat.reset_index(inplace=True,drop=True)

    subcat.loc[:,'DescendantID']=-1

    path_out=path.split('.h')[0]+'_btree.hdf5'

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
                    loc_x_match=np.abs(positions_next_x-subhalo[f'CentreOfPotential_x'])<=0.2
                    mass_offset=np.abs(np.log10(mass_next/subhalo['Group_M_Crit200']))
                    mass_match=mass_offset<=0.15
                    match=np.logical_and(loc_x_match,mass_match)
                    
                    if not np.nansum(match):
                        continue
                    elif np.nansum(match)==1:
                        desc_ids[isub]=subcat_next.loc[match,'GalaxyID'].values[0]
                    else:
                        match=np.logical_and.reduce([match,np.abs(positions_next_y-subhalo[f'CentreOfPotential_y'])<=0.2,np.abs(positions_next_z-subhalo[f'CentreOfPotential_z'])<=0.2])
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

    




    