
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time
import illustris_python as tng_tools

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

        subfind_raw= tng_tools.groupcat.load(basepath,snapNum=snapnum)
        subcat=subfind_raw['subhalos']
        groupcat=subfind_raw['halos']

        hfac=subfind_raw['header']['HubbleParam']
        zval=subfind_raw['header']['Redshift']

        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        mcut=3e10

        ### group data
        group_df.loc[:,'GroupMass']=groupcat['GroupMass'][:]*10**10/hfac
        group_df.loc[:,'Group_M_Crit200']=groupcat['Group_M_Crit200'][:]*10**10/hfac
        group_df.loc[:,'Group_R_Crit200']=groupcat['Group_R_Crit200'][:]*1e-3
        group_df.loc[:,[f'GroupCentreOfPotential_{x}' for x in 'xyz']]=groupcat['GroupPos'][:]*1e-3
        group_df.loc[:,'GroupNumber']=np.array(list(range(group_df.shape[0]))).astype(np.uint64)
        group_df=group_df.loc[group_df.GroupMass>=mcut,:].copy()
        group_df.reset_index(drop=True,inplace=True)


        ### subhalo data
        subhalo_df['GroupNumber']=subcat['SubhaloGrNr'][:]
        subhalo_df['Flag']=subcat['SubhaloFlag'][:]
        subhalo_df['Vmax']=subcat['SubhaloVmax'][:]
        subhalo_df['VmaxRadius']=subcat['SubhaloVmaxRad'][:]*1e-3
        subhalo_df['HalfMassRad']=subcat['SubhaloHalfmassRad'][:]*1e-3

        subhalo_df['Mass']=np.nansum(subcat['SubhaloMassType'][:],axis=1)*10**10/hfac
        subhalo_df.loc[:,[f'MassType_{itype}' for itype in [0,1,2,3,4,5]]]=subcat['SubhaloMassType'][:]*10**10/hfac
        subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat['SubhaloPos'][:]*1e-3
        subhalo_df.loc[:,[f'CentreOfMass_{x}' for x in 'xyz']]=subcat['SubhaloCM'][:]*1e-3
        subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat['SubhaloVel'][:]
        subhalo_df.loc[:,[f'Spin_{x}' for x in 'xyz']]=subcat['SubhaloSpin'][:]

        subhalo_df.loc[:,'SnapNum']=snapnum
        subhalo_df.loc[:,'Redshift']=zval
        subhalo_df.loc[:,'SubhaloIndex']=np.int64(list(range(subhalo_df.shape[0])))
        subhalo_df.loc[:,'SubhaloIDRaw']=np.int64(10**12*snapnum+subhalo_df.loc[:,'SubhaloIndex'].values)

        subhalo_df=subhalo_df.loc[subhalo_df['Mass'].values>=mcut/10,:].copy()
        subhalo_df.reset_index(drop=True,inplace=True)

        logging.info(f'Matching groups... [runtime {time.time()-t0:.2f} sec]')
        numgroups=group_df.shape[0]
        groupkeys=list(group_df.keys())
        for igroup,group in group_df.iterrows():
            if not igroup%1000:
                logging.info(f'{igroup/numgroups*100:.1f}% done with groups [runtime {time.time()-t0:.2f} sec]')

            groupmatch=subhalo_df['GroupNumber'].values==group['GroupNumber']
            for key in groupkeys:
                subhalo_df.loc[groupmatch,key]=group[key]
            subhalo_df.loc[groupmatch,'SubGroupNumber']=np.argsort(np.argsort(-subhalo_df.loc[groupmatch,'Mass'].values))

        subhalo_df=subhalo_df.loc[subhalo_df['GroupMass'].values>=mcut,:].copy()
        subhalo_df.reset_index(drop=True,inplace=True)


        # logging.info(f'')
        # logging.info(f'Adding trees... [runtime {time.time()-t0:.2f} sec]')

        # n=subhalo_df.shape[0]
        # subids=np.zeros(n)-1
        # descids=np.zeros(n)-1
        # progids=np.zeros(n)-1        
        # progmass=np.zeros(n)-1        
        
        # for isub,subhalo in subhalo_df.iterrows():
        #     if not isub%1000:
        #         logging.info(f'{isub/n*100:.1f}% done with trees [runtime {time.time()-t0:.2f} sec]')

        #     subhalo_idx=int(subhalo['SubhaloIndex'])
        #     subhalo_tree=tng_tools.sublink.loadTree(basepath,snapNum=subhalo['SnapNum'],id=subhalo_idx,onlyMPB=True,fields=['SubhaloID','SubhaloIDRaw','DescendantID','Mass'])
            
        #     if subhalo_tree:
        #         subids[isub]=subhalo_tree['SubhaloID'][0]
        #         descids[isub]=subhalo_tree['DescendantID'][0]
        #         try:
        #             progids[isub]=subhalo_tree['SubhaloID'][1]
        #             progmass[isub]=subhalo_tree['Mass'][1]
        #         except:
        #             progids[isub]=-2

        # subhalo_df.loc[:,'GalaxyID']=subids
        # subhalo_df.loc[:,'DescendantID']=descids
        # subhalo_df.loc[:,'MainProgenitorID']=progids
        # subhalo_df.loc[:,'MainProgenitorMass']=progmass

        subhalo_dfs.append(subhalo_df)


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



def gen_btree(path,snapidxmin=0):

    t0=time.time()
    if os.path.exists('logs/gen_btree.log'):
        os.remove('logs/gen_btree.log')

    logging.basicConfig(filename='logs/gen_btree.log', level=logging.INFO)
    logging.info(f'Loading subhalo catalogue from {path} [runtime {time.time()-t0:.2f} sec]')

    subcat=pd.read_hdf(path,key='Subhalo')
    subcat.sort_values(by=['SnapNum','Mass'],ascending=[False,False],inplace=True)
    subcat.reset_index(inplace=True,drop=True)

    if not 'GalaxyID' in list(subcat.keys()):
        subcat['GalaxyID']=subcat['SubhaloIDRaw'].values

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
