
import os
import h5py
import numpy as np
import pandas as pd
import logging
import time
import illustris_python as tng_tools

def read_subcat(basepath,snapnums=None):
    if os.path.exists('logs/extract_subhalo.log'):
        os.remove('logs/extract_subhalo.log')
        
    t0=time.time()
    logging.basicConfig(filename='logs/extract_subhalo.log', level=logging.INFO)
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

        mcut=1e12

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
        groupkeys=list(group_df.keys())[1:]
        for igroup,group in group_df.iterrows():
            if not igroup%1000:
                logging.info(f'{igroup/numgroups*100:.1f}% done with groups [runtime {time.time()-t0:.2f} sec]')

            groupmatch=subhalo_df['GroupNumber'].values==group['GroupNumber']
            for key in groupkeys:
                subhalo_df.loc[groupmatch,key]=group[key]
            subhalo_df.loc[groupmatch,'SubGroupNumber']=np.argsort(np.argsort(-subhalo_df.loc[groupmatch,'Mass'].values))

        subhalo_df=subhalo_df.loc[subhalo_df['GroupMass'].values>=mcut,:].copy()
        subhalo_df.reset_index(drop=True,inplace=True)


        logging.info(f'')
        logging.info(f'Adding trees... [runtime {time.time()-t0:.2f} sec]')

        n=subhalo_df.shape[0]
        subids=np.zeros(n)-1
        descids=np.zeros(n)-1
        progids=np.zeros(n)-1        
        progmass=np.zeros(n)-1        
        
        for isub,subhalo in subhalo_df.iterrows():
            if not isub%1000:
                logging.info(f'{isub/n*100:.1f}% done with trees [runtime {time.time()-t0:.2f} sec]')

            subhalo_idx=int(subhalo['SubhaloIndex'])
            subhalo_tree=tng_tools.sublink.loadTree(basepath,snapNum=subhalo['SnapNum'],id=subhalo_idx,onlyMPB=True,fields=['SubhaloID','SubhaloIDRaw','DescendantID','Mass'])
            
            if subhalo_tree:
                subids[isub]=subhalo_tree['SubhaloID'][0]
                descids[isub]=subhalo_tree['DescendantID'][0]
                try:
                    progids[isub]=subhalo_tree['SubhaloID'][1]
                    progmass[isub]=subhalo_tree['Mass'][1]
                except:
                    progids[isub]=-2

        subhalo_df.loc[:,'GalaxyID']=subids
        subhalo_df.loc[:,'DescendantID']=descids
        subhalo_df.loc[:,'MainProgenitorID']=progids
        subhalo_df.loc[:,'MainProgenitorMass']=progmass

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

    outname=f'catalogues/catalogue_subhalo_{str(int(snapnums[0])).zfill(3)}_to_{str(int(snapnums[-1])).zfill(3)}.hdf5'
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure to {outname}...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subcat.to_hdf(f'{outname}',key='Subhalo')

    return subcat