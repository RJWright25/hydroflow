# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_catalogue.py: routine to combine output gas flow catalogues.

import os
import numpy as np
import pandas as pd

from hydroflow.run.tools_hpc import create_dir

def combine_catalogs(path_subcat,path_gasflow,depth=1,snapmin=None,snapmax=None,mcut=8,verbose=False):
    subcat=pd.read_hdf(path_subcat)

    snap_key='SnapNum'
    idx_key='GalaxyID'
    mass_key='Mass'

    snaplims=np.logical_and(snapmin,snapmax)

    if not snaplims:
        snap_mask=subcat[snap_key].values>=0
        snapmin=np.nanmin(subcat[snap_key].values);snapmax=np.nanmax(subcat[snap_key].values)
    else:
        snap_mask=np.logical_and(subcat[snap_key]>=snapmin,subcat[snap_key]<=(snapmax+1))

    subcat_mask=np.logical_and.reduce([snap_mask,subcat[mass_key].values>=10**mcut])
    subcat_masked=subcat.loc[subcat_mask,:].copy();del subcat
    subcat_masked.sort_values(by=idx_key,inplace=True)
    subcat_masked.reset_index(drop=True,inplace=True)

    if depth:
        depths=[depth]
        depth_out=str(depth).zfill(2)
    else:
        depths=list(range(1,13))
        depth_out='all'

    if snapmax-snapmin==0:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmax)}.hdf5'
    else:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmin)}to{int(snapmax)}.hdf5'

    for depth in depths:
        print(depth)

        snapdirs=sorted(os.listdir(path_gasflow))
        snapdirs=[snapdir for snapdir in snapdirs if (f'd{str(depth).zfill(2)}' in snapdir) and ('gas' not in snapdir)]

        snap_outputs=[]
        for snapdir in snapdirs:
            snap=snapdir.split('snap')[-1]
            snap=int(snap[:3])
            snapdir_path=path_gasflow+snapdir
            if snap>=snapmin and snap<=snapmax and os.path.exists(snapdir_path):

                isnap_files=sorted(os.listdir(snapdir_path))
                isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files]
            else:
                continue
            
            print(f'Loading gasflow files for snap {snap} delta {depth} ({len(isnap_files)})')

            if verbose: 
                print(snapdir)
                print(isnap_files)

            isnap_outputs=[]
            for iifile,file in enumerate(isnap_files):
                ifile=pd.read_hdf(file,key='Gasflow')
                isnap_outputs.append(ifile)
            try:
                isnap_outputs=pd.concat(isnap_outputs)
                isnap_outputs.sort_values(by='HydroflowID',inplace=True)
                isnap_outputs.reset_index(drop=True,inplace=True)
                snap_outputs.append(isnap_outputs)
            except:
                print(f'No outputs for {snap} depth {depth}')
                continue
        
        if snap_outputs:
            snap_outputs=pd.concat(snap_outputs)
            snap_outputs.sort_values(by='HydroflowID',inplace=True)
            snap_outputs.reset_index(drop=True,inplace=True)
            print(snap_outputs.shape[0], f' hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')

            snap_outputs['HydroflowID']=snap_outputs['HydroflowID'].values.astype(np.int64)
            hydroflow=snap_outputs['HydroflowID'].values
            nodeidx=subcat_masked[idx_key].values

            valid_idx_hydroflow_in_subcat=np.searchsorted(a=nodeidx,v=hydroflow)
            valid_idx_hydroflow_in_hydroflow=np.array(list(range(len(hydroflow))))
            valid=valid_idx_hydroflow_in_subcat<len(nodeidx)

            valid_idx_hydroflow_in_subcat=valid_idx_hydroflow_in_subcat[np.where(valid)]
            valid_idx_hydroflow_in_hydroflow=valid_idx_hydroflow_in_hydroflow[np.where(valid)]

            for index,(ihydro,isubcat) in enumerate(zip(valid_idx_hydroflow_in_hydroflow,valid_idx_hydroflow_in_subcat)):
                if not nodeidx[isubcat]==hydroflow[ihydro]:
                    print(nodeidx[isubcat],hydroflow[ihydro])
                    valid_idx_hydroflow_in_hydroflow[index]=-1
                    valid_idx_hydroflow_in_subcat[index]=-1

            hydroflow_idxs=valid_idx_hydroflow_in_hydroflow[np.where(valid_idx_hydroflow_in_hydroflow>=0)]
            subcat_idxs=valid_idx_hydroflow_in_subcat[np.where(valid_idx_hydroflow_in_subcat>=0)]

            print(f'Adding depth {depth} data to subcat')
            output_columns=[column+f'-d{str(depth).zfill(2)}' for column in list(snap_outputs.columns)]
            subcat_masked.loc[subcat_idxs,output_columns]=snap_outputs.loc[hydroflow_idxs,:].values


    create_dir(outpath)
    subcat_masked=subcat_masked.sort_values(by=['SnapNum','Mass'],ascending=[False,False],ignore_index=True)
    subcat_masked.reset_index(inplace=True)

    subcat_masked.to_hdf(outpath,key='Gasflow')


    
    return subcat_masked