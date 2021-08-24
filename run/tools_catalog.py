
import os
import numpy as np
import pandas as pd

from hydroflow.run.tools_hpc import create_dir

def combine_catalogs(path_subcat,path_gasflow,depth=1,snapmin=None,snapmax=None,mcut=8):
    subcat=pd.read_hdf(path_subcat)
    if 'snipshotidx' in list(subcat.columns):
        snap_key='snipshotidx'
        idx_key='nodeIndex'
        mass_key='ApertureMeasurements/Mass/030kpc_4'
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
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmax)}to{int(snapmax)}.hdf5'

    for depth in depths:

        allfiles=[]
        snapdirs=sorted(os.listdir(path_gasflow))
        snapdirs=[snapdir for snapdir in snapdirs if (f'd{str(depth).zfill(2)}' in snapdir) and ('gas' not in snapdir)]

        for snapdir in snapdirs:
            print(snapdir)
            snap=snapdir.split('snap')[-1]
            print(snap)
            snap=int(snap[:3])
            if snap>=snapmin and snap<=snapmax:
                snapdir_path=path_gasflow+snapdir
                isnap_files=sorted(os.listdir(snapdir_path))
                isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files]
                allfiles.extend(isnap_files)

        print(f'Loading gasflow files for snap {snap} delta {depth} ({len(allfiles)})')
        outputs=[]
        for iifile,file in enumerate(allfiles):
            ifile=pd.read_hdf(file,key='Gasflow')
            outputs.append(ifile)
        try:
            outputs=pd.concat(outputs)
        except:
            print(f'No outputs for {snap} depth {depth}')
            continue

        outputs.sort_values(by='hydroflowID',inplace=True)
        outputs.reset_index(drop=True,inplace=True)

        print(outputs.shape[0], f' hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')

        outputs['hydroflowID']=outputs['hydroflowID'].values.astype(np.int64)
        hydroflow=outputs['hydroflowID'].values
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
        output_columns=[column+f'_d{str(depth).zfill(2)}' for column in list(outputs.columns)]
        subcat_masked.loc[subcat_idxs,output_columns]=outputs.loc[hydroflow_idxs,:].values

    create_dir(outpath)
    subcat_masked.to_hdf(outpath,key='Gasflow')
    
    return subcat_masked