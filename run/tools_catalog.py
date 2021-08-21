
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
    else:
        snap_mask=np.logical_and(subcat[snap_key]>=snapmin,subcat[snap_key]<=(snapmax+1))
    subcat_mask=np.logical_and.reduce([snap_mask,subcat[mass_key].values>=10**mcut])

    subcat_masked=subcat.loc[subcat_mask,:].copy();del subcat
    subcat_masked.sort_values(by=idx_key,inplace=True)
    subcat_masked.reset_index(drop=True,inplace=True)

    allfiles=[]
    snapdirs=sorted(os.listdir(path_gasflow))
    snapdirs=[snapdir for snapdir in snapdirs if f'd{str(depth).zfill(2)}' in snapdir]

    for snapdir in snapdirs:
        print(snapdir)
        snap=snapdir.split('snap_')
        snap=int(snapdir.split('snap_')[-1][:3])
        print(snap)
        if snap>=snapmin and snap<=snapmax:
            snapdir_path=path_gasflow+snapdir
            isnap_files=sorted(os.listdir(snapdir_path))
            isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files]
            allfiles.extend(isnap_files)

    print(f'Loading gasflow files ({len(allfiles)})')

    outputs=[]
    for iifile,file in enumerate(allfiles):
        ifile=pd.read_hdf(file,key='Gasflow')
        outputs.append(ifile)
    
    print('Done loading gasflow files')

    outputs=pd.concat(outputs)
    outputs.sort_values(by='hydroflowID',inplace=True)
    outputs.reset_index(drop=True,inplace=True)

    print(outputs.shape[0], ' hydroflow outputs')
    print(subcat_masked.shape[0], ' masked subcat outputs')

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

    print('Adding to subcat ...')
    subcat_masked.loc[subcat_idxs,list(outputs.columns)]=outputs.loc[hydroflow_idxs,:]
    print('Finished adding to subcat ...')

    outpath=path_gasflow+f'/gasflow_d{str(depth).zfill(2)}.hdf5'
    create_dir(outpath)
    subcat_masked.to_hdf(outpath,key='Gasflow')
    
    return subcat_masked