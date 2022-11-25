# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_catalogue.py: routine to combine output gas flow catalogues.

import os
import numpy as np
import pandas as pd

from hydroflow.run.tools_hpc import create_dir

def pddf_to_hdf(filename, data, columns=None, maxColSize=200, **kwargs):
    """Write a `pandas.DataFrame` with a large number of columns
    to one HDFStore.

    Parameters
    -----------
    filename : str
        name of the HDFStore
    data : pandas.DataFrame
        data to save in the HDFStore
    columns: list
        a list of columns for storing. If set to `None`, all 
        columns are saved.
    maxColSize : int (default=2000)
        this number defines the maximum possible column size of 
        a table in the HDFStore.

    """
    import numpy as np
    from collections import ChainMap
    store = pd.HDFStore(filename, **kwargs)
    if columns is None:
        columns = data.columns
    colSize = columns.shape[0]
    if colSize > maxColSize:
        numOfSplits = np.ceil(colSize / maxColSize).astype(int)
        colsSplit = [
            columns[i * maxColSize:(i + 1) * maxColSize]
            for i in range(numOfSplits)
        ]
        _colsTabNum = ChainMap(*[
            dict(zip(columns, ['data{}'.format(num)] * colSize))
            for num, columns in enumerate(colsSplit)
        ])
        colsTabNum = pd.Series(dict(_colsTabNum)).sort_index()
        for num, cols in enumerate(colsSplit):
            store.put('data{}'.format(num), data[cols], format='table')
        store.put('colsTabNum', colsTabNum, format='fixed')
    else:
        store.put('data', data[columns], format='table')
    store.close()




def hdf_to_pddf(filename, columns=None, **kwargs):
    """Read a `pandas.DataFrame` from a HDFStore.

    Parameter
    ---------
    filename : str
        name of the HDFStore
    columns : list
        the columns in this list are loaded. Load all columns, 
        if set to `None`.

    Returns
    -------
    data : pandas.DataFrame
        loaded data.

    """
    store = pd.HDFStore(filename)
    data = []
    colsTabNum = store.select('colsTabNum')
    if colsTabNum is not None:
        if columns is not None:
            tabNums = pd.Series(
                index=colsTabNum[columns].values,
                data=colsTabNum[columns].values).sort_index()
            print(tabNums)
            for table in tabNums.unique():
                print(table)
                data.append(
                    store.select(table, columns=tabNums[table], **kwargs))
        else:
            for table in colsTabNum.unique():
                data.append(store.select(table, **kwargs))
        data = pd.concat(data, axis=1).sort_index(axis=1)
    else:
        data = store.select('data', columns=columns)
    store.close()
    return data


def combine_catalogs(path_subcat,path_gasflow,depth=1,snapmin=None,snapmax=None,mcut=10,verbose=False):
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

    print(np.nanmean(subcat[mass_key].values>=10**mcut))

    subcat_mask=np.logical_and.reduce([snap_mask,subcat[mass_key].values>=10**mcut])
    subcat_masked=subcat.loc[subcat_mask,:].copy();del subcat
    subcat_masked.sort_values(by=idx_key,inplace=True)
    subcat_masked.reset_index(drop=True,inplace=True)

    if len(str(depth))==1:
        depths=[depth]
        depth_out=str(depth).zfill(2)
    else:
        depths=depth
        depth_out='x'

    if snapmax-snapmin==0:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmax)}.hdf5'
        outpath_compressed=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmax)}_subset.hdf5'
    else:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmin)}to{int(snapmax)}.hdf5'
        outpath_compressed=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snapmin)}to{int(snapmax)}_subset.hdf5'

    for depth in depths:

        snapdirs=sorted(os.listdir(path_gasflow))
        snapdirs=[snapdir for snapdir in snapdirs if (f'd{str(depth).zfill(2)}' in snapdir) and ('gas' not in snapdir)]
        snap_insnapdirs=[int(snapdir.split('snap')[-1][:3]) for snapdir in snapdirs]

        snap_outputs=[]
        for snapdir in snapdirs:
            snap=snapdir.split('snap')[-1]
            snap=int(snap[:3])
            snapdir_path=path_gasflow+snapdir
            if snap>=snapmin and snap<=snapmax:
                isnap_files=sorted(os.listdir(snapdir_path))
                isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files]
            else:
                continue

            if len(isnap_files)>0:
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

            print('Matching hydroflow and subcat outputs')
            valid_idx_hydroflow_in_subcat=np.searchsorted(a=nodeidx,v=hydroflow)
            valid_idx_hydroflow_in_hydroflow=np.array(list(range(len(hydroflow))))
            valid=valid_idx_hydroflow_in_subcat<len(nodeidx)

            valid_idx_hydroflow_in_subcat=valid_idx_hydroflow_in_subcat[np.where(valid)]
            valid_idx_hydroflow_in_hydroflow=valid_idx_hydroflow_in_hydroflow[np.where(valid)]

            print('Verifying hydroflow and subcat outputs')
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
    
    print(f'Compressing output for desired snaps')
    mask_output=np.zeros(subcat_masked.shape[0])
    for snap in snap_insnapdirs:
        mask_output=np.logical_or(mask_output,subcat_masked.SnapNum==snap)

    subcat_masked=subcat_masked.loc[mask_output,:].copy()
    subcat_masked=subcat_masked.sort_values(by=['SnapNum','Mass'],ascending=[False,False],ignore_index=True)
    subcat_masked.reset_index(inplace=True)
    pddf_to_hdf(outpath,data=subcat_masked)

    return subcat_masked