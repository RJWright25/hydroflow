# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_catalogue.py: routine to combine output gas flow catalogues.

import os
import numpy as np
import pandas as pd
import h5py
import time
import logging

from hydroflow.run.tools_hpc import create_dir

def dump_hdf(fname,data,verbose=False):
    """
    Dump a pandas DataFrame to an hdf5 file. 

    Input:
    -----------
    fname: str
        Path to the output file.
    data: pd.DataFrame
        DataFrame to dump.
    verbose: bool
        Print progress.

    Output:
    -----------
    None
    (Creates an hdf5 file at the specified path.)

    """

    if os.path.exists(fname):
        print('Removing existing output file ...')
        os.remove(fname)
    
    columns=list(data.columns)

    outfile=h5py.File(fname,"w")

    for icol,column in enumerate(columns):
        if verbose:
            print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        outfile.create_dataset(name=column,data=data[column].values)

    outfile.close()

def read_hdf(fname,columns=None,verbose=False):
    """
    read_hdf: Read an hdf5 file (generated with dump_hdf) and return a pandas DataFrame.

    Input:
    -----------
    fname: str
        Path to the input file.

    columns: list
        List of columns to read. If None, all columns are read.

    verbose: bool
        Print progress.

    Output:
    -----------
    outdf: pd.DataFrame
        DataFrame containing the data from the hdf5 file.


    """

    infile=h5py.File(fname,mode='r')
    if not columns:
        columns=list(infile.keys())
    outdf={}
    failed=[]

    for icol, column in enumerate(columns):
        if verbose:
            print(f'Reading {column} ... {icol+1}/{len(columns)}')
        try:
            data=infile[column][:]
        except:
            if verbose:
                print(f'Failed to read {column}')
            failed.append(column)

        outdf[column]=data
    
    infile.close()

    outdf=pd.DataFrame(outdf)

    if failed:
        if verbose:
            print('Note, failed to load the following fields:')
            for column in failed:
                print(column)

    return outdf


def combine_catalogs(path_subcat,path_gasflow,depth=1,snaps=None,mcut=10,verbose=False):
    """
    combine_catalogs: Combine subhalo and gas flow catalogues.
    
    Input:
    -----------
    path_subcat: str
        Path to the subhalo catalogue.
    path_gasflow: str
        Path to the gas flow catalogues.
    depth: int or list
        Depth of the gas flow catalogues to combine (if a Lagrangian calc).
    snaps: list
        List of snapshot indices to include.
    mcut: float
        Minimum mass of subhaloes to include [log10(M/Msun)].
    verbose: bool
        Print progress.
    """
    
    t1=time.time()

    if os.path.exists('jobs/combine_catalogs.log'):
        os.remove('jobs/combine_catalogs.log')

    logging.basicConfig(filename='jobs/combine_catalogs.log', level=logging.INFO)
    logging.info(f'Reading subhalo catalog and masking... time = {time.time()-t1:.2f}) \n')
   
    subcat=pd.read_hdf(path_subcat)
    snap_key='SnapNum'
    idx_key='GalaxyID'
    mass_key='Mass'

    if not snaps:
        snap_min=int(np.nanmin(subcat[snap_key].values))
        snap_max=int(np.nanmax(subcat[snap_key].values))
        snaps=list(range(snap_min,(snap_max+1)))
    else:
        snap_min=int(np.nanmin(snaps))
        snap_max=int(np.nanmax(snaps))

    snap_mask=np.logical_or.reduce([subcat[snap_key].values==snap for snap in snaps])
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

    if snap_max-snap_min==0:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snap_max)}.hdf5'
    else:
        outpath=path_gasflow+f'/gasflow_d{depth_out}_snap{int(snap_min)}to{int(snap_max)}.hdf5'

    for depth in depths:
        logging.info(f'Reading hydroflow outputs for depth {depth} ... (t={time.time()-t1}) \n')

        snapdirs=sorted(os.listdir(path_gasflow))
        logging.info(f'Snapdirs: {snapdirs} ... \n')

        snapdirs=[snapdir for snapdir in snapdirs if (f'd{str(depth).zfill(2)}' in snapdir) and ('gas' not in snapdir)]

        snap_outputs=[]
        for snapdir in snapdirs:
            snap=snapdir.split('snap')[-1]
            snap=int(snap[:3])
            snapdir_path=path_gasflow+snapdir

            if snap in snaps:
                isnap_files=sorted(os.listdir(snapdir_path))
                isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files]
            else:
                continue

            if len(isnap_files)>0:
                logging.info(f'Loading gasflow files for snap {snap} delta {depth} ({len(isnap_files)}): t = {time.time()-t1:.2f}\n')
                if verbose: 
                    print(snapdir)
                    print(isnap_files)

                isnap_outputs=[]
                for iifile,file in enumerate(isnap_files):
                    ifile=pd.read_hdf(file,key='Gasflow')
                    isnap_outputs.append(ifile)

                isnap_outputs=pd.concat(isnap_outputs)
                snap_outputs.append(isnap_outputs)
        
        if snap_outputs:
            snap_outputs=pd.concat(snap_outputs)
            snap_outputs.sort_values(by='HydroflowID',inplace=True)
            snap_outputs.reset_index(drop=True,inplace=True)
            print(snap_outputs.shape[0], f' hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')
            logging.info(f'Columns: {list(snap_outputs.columns)}: t = {time.time()-t1:.2f}\n')

            snap_outputs['HydroflowID']=snap_outputs['HydroflowID'].values.astype(np.int64)
            hydroflow=snap_outputs['HydroflowID'].values
            nodeidx=subcat_masked[idx_key].values

            print('Matching hydroflow and subcat outputs ...')
            logging.info(f'Matching hydroflow and subcat outputs: t = {time.time()-t1:.2f}\n')

            valid_idx_hydroflow_in_subcat=np.searchsorted(a=nodeidx,v=hydroflow)
            valid_idx_hydroflow_in_hydroflow=np.array(list(range(len(hydroflow))))
            valid=valid_idx_hydroflow_in_subcat<len(nodeidx)

            valid_idx_hydroflow_in_subcat=valid_idx_hydroflow_in_subcat[np.where(valid)]
            valid_idx_hydroflow_in_hydroflow=valid_idx_hydroflow_in_hydroflow[np.where(valid)]

            print('Verifying hydroflow and subcat outputs ...')
            for index,(ihydro,isubcat) in enumerate(zip(valid_idx_hydroflow_in_hydroflow,valid_idx_hydroflow_in_subcat)):
                if not nodeidx[isubcat]==hydroflow[ihydro]:
                    print(nodeidx[isubcat],hydroflow[ihydro])
                    valid_idx_hydroflow_in_hydroflow[index]=-1
                    valid_idx_hydroflow_in_subcat[index]=-1

            hydroflow_idxs=valid_idx_hydroflow_in_hydroflow[np.where(valid_idx_hydroflow_in_hydroflow>=0)]
            subcat_idxs=valid_idx_hydroflow_in_subcat[np.where(valid_idx_hydroflow_in_subcat>=0)]

            print(f'Adding data to subcat')
            allcols=list(snap_outputs.columns)
            colsout=[col for col in allcols if not 'pkmps' in col]
            output_columns=[column+f'-d{str(depth).zfill(2)}' for column in colsout]
            subcat_masked.loc[subcat_idxs,output_columns]=snap_outputs.loc[hydroflow_idxs,colsout].values

    create_dir(outpath)
    
    subcat_masked=subcat_masked.sort_values(by=['SnapNum','Mass'],ascending=[False,False],ignore_index=True)
    subcat_masked.reset_index(inplace=True)

    dump_hdf(outpath,data=subcat_masked,verbose=True)

    return subcat_masked