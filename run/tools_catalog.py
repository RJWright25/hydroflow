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

def dump_hdf_group(fname, group, data, metadata={}, verbose=False):
    """
    Dump a pandas DataFrame to an HDF5 file under a specified group.

    Parameters:
    ------------
    fname : str
        Output file path.
    group : str
        HDF5 group name to write data to.
    data : pd.DataFrame
        DataFrame to dump.
    metadata : dict
        Optional metadata to attach to the group.
    verbose : bool
        Whether to print progress.
    """
    if os.path.exists(fname):
        outfile = h5py.File(fname, "r+")
        if group in outfile:
            if verbose:
                print(f'Removing existing group {group} in {fname} ...')
            del outfile[group]
    else:
        if verbose:
            print(f'Creating new file {fname} ...')
        if not os.path.exists(os.path.dirname(fname)):
            try:
                os.makedirs(os.path.dirname(fname))
            except:
                pass
        outfile = h5py.File(fname, "w")

    columns = list(data.columns)

    for icol, column in enumerate(columns):
        if verbose:
            print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        if group in outfile:
            if column in outfile[group]:
                del outfile[f'{group}/{column}']
        outfile.create_dataset(name=f'{group}/{column}', data=data[column].values)

    for key in metadata.keys():
        outfile[group].attrs[key] = metadata[key]
    outfile.close()

def dump_hdf(fname, data, verbose=False):
    """
    Dump a pandas DataFrame to an HDF5 file.

    Parameters:
    ------------
    fname : str
        Output file path.
    data : pd.DataFrame
        DataFrame to write.
    verbose : bool
        Whether to print progress.
    """
    if os.path.exists(fname):
        print('Removing existing output file...')
        os.remove(fname)
    columns = list(data.columns)
    outfile = h5py.File(fname, "w")
    for icol, column in enumerate(columns):
        if verbose:
            print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        outfile.create_dataset(name=column, data=data[column].values)
    outfile.close()

def read_hdf(fname, columns=None, verbose=False):
    """
    Read an HDF5 file and return a pandas DataFrame.

    Parameters:
    ------------
    fname : str
        Path to the HDF5 file.
    columns : list or None
        Specific columns to load. If None, all columns are loaded.
    verbose : bool
        Whether to print progress.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the data.
    """
    infile = h5py.File(fname, mode='r')
    if not columns:
        columns = list(infile.keys())
    outdf = {}
    failed = []

    for icol, column in enumerate(columns):
        if verbose:
            print(f'Reading {column} ... {icol+1}/{len(columns)}')
        if not 'Header' in column:
            try:
                data = infile[column][:]
            except:
                if verbose:
                    print(f'Failed to read {column}')
                failed.append(column)
            outdf[column] = data

    outdf = pd.DataFrame(outdf)

    if 'Header' in infile.keys():
        if 'metadata' in infile['Header'].attrs.keys():
            metadata_path = infile['Header'].attrs['metadata']
            outdf.attrs['metadata'] = metadata_path

    infile.close()

    if failed:
        if verbose:
            print('Note, failed to load the following fields:')
            for column in failed:
                print(column)

    return outdf

def combine_catalogs(path_subcat, path_gasflow, snaps=None, mcut=10, verbose=False):
    """
    Combine subhalo and gas flow catalogues into a single DataFrame.

    Parameters:
    ------------
    path_subcat : str
        Path to the subhalo catalogue file.
    path_gasflow : str
        Directory containing the gas flow catalogues.
    snaps : list or None
        List of snapshot indices to include. If None, all available snapshots are used.
    mcut : float
        Minimum mass cut in log10(M/Msun).
    verbose : bool
        Whether to print progress.

    Returns:
    --------
    pd.DataFrame
        Combined catalogue.
    """
    t1 = time.time()
    if os.path.exists('jobs/combine_catalogs.log'):
        os.remove('jobs/combine_catalogs.log')
    logging.basicConfig(filename='jobs/combine_catalogs.log', level=logging.INFO)
    logging.info(f'Reading subhalo catalog and masking... time = {time.time()-t1:.2f}) \n')

    subcat = read_hdf(path_subcat)
    snap_key = 'SnapNum'
    idx_key = 'GalaxyID'
    mass_key = 'Mass'
    calc_str = 'nvol' + path_gasflow.split('nvol')[-1].split('/')[0]
    path_gasflow+='/'

    if not len(snaps) > 0:
        snap_min = int(np.nanmin(subcat[snap_key].values))
        snap_max = int(np.nanmax(subcat[snap_key].values))
        snaps = list(range(snap_min, (snap_max + 1)))
    else:
        snap_min = int(np.nanmin(snaps))
        snap_max = int(np.nanmax(snaps))

    snap_mask = np.logical_or.reduce([subcat[snap_key].values == snap for snap in snaps])
    subcat_mask = np.logical_and.reduce([snap_mask, subcat[mass_key].values >= 10 ** mcut])
    subcat_masked = subcat.loc[subcat_mask, :].copy(); del subcat
    subcat_masked.sort_values(by=idx_key, inplace=True)
    subcat_masked.reset_index(drop=True, inplace=True)

    if snap_max - snap_min == 0:
        outpath = path_gasflow + f'/gasflow_snap{str(int(snap_max)).zfill(3)}_{calc_str}.hdf5'
    else:
        outpath = path_gasflow + f'/gasflow_snap{str(int(snap_min)).zfill(3)}to{str(int(snap_max)).zfill(3)}_{calc_str}.hdf5'

    logging.info(f'Reading hydroflow outputs ... (t={time.time()-t1}) \n')
    snapdirs = sorted(os.listdir(path_gasflow))
    logging.info(f'Snapdirs: {snapdirs} ... \n')

    snap_outputs = []

    for snapdir in snapdirs:
        print("Reading gasflow files for snapdir:", path_gasflow+'/'+snapdir)
        snap = snapdir.split('snap')[-1]
        snap = int(snap[:3])
        snapdir_path = path_gasflow + snapdir

        if snap in snaps:
            isnap_files = sorted(os.listdir(snapdir_path))
            isnap_files = [snapdir_path + '/' + f for f in isnap_files if 'ivol' in f]
        else:
            continue

        if len(isnap_files) > 0:
            logging.info(f'Loading gasflow files for snap {snap} ({len(isnap_files)}): t = {time.time()-t1:.2f}\n')
            if verbose:
                print(snapdir)
                print(isnap_files)

            isnap_outputs = [read_hdf(f) for f in isnap_files]
            isnap_outputs = pd.concat(isnap_outputs)
            snap_outputs.append(isnap_outputs)

    if snap_outputs:
        snap_outputs = pd.concat(snap_outputs)
        snap_outputs.sort_values(by='HydroflowID', inplace=True)
        snap_outputs.reset_index(drop=True, inplace=True)
        print(snap_outputs.shape[0], f' hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')
        logging.info(f'{snap_outputs.shape[0]} hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs\n')
        logging.info(f'Columns: {list(snap_outputs.columns)}: t = {time.time()-t1:.2f}\n')

        snap_outputs['HydroflowID'] = snap_outputs['HydroflowID'].values.astype(np.int64)
        hydroflow = snap_outputs['HydroflowID'].values
        nodeidx = subcat_masked[idx_key].values

        print('Matching hydroflow and subcat outputs ...')
        logging.info(f'Matching hydroflow and subcat outputs: t = {time.time()-t1:.2f}\n')

        valid_idx_hydroflow_in_subcat = np.searchsorted(a=nodeidx, v=hydroflow)
        valid_idx_hydroflow_in_hydroflow = np.arange(len(hydroflow))
        valid = valid_idx_hydroflow_in_subcat < len(nodeidx)

        valid_idx_hydroflow_in_subcat = valid_idx_hydroflow_in_subcat[valid]
        valid_idx_hydroflow_in_hydroflow = valid_idx_hydroflow_in_hydroflow[valid]

        logging.info(f'Verifying indices of hydroflow and subcat outputs: t = {time.time()-t1:.2f}\n')
        print('Verifying indices of hydroflow and subcat outputs ...')
        for index, (ihydro, isubcat) in enumerate(zip(valid_idx_hydroflow_in_hydroflow, valid_idx_hydroflow_in_subcat)):
            if not nodeidx[isubcat] == hydroflow[ihydro]:
                print(nodeidx[isubcat], hydroflow[ihydro])
                valid_idx_hydroflow_in_hydroflow[index] = -1
                valid_idx_hydroflow_in_subcat[index] = -1

        hydroflow_idxs = valid_idx_hydroflow_in_hydroflow[valid_idx_hydroflow_in_hydroflow >= 0]
        subcat_idxs = valid_idx_hydroflow_in_subcat[valid_idx_hydroflow_in_subcat >= 0]

        logging.info(f'Adding data to subcat: t = {time.time()-t1:.2f}\n')
        print(f'Adding data to subcat')

        update_data = pd.DataFrame(
            snap_outputs.loc[hydroflow_idxs, list(snap_outputs.columns)].values,
            columns=list(snap_outputs.columns),
            index=subcat_idxs
        )
        subcat_masked.update(update_data)

    print(f'Writing to {outpath} ...')
    logging.info(f'Writing to {outpath} ...')
    create_dir(outpath)

    subcat_masked = subcat_masked.sort_values(by=['SnapNum', 'Mass'], ascending=[False, False], ignore_index=True)
    subcat_masked.reset_index(inplace=True)
    dump_hdf(outpath, data=subcat_masked, verbose=True)

    return subcat_masked
