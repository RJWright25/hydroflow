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


def dump_hdf_group(fname,group,data,metadata={},verbose=False):
    """
    Dump a pandas DataFrame to an hdf5 file in a specified group.


    Input:
    -----------
    fname: str
        Path to the output file.
    group: str
        Group within the hdf5 file.
    data: pd.DataFrame
        DataFrame to dump.
    metadata: dict
        Dictionary containing metadata to add to the group (optional).
    verbose: bool
        Print progress.


    Output:
    -----------
    None
    (Creates a hdf5 file at the specified path if doesn't exist, or add a new group.)


    """
    # Check if the file exists
    if os.path.exists(fname):
        outfile=h5py.File(fname,"r+")
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
        outfile=h5py.File(fname,"w")
    columns=list(data.columns)


    # Create the group and add the requested columns
    for icol,column in enumerate(columns):
        if verbose:
            print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        
        # Remove data if it already exists
        if group in outfile:
            if column in outfile[group]:
                del outfile[f'{group}/{column}']


        outfile.create_dataset(name=f'{group}/{column}',data=data[column].values)


    # Add optional metadata to the group
    for key in metadata.keys():
        outfile[group].attrs[key]=metadata[key]
    outfile.close()




# Dump a pandas DataFrame to an hdf5 file
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
        print('Removing existing output file...')
        os.remove(fname)
    
    columns=list(data.columns)


    outfile=h5py.File(fname,"w")


    for icol,column in enumerate(columns):
        if verbose:
            print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        outfile.create_dataset(name=column,data=data[column].values)


    outfile.close()


# Read an hdf5 file and return a pandas DataFrame
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


    # Open the hdf5 file and get columns if not specified
    infile=h5py.File(fname,mode='r')
    if not columns:
        columns=list(infile.keys())
    outdf={}
    failed=[]


    # Read the requested columns
    for icol, column in enumerate(columns):
        if verbose:
            print(f'Reading {column} ... {icol+1}/{len(columns)}')


        # Skip the header if it exists; read the data otherwise
        if not 'Header' in column:
            try:
                data=infile[column][:]
            except:
                if verbose:
                    print(f'Failed to read {column}')
                failed.append(column)


            outdf[column]=data
    
    outdf=pd.DataFrame(outdf)


    # Search for metadata in the header
    if 'Header' in infile.keys():
        if 'metadata' in infile['Header'].attrs.keys():
            metadata_path=infile['Header'].attrs['metadata']
            outdf.attrs['metadata']=metadata_path


    infile.close()


    # Print any failed columns
    if failed:
        if verbose:
            print('Note, failed to load the following fields:')
            for column in failed:
                print(column)


    return outdf


def combine_catalogues(path_subcat, path_gasflow, snaps=None, mcut=10, verbose=False):
    t1 = time.time()

    if 'catalogues'  in path_subcat:
        path_run=path_subcat.split('catalogues')[0]
        log_path = path_run+'/jobs/combine_catalogues.log'
        if not os.path.exists(path_run+'jobs'):
            os.makedirs(path_run+'jobs')
    else:
        log_path = 'combine_catalogues.log'

    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    print('Reading subhalo catalogue and masking...')
    logging.info(f"Started processing at {time.ctime(t1)}")

    subcat = read_hdf(path_subcat)
    snap_key, idx_key, mass_key = 'SnapNum', 'GalaxyID', 'Mass'

    calc_str = 'nvol' + path_gasflow.split('nvol')[-1].split('/')[0]

    if not snaps:
        snaps = list(range(int(np.nanmin(subcat[snap_key])), int(np.nanmax(subcat[snap_key])) + 1))
    snap_set = set(snaps)

    mask = subcat[snap_key].isin(snap_set) & (subcat[mass_key] >= 10 ** mcut)
    subcat_masked = subcat.loc[mask].copy()
    subcat_masked.sort_values(by=idx_key, inplace=True)
    subcat_masked.reset_index(drop=True, inplace=True)
    del subcat

    snap_range = (min(snaps), max(snaps))
    snap_str = f"{snap_range[0]:03d}" if snap_range[0] == snap_range[1] else f"{snap_range[0]:03d}to{snap_range[1]:03d}"
    outpath = os.path.join(path_gasflow, f"gasflow_snap{snap_str}_{calc_str}.hdf5")

    print('Reading hydroflow outputs ...')
    logging.info(f"Reading hydroflow outputs... Time: {time.time() - t1:.2f}s")

    snapdirs = sorted([d for d in os.listdir(path_gasflow) if d.startswith("snap")])
    print(f'Snapdirs: {snapdirs}')
    logging.info(f"Snapdirs: {snapdirs}")

    snap_outputs = []

    for snapdir in snapdirs:
        try:
            snap = int(snapdir.split("snap")[-1][:3])
        except ValueError:
            continue
        if snap not in snap_set:
            continue

        snapdir_path = os.path.join(path_gasflow, snapdir)
        if not os.path.isdir(snapdir_path):
            continue

        isnap_files = [os.path.join(snapdir_path, f) for f in os.listdir(snapdir_path) if 'ivol' in f]
        if not isnap_files:
            continue

        print(snapdir)
        print(isnap_files)
        logging.info(f"Loading {len(isnap_files)} files for snap {snap}")
        isnap_outputs = [read_hdf(f) for f in isnap_files]
        snap_outputs.append(pd.concat(isnap_outputs, ignore_index=True))

    if not snap_outputs:
        print('No hydroflow outputs found.')
        logging.info("No hydroflow outputs found.")
        return subcat_masked

    snap_outputs = pd.concat(snap_outputs, ignore_index=True)
    snap_outputs.sort_values(by='HydroflowID', inplace=True)
    snap_outputs.reset_index(drop=True, inplace=True)
    snap_outputs['HydroflowID'] = snap_outputs['HydroflowID'].astype(np.int64)

    print(f'{snap_outputs.shape[0]} hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')
    logging.info(f"{len(snap_outputs)} hydroflow rows; columns = {len(snap_outputs.columns)}")

    print('Matching hydroflow and subcat outputs ...')
    logging.info("Matching hydroflow and subcat outputs...")

    # Match by GalaxyID <-> HydroflowID
    common_ids = np.intersect1d(subcat_masked['GalaxyID'].values, snap_outputs['HydroflowID'].values)

    hydroflow_matched = snap_outputs[snap_outputs['HydroflowID'].isin(common_ids)].copy()
    hydroflow_matched.set_index('HydroflowID', inplace=True)
    subcat_masked.set_index('GalaxyID', inplace=True)

    print('Adding data to subcat')
    logging.info("Adding data to subcat...")

    # Combine matched rows using concat to preserve all columns, then drop-and-replace
    subcat_rows = subcat_masked.loc[hydroflow_matched.index]
    combined_rows = pd.concat([subcat_rows, hydroflow_matched], axis=1)
    combined_rows = combined_rows.loc[:, ~combined_rows.columns.duplicated(keep='last')]

    # Drop old rows and append updated ones
    subcat_masked = subcat_masked.drop(index=combined_rows.index, errors='ignore')
    subcat_masked = pd.concat([subcat_masked, combined_rows], axis=0)

    # De-fragment and restore index
    subcat_masked = subcat_masked.copy()
    subcat_masked.reset_index(inplace=True)

    print(f'Writing to {outpath} ...')
    logging.info(f"Writing to {outpath}")
    create_dir(outpath)

    subcat_masked.sort_values(by=['SnapNum', 'Mass'], ascending=[False, False], inplace=True, ignore_index=True)
    subcat_masked.reset_index(drop=True, inplace=True)
    dump_hdf(outpath, data=subcat_masked, verbose=verbose)

    return subcat_masked