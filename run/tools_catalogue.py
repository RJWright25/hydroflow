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




def combine_catalogues(path_subcat,path_gasflow,snaps=None,mcut=10,verbose=False):
    """
    combine_catalogues: Combine subhalo and gas flow catalogues.
    
    Input:
    -----------
    path_subcat: str
        Path to the subhalo catalogue.
    path_gasflow: str
        Path to the gas flow catalogues.
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
   
    subcat=read_hdf(path_subcat)
    snap_key='SnapNum'
    idx_key='GalaxyID'
    mass_key='Mass'


    #gasflow calc string
    calc_str='nvol'+path_gasflow.split('nvol')[-1].split('/')[0]


    if not len(snaps)>0:
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


    if snap_max-snap_min==0:
        outpath=path_gasflow+f'/gasflow_snap{str(int(snap_max)).zfill(3)}_{calc_str}.hdf5'
    else:
        outpath=path_gasflow+f'/gasflow_snap{str(int(snap_min)).zfill(3)}to{str(int(snap_max)).zfill(3))}_{calc_str}.hdf5'


    logging.info(f'Reading hydroflow outputs ... (t={time.time()-t1}) \n')


    snapdirs=sorted(os.listdir(path_gasflow))
    logging.info(f'Snapdirs: {snapdirs} ... \n')


    snapdirs=[snapdir for snapdir in snapdirs]
    snap_outputs=[]


    for snapdir in snapdirs:
        snap=snapdir.split('snap')[-1]
        snap=int(snap[:3])
        snapdir_path=path_gasflow+snapdir
        
        if snap in snaps and os.path.isdir(snapdir_path):
            isnap_files=sorted(os.listdir(snapdir_path))
            isnap_files=[snapdir_path+'/'+isnap_file for isnap_file in isnap_files if 'ivol' in isnap_file]

        else:
            continue

        if len(isnap_files)>0:
            logging.info(f'Loading gasflow files for snap {snap} ({len(isnap_files)}): t = {time.time()-t1:.2f}\n')
            if verbose: 
                print(snapdir)
                print(isnap_files)


            isnap_outputs=[]
            for iifile,file in enumerate(isnap_files):
                isnap_outputs.append(read_hdf(file))


            isnap_outputs=pd.concat(isnap_outputs)
            snap_outputs.append(isnap_outputs)
    
    if snap_outputs:
        snap_outputs=pd.concat(snap_outputs)
        snap_outputs.sort_values(by='HydroflowID',inplace=True)
        snap_outputs.reset_index(drop=True,inplace=True)
        print(snap_outputs.shape[0], f' hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs')
        logging.info(f'{snap_outputs.shape[0]} hydroflow outputs and {subcat_masked.shape[0]} masked subcat outputs\n')
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
        
        logging.info(f'Verifying indices of hydroflow and subcat outputs: t = {time.time()-t1:.2f}\n')
        print('Verifying indices of hydroflow and subcat outputs ...')
        for index,(ihydro,isubcat) in enumerate(zip(valid_idx_hydroflow_in_hydroflow,valid_idx_hydroflow_in_subcat)):
            if not nodeidx[isubcat]==hydroflow[ihydro]:
                print(nodeidx[isubcat],hydroflow[ihydro])
                valid_idx_hydroflow_in_hydroflow[index]=-1
                valid_idx_hydroflow_in_subcat[index]=-1


        hydroflow_idxs=valid_idx_hydroflow_in_hydroflow[np.where(valid_idx_hydroflow_in_hydroflow>=0)]
        subcat_idxs=valid_idx_hydroflow_in_subcat[np.where(valid_idx_hydroflow_in_subcat>=0)]
        
        logging.info(f'Adding data to subcat: t = {time.time()-t1:.2f}\n')
        print(f'Adding data to subcat')
        subcat_masked.loc[subcat_idxs,list(snap_outputs.columns)]=snap_outputs.loc[hydroflow_idxs,list(snap_outputs.columns)].values
    
    print(f'Writing to {outpath} ...')
    logging.info(f'Writing to {outpath} ...')
    create_dir(outpath)
    
    subcat_masked=subcat_masked.sort_values(by=['SnapNum','Mass'],ascending=[False,False],ignore_index=True)
    subcat_masked.reset_index(inplace=True)


    dump_hdf(outpath,data=subcat_masked,verbose=True)


    return subcat_masked
