import os
import numpy as np
import pandas as pd
import h5py
import time
import logging

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd


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
    if metadata is None:
        metadata = {}
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with h5py.File(fname, "a") as f:
        # Delete group if present
        if group in f:
            del f[group]
        g = f.create_group(group)

        # Create datasets
        for icol, col in enumerate(data.columns):
            if verbose:
                print(f"Dumping {col} ... {icol+1}/{len(data.columns)}")
            g.create_dataset(col, data=data[col].to_numpy(copy=False))

        # Metadata
        for k, v in metadata.items():
            g.attrs[k] = v



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
            if len(columns)>2000:
                if icol%1000==0:
                    print(f'Dumping {column} ... {icol+1}/{len(columns)}')
            else:
                if icol%100==0:
                    print(f'Dumping {column} ... {icol+1}/{len(columns)}')
        try:            
            outfile.create_dataset(name=column,data=data[column].values)
        except:
            print(f'Failed to dump {column} - skipping')
            print(data[column].values)
            raise

    outfile.close()


def read_hdf(fname, columns=None, verbose=False):
    """
    Read an HDF5 file (written by dump_hdf) into a pandas DataFrame.
    """
    out = {}
    failed = []

    with h5py.File(fname, "r") as infile:

        # Decide which datasets to read
        if columns is None:
            cols = list(infile.keys())
        else:
            cols = list(columns)

        # skip header explicitly
        cols = [c for c in cols if c != "Header"]

        # if user passed columns, filter to those that actually exist
        # (avoids exceptions in the loop)
        if columns is not None:
            missing = [c for c in cols if c not in infile]
            if missing:
                failed.extend(missing)
            cols = [c for c in cols if c in infile]

        # Read datasets
        for icol, col in enumerate(cols, start=1):
            if verbose:
                print(f"Reading {col} ... {icol}/{len(cols)}")

            try:
                out[col] = infile[col][:]
            except (OSError, KeyError, ValueError, RuntimeError) as e:
                failed.append(col)
                if verbose:
                    print(f"Failed to read {col}: {e}")

        df = pd.DataFrame(out)

        # Header attrs -> df.attrs
        if "Header" in infile:
            hdr = infile["Header"]
            if "metadata" in hdr.attrs:
                df.attrs["metadata"] = hdr.attrs["metadata"]

    if failed and verbose:
        print("Note, failed to load the following fields:")
        for col in failed:
            print(col)

    return df


def _read_one_hdf(path):
    return read_hdf(path)


def combine_catalogues(path_hydroflow, snaps=None, mcut=10, verbose=False, nproc=None, log_every=200):

    path_hydroflow = Path(path_hydroflow)

    # logging
    if "catalogues" in str(path_hydroflow):
        path_run = Path(str(path_hydroflow).split("catalogues")[0])
        jobs_dir = path_run / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = jobs_dir / "combine_catalogues.log"
    else:
        log_path = Path("combine_catalogues.log")

    if log_path.exists():
        log_path.unlink()

    print(f"Logging to {log_path}")
    logging.basicConfig(filename=str(log_path), level=logging.INFO)
    t1 = time.time()
    logging.info(f"Started combine catalogues at {time.ctime(t1)}")

    # snapshot list
    if snaps is None:
        snaps = []
        for d in path_hydroflow.iterdir():
            if d.is_dir() and d.name.startswith("snap"):
                try:
                    snaps.append(int(d.name.split("snap")[-1][:3]))
                except ValueError:
                    pass
        snaps = sorted(set(snaps))

    if not snaps:
        logging.info("No snapshots requested / found.")
        print("No snapshots requested / found.")
        return []

    snap_set = set(snaps)
    snap_range = (min(snaps), max(snaps))

    calc_str = "nvol" + str(path_hydroflow).split("nvol")[-1].split("/")[0]
    snap_str = f"{snap_range[0]:03d}" if snap_range[0] == snap_range[1] else f"{snap_range[0]:03d}to{snap_range[1]:03d}"
    outpath = path_hydroflow / f"gasflow_snap{snap_str}_{calc_str}.hdf5"

    # gather ivol files
    ivol_files = []
    for d in path_hydroflow.iterdir():
        if not (d.is_dir() and d.name.startswith("snap")):
            continue
        try:
            snap = int(d.name.split("snap")[-1][:3])
        except ValueError:
            continue
        if snap not in snap_set:
            continue
        ivol_files.extend(d.glob("*ivol*"))

    ivol_files = [str(p) for p in ivol_files]

    if not ivol_files:
        logging.info("No hydroflow outputs found for the specified snapshots.")
        print("No hydroflow outputs found for the specified snapshots")
        return []

    # parallel read
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 4) - 1)

    logging.info(f"Found {len(ivol_files)} files. Reading with nproc={nproc} (t={time.time()-t1:.2f}s)")
    dfs = []

    t_read0 = time.time()
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = [ex.submit(_read_one_hdf, f) for f in ivol_files]
        for i, fut in enumerate(as_completed(futures), start=1):
            df = fut.result()
            if df is not None and len(df) > 0:
                dfs.append(df)
            if log_every and (i % log_every == 0):
                logging.info(f"  read {i}/{len(ivol_files)} files (dt={time.time()-t_read0:.1f}s)")

    if not dfs:
        logging.info("All reads returned empty.")
        print("All reads returned empty.")
        return []

    # combine once
    logging.info(f"Concatenating {len(dfs)} frames (t={time.time()-t1:.2f}s)")
    snap_outputs = pd.concat(dfs, ignore_index=True, copy=False)

    if "HydroflowID" in snap_outputs.columns:
        snap_outputs["HydroflowID"] = snap_outputs["HydroflowID"].astype(np.int64, copy=False)

    # final ordering
    sort_cols = ["SnapNum", "Group_M_Crit200", "SubGroupNumber"]
    if all(c in snap_outputs.columns for c in sort_cols):
        snap_outputs.sort_values(
            by=sort_cols,
            ascending=[False, False, True],
            inplace=True,
            kind="mergesort",
            ignore_index=True,
        )
    elif "HydroflowID" in snap_outputs.columns:
        snap_outputs.sort_values(by="HydroflowID", inplace=True, kind="mergesort", ignore_index=True)

    # write
    logging.info(f"Writing to {outpath} (t={time.time()-t1:.2f}s)")
    print(f"Writing to {outpath} ... (t={time.time()-t1:.2f}s)")
    create_dir(str(outpath))
    dump_hdf(str(outpath), data=snap_outputs, verbose=verbose)

    logging.info(f"Done. rows={len(snap_outputs)} (t={time.time()-t1:.2f}s)")
    return snap_outputs