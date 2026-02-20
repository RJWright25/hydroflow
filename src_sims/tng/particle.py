import os
import h5py
import time
import logging
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import (
    get_limits, calc_temperature, partition_neutral_gas,
    constant_gpmsun, constant_cmpkpc
)

# --------------------------------------------------------------------------------------
# READ PARTICLE DATA (TNG)
# --------------------------------------------------------------------------------------
def read_subvol(path, ivol, nslice, metadata, logfile=None, verbose=False, maxifile=None):
    """
    Read particle data belonging to a spatial subvolume from a multi–file TNG/Illustris-style
    snapshot and return a unified pandas catalogue plus KDTree for spatial queries.

    The routine loops over all snapshot chunks, selects particles inside a cubic
    subvolume, converts units to physical analysis units, computes derived thermodynamic
    quantities for gas, and concatenates the results into a single DataFrame required for subsequent processing.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    path : str
        Path to *one* snapshot chunk (e.g. snap_XXX.0.hdf5). The containing directory
        is scanned and all chunks belonging to the snapshot are loaded.

    ivol : int
        Index of the subvolume to read (0 ≤ ivol < nslice^3). The simulation box is
        divided evenly into nslice × nslice × nslice cubes.

    nslice : int
        Number of subdivisions along each axis defining the spatial tiling.

    metadata : object
        Metadata container produced by HYDROFLOW initialisation. Must contain:
            boxsize  : comoving box size [cMpc]
            hval     : little-h
            snapshots_flist : list of snapshot filenames
            snapshots_afac  : scale factor per snapshot
            snapshots_z     : redshift per snapshot

    logfile : str, optional
        If provided, progress and diagnostics are written to this log file.

    verbose : bool, optional
        Print progress information to stdout in addition to logging.

    ---------------------------------------------------------------------------
    Particle selection behaviour
    ---------------------------------------------------------------------------
    • Particles are selected if their COMOVING position lies inside the subvolume.
    • A downsampling stride is applied:
            gas  (ptype 0): keep all
            DM   (ptype 1): keep every 2nd particle
            star (ptype 4): keep every 2nd particle
      Masses are re-weighted so total mass is conserved statistically.

    ---------------------------------------------------------------------------
    Units of returned quantities
    ---------------------------------------------------------------------------
    Coordinates_* : comoving Mpc
    Velocities_*  : peculiar km/s
    Masses        : Msun
    Density       : g / cm^3
    Temperature   : K

    Hydrogen partitioning:
        mfrac_HI_BR06, mfrac_H2_BR06
        computed using Rahmati (2013) + Blitz & Rosolowsky (2006)

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    pdata : pandas.DataFrame
        Unified particle catalogue sorted by ParticleIDs.
        Contains gas, dark matter, and stellar particles.

    pdata_kdtree : scipy.spatial.cKDTree
        KDTree built from (Coordinates_x, Coordinates_y, Coordinates_z)
        for fast spatial neighbour searches.

    ---------------------------------------------------------------------------
    Notes
    ---------------------------------------------------------------------------
    - The KDTree uses comoving coordinates.
    - Temperature is computed only for gas particles.
    - InternalEnergy and ElectronAbundance are removed after temperature calculation.

    """

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
    log = logging.getLogger(__name__)

    if logfile is not None:
        log.info(f"Reading subvolume {ivol} from {path}...")

    # ------------------------------------------------------------------
    # Metadata / snapshot scalars
    # ------------------------------------------------------------------
    boxsize = metadata.boxsize
    hval = metadata.hval

    snap_idx_in_metadata = np.where(metadata.snapshots_flist == path)[0][0]
    afac = float(metadata.snapshots_afac[snap_idx_in_metadata])
    zval = float(metadata.snapshots_z[snap_idx_in_metadata])

    # Precompute common conversion factors
    dconv = 1e-3/hval #to cMpc
    mconv = 1e10/hval #to Msun
    vconv = np.sqrt(afac)
    rhoconv = 1e10 * (hval**2) / (afac**3) #to Msun/pkpc^3

    # ------------------------------------------------------------------
    # Snapshot chunk list
    # ------------------------------------------------------------------
    snap_dir = os.path.dirname(path)
    isnap_flist = sorted(
        [os.path.join(snap_dir, fname) for fname in os.listdir(snap_dir) if fname.endswith(".hdf5")]
    )
    numfiles = len(isnap_flist)
    log.info(f"Reading {numfiles} files from {snap_dir}...")


    # ------------------------------------------------------------------
    # Downsampling stride per particle type
    # NOTE: stride is used as "keep every Nth particle" and then re-scale mass by stride
    # ------------------------------------------------------------------
    ptype_subset = {
        0: 1,  # gas: keep all
        1: 2,  # DM: keep every 2nd
        4: 2,  # stars: keep every 2nd
    }

    # Extra fields to read for each ptype (ParticleIDs/Coordinates/Velocities/Masses are always read)
    ptype_fields = {
        0: ["InternalEnergy", "ElectronAbundance", "Density", "StarFormationRate", "GFM_Metallicity"],
        1: [],
        4: ["GFM_Metallicity"],
    }
    # Column name templates (kept identical)
    coord_cols = [f"Coordinates_{d}" for d in "xyz"]
    vel_cols = [f"Velocities_{d}" for d in "xyz"]

    # ------------------------------------------------------------------
    # Subvolume limits (comoving Mpc)
    # ------------------------------------------------------------------
    lims = get_limits(ivol, nslice, boxsize, buffer=0.5)
    # lims = [xmin,xmax, ymin,ymax, zmin,zmax]
    xmin, xmax, ymin, ymax, zmin, zmax = lims


    # ------------------------------------------------------------------
    # Collect DataFrames into list
    # ------------------------------------------------------------------
    df_all = []
    t0 = time.time()

    if maxifile is not None:
        half_numfiles = np.round(numfiles/2).astype(int)
        isnap_flist = isnap_flist[(half_numfiles+1):(half_numfiles+maxifile+1)]
        log.info(f"Limiting to maxifile={maxifile}, now reading {len(isnap_flist)} files...")
        print(f"Limiting to maxifile={maxifile}, now reading {len(isnap_flist)} files...")
    # ------------------------------------------------------------------
    # Loop over snapshot chunks
    # ------------------------------------------------------------------
    for ifile, ifname in enumerate(isnap_flist):
        try:
            if verbose:
                print(f"Opening file {ifile+1}/{numfiles}: {ifname}")
            log.info(f"Opening file {ifile+1}/{numfiles}: {ifname}")
            print(f"Opening file {ifile+1}/{numfiles}: {ifname}")

            with h5py.File(ifname, "r") as f:
                npart_thisfile = f["Header"].attrs["NumPart_ThisFile"]
                mass_table = f["Header"].attrs["MassTable"]

                # Collect ptype DataFrames for this file, then append to df_all
                df_file_parts = []

                for ptype in ptype_fields.keys():
                    print(f"Processing particle type {ptype}... [file time: {time.time() - t0:.2f} s]")

                    if maxifile is not None:
                        print(f"maxifile={maxifile} limit is active, skipping {ptype} particles in this file.")
                        if ptype>0:
                            continue

                    n_this = int(npart_thisfile[ptype])
                    if n_this <= 0:
                        log.info(f"No {ptype} particles in this file!")
                        continue

                    gname = f"PartType{ptype}"
                    if gname not in f:
                        log.info(f"Missing group {gname} in file!")
                        continue

                    g = f[gname]

                    # ------------------------------------------------------
                    # 1) Build subvolume mask using Coordinates only
                    #    Coordinates are stored in ckpc/h; convert -> cMpc
                    # ------------------------------------------------------
                    print(f"Building spatial mask for {ptype} particles... [file time: {time.time() - t0:.2f} s]")
                    coords = g["Coordinates"][:]  # (N,3)
                    coords = coords * dconv  # ckpc/h -> cMpc

                    # Vectorised mask (no per-dimension loop)
                    mask = (
                        (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
                        (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax) &
                        (coords[:, 2] >= zmin) & (coords[:, 2] <= zmax)
                    )

                    n_invol = int(np.count_nonzero(mask))
                    if n_invol == 0:
                        log.info(f"No {ptype} particles in this file subvolume!")
                        continue

                    # Indices of particles in the subvolume, then apply stride
                    idx = np.flatnonzero(mask)
                    stride = int(ptype_subset[ptype])
                    if stride > 1:
                        idx = idx[::stride]

                    # ------------------------------------------------------
                    # 2) Load always-present fields for idx
                    # ------------------------------------------------------
                    # Particle IDs
                    print(f"Reading ParticleIDs for {ptype} particles... [file time: {time.time() - t0:.2f} s]")    
                    pids = g["ParticleIDs"][idx].astype(np.int64, copy=False)

                    # Coordinates (already converted to cMpc, but we need subset + stride)
                    coords_sel = coords[idx, :].astype(np.float64, copy=False)

                    # Velocities: stored as peculiar
                    print(f"Reading Velocities for {ptype} particles... [file time: {time.time() - t0:.2f} s]")
                    vxyz = g["Velocities"][idx, :].astype(np.float64, copy=False) * vconv

                    # Masses:
                    # - DM uses MassTable
                    # - others use particle Masses dataset
                    if ptype != 1:
                        m = g["Masses"][idx].astype(np.float64, copy=False) * mconv
                        # Re-weight mass because we're downsampling by stride (keep identical behaviour)
                        if stride > 1:
                            m = m * stride
                        m = m.astype(np.float32, copy=False)
                    else:
                        # constant DM particle mass from MassTable (in 1e10 Msun/h)
                        mconst = float(mass_table[ptype]) * mconv
                        m = (np.full(idx.shape[0], mconst, dtype=np.float32) * stride)

                    # ParticleType column (kept as uint16 like before)
                    ptype_arr = np.full(idx.shape[0], ptype, dtype=np.uint16)

                    # ------------------------------------------------------
                    # 3) Assemble DataFrame for this ptype (fast path: dict of arrays)
                    # ------------------------------------------------------
                    out = {
                        "ParticleIDs": pids,
                        "ParticleType": ptype_arr,
                        coord_cols[0]: coords_sel[:, 0],
                        coord_cols[1]: coords_sel[:, 1],
                        coord_cols[2]: coords_sel[:, 2],
                        vel_cols[0]: vxyz[:, 0],
                        vel_cols[1]: vxyz[:, 1],
                        vel_cols[2]: vxyz[:, 2],
                        "Masses": m,
                    }

                    # ------------------------------------------------------
                    # 4) Load extra ptype-specific fields (only for idx)
                    # ------------------------------------------------------
                    print(f"Reading extra fields for {ptype} particles... [file time: {time.time() - t0:.2f} s]")
                    if ptype_fields[ptype]:
                        for field in ptype_fields[ptype]:
                            try:
                                arr = g[field][idx]
                            except Exception:
                                log.info(
                                    f"Trouble reading field {field} for ptype {ptype} in file {ifile+1}/{numfiles}. "
                                    "Skipping this field."
                                )
                                continue

                            # Keep exact naming convention:
                            #   - 'GFM_Metallicity' -> 'Metallicity' (field[4:])
                            #   - everything else keeps its name
                            if "GFM" in field:
                                out[field[4:]] = arr
                            else:
                                out[field] = arr.astype(np.float64, copy=False)

                    df_pt = pd.DataFrame(out)

                    # ------------------------------------------------------
                    # 5) Unit conversions / derived quantities (kept identical)
                    # ------------------------------------------------------
                    if ptype == 0:
                        # Density: raw in 1e10/h (ckpc/h)^-3
                        # Convert to Msun/pkpc^3:
                        #   multiply by 1e10 * h^2 / a^3

                        dens = df_pt["Density"].to_numpy(dtype=np.float64, copy=False)
                        dens = dens*rhoconv   # Msun/pkpc^3

                        # Msun/pkpc^3 -> g/cm^3
                        dens = dens * float(constant_gpmsun) / (float(constant_cmpkpc) ** 3)
                        df_pt["Density"] = dens

                        # Temperature from InternalEnergy + ElectronAbundance (then drop those columns)
                        df_pt["Temperature"] = calc_temperature(df_pt, XH=0.76, gamma=5/3)
                        # Keep the exact column removals
                        if "InternalEnergy" in df_pt.columns:
                            del df_pt["InternalEnergy"]
                        if "ElectronAbundance" in df_pt.columns:
                            del df_pt["ElectronAbundance"]

                    df_file_parts.append(df_pt)

                # End ptype loop

        except Exception as e:
            log.error(f"Error reading file {ifname}: {e}")
            continue

        # Concatenate for this file once
        if df_file_parts:
            df_file = pd.concat(df_file_parts, ignore_index=True)
            df_all.append(df_file)

        log.info(f"Loaded ifile {ifile+1}/{numfiles} in {time.time() - t0:.3f} sec")

    # ------------------------------------------------------------------
    # Final concatenation across files (single concat)
    # ------------------------------------------------------------------
    log.info("Concatenating particle data across files...")
    if len(df_all) == 0:
        pdata = pd.DataFrame()
        pdata_kdtree = cKDTree(np.empty((0, 3)))
        return pdata, pdata_kdtree

    pdata = pd.concat(df_all, ignore_index=True)

    # Keep exact behaviour: sort by ParticleIDs
    pdata.sort_values(by="ParticleIDs", inplace=True)
    pdata.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Hydrogen partitioning 
    # ------------------------------------------------------------------
    log.info("Adding hydrogen partitioning...")
    gas = (pdata["ParticleType"].to_numpy() == 0)

    fHI, fH2, fHII = partition_neutral_gas(pdata, redshift=zval, sfonly=False)

    log.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    log.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")

    pdata.loc[:, ["mfrac_HI_BR06", "mfrac_H2_BR06"]] = np.nan
    pdata.loc[gas, "mfrac_HI_BR06"] = fHI
    pdata.loc[gas, "mfrac_H2_BR06"] = fH2

    frac_gas = np.sum(gas) / pdata.shape[0]
    print(f"Fraction of gas particles: {frac_gas:.2e}")
    log.info(f"Fraction of gas particles: {frac_gas:.2e}")

    # ------------------------------------------------------------------
    # KDTree 
    # ------------------------------------------------------------------
    log.info("Creating KDTree for particle data...")
    xyz = pdata.loc[:, coord_cols].to_numpy(dtype=np.float64, copy=False)
    pdata_kdtree = cKDTree(xyz)

    return pdata, pdata_kdtree
