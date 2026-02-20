import logging
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from hydroflow.src_physics.utils import (
    get_limits, calc_temperature, partition_neutral_gas,
    constant_gpmsun, constant_cmpkpc
)

# --------------------------------------------------------------------------------------
# READ PARTICLE DATA (SIMBA)
# --------------------------------------------------------------------------------------
def read_subvol(path, ivol, nslice, metadata, logfile=None, verbose=False, maxifile=None):
    """
    Read particle data belonging to a spatial subvolume from a SIMBA snapshot (single HDF5 file)
    and return a unified pandas catalogue plus KDTree for spatial queries.

    The routine loads the snapshot once, selects particles inside a buffered cubic subvolume,
    converts units into the HYDROFLOW analysis conventions, computes derived thermodynamic
    quantities for gas, and concatenates the results into a single DataFrame required for
    subsequent processing.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    path : str
        Path to the SIMBA snapshot file (single HDF5 file).

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
    • Particles are selected if their COMOVING position lies inside the buffered subvolume.
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
        log.info(f"Reading SIMBA subvolume {ivol} from {path}...")

    # ------------------------------------------------------------------
    # Metadata / snapshot scalars
    # ------------------------------------------------------------------
    boxsize = metadata.boxsize
    hval = metadata.hval

    snap_idx_in_metadata = np.where(metadata.snapshots_flist == path)[0][0]
    afac = float(metadata.snapshots_afac[snap_idx_in_metadata])
    zval = float(metadata.snapshots_z[snap_idx_in_metadata])

    # Precompute common conversion factors
    dconv = 1e-3 / hval            # ckpc/h -> cMpc
    mconv = 1e10 / hval            # 1e10 Msun/h -> Msun
    vconv = np.sqrt(afac)          # velocity scaling (peculiar km/s)
    rhoconv = 1e10 * (hval**2) / (afac**3)  # to Msun/pkpc^3 (from 1e10/h (ckpc/h)^-3)

    # ------------------------------------------------------------------
    # Subvolume limits (comoving Mpc)
    # ------------------------------------------------------------------
    lims = get_limits(ivol, nslice, boxsize, buffer=0.1)
    xmin, xmax, ymin, ymax, zmin, zmax = lims

    # ------------------------------------------------------------------
    # Particle configuration
    # ------------------------------------------------------------------
    # Extra fields to read per ptype (ParticleIDs/Coordinates/Velocities/Masses are always read)
    ptype_fields = {
        0: ["InternalEnergy", "ElectronAbundance", "Density", "Metallicity", "StarFormationRate"],
        1: [],
        4: ["Metallicity"],
    }

    # Downsampling stride per ptype (keep identical behaviour)
    ptype_subset = {
        0: 1,
        1: 2,
        4: 2,
    }

    coord_cols = [f"Coordinates_{d}" for d in "xyz"]
    vel_cols = [f"Velocities_{d}" for d in "xyz"]

    # Collect per-ptype DataFrames then concat once (faster than repeated concat)
    df_parts = []

    # ------------------------------------------------------------------
    # Open the SIMBA snapshot file once
    # ------------------------------------------------------------------
    with h5py.File(path, "r") as f:
        npart_thisfile = f["Header"].attrs["NumPart_ThisFile"]

        # ------------------------------------------------------------------
        # Loop over particle types
        # ------------------------------------------------------------------
        for ptype in ptype_fields.keys():
            n_this = int(npart_thisfile[ptype])
            log.info(f"Reading ptype {ptype}... (N_thisfile={n_this})")

            if n_this <= 0:
                log.info(f"No ptype {ptype} particles in this file!")
                continue

            gname = f"PartType{ptype}"
            if gname not in f:
                log.info(f"Missing group {gname} in file!")
                continue
            g = f[gname]

            # --------------------------------------------------------------
            # 1) Build subvolume mask from Coordinates only (fast, vectorised)
            # --------------------------------------------------------------
            coords = g["Coordinates"][:] * dconv  # (N,3) in cMpc

            if verbose:
                log.info(f"ptype {ptype} min/max coordinates: {coords.min()}/{coords.max()}")

            mask = (
                (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
                (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax) &
                (coords[:, 2] >= zmin) & (coords[:, 2] <= zmax)
            )

            idx = np.flatnonzero(mask)
            if idx.size == 0:
                log.info(f"No ivol ptype {ptype} particles in this file!")
                continue

            # Apply stride after masking (keeps deterministic downsampling)
            stride = int(ptype_subset[ptype])
            if stride > 1:
                idx = idx[::stride]

            # --------------------------------------------------------------
            # 2) Load always-present fields (subset by idx)
            # --------------------------------------------------------------
            pids = g["ParticleIDs"][idx].astype(np.int64, copy=False)
            coords_sel = coords[idx, :].astype(np.float64, copy=False)
            vxyz = g["Velocities"][idx, :].astype(np.float64, copy=False) * vconv

            # Masses in Msun; re-weight by stride to conserve total mass statistically
            m = g["Masses"][idx].astype(np.float64, copy=False) * mconv
            if stride > 1:
                m = m * stride

            ptype_arr = np.full(idx.size, ptype, dtype=np.uint16)

            # --------------------------------------------------------------
            # 3) Assemble output columns (exact HYDROFLOW naming)
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # 4) Load extra ptype-specific fields
            # --------------------------------------------------------------
            for field in ptype_fields[ptype]:
                try:
                    if field == "Metallicity":
                        out[field] = g[field][idx, 0]
                    else:
                        out[field] = g[field][idx].astype(np.float64, copy=False)
                except Exception:
                    log.info(f"Trouble reading field {field} for ptype {ptype}. Skipping.")
                    continue

            df_pt = pd.DataFrame(out)

            # --------------------------------------------------------------
            # 5) Unit conversions / derived quantities (gas only)
            # --------------------------------------------------------------
            if ptype == 0:
                # Density conversion to g/cm^3
                dens = df_pt["Density"].to_numpy(dtype=np.float64, copy=False)
                dens = dens * rhoconv  # Msun/pkpc^3
                dens = dens * float(constant_gpmsun) / (float(constant_cmpkpc) ** 3)  # g/cm^3
                df_pt["Density"] = dens

                # Temperature from InternalEnergy + ElectronAbundance, then drop those columns
                log.info("Calculating temperature for gas particles...")
                df_pt["Temperature"] = calc_temperature(df_pt, XH=0.76, gamma=5/3)
                if "InternalEnergy" in df_pt.columns:
                    del df_pt["InternalEnergy"]
                if "ElectronAbundance" in df_pt.columns:
                    del df_pt["ElectronAbundance"]

            df_parts.append(df_pt)

    # ------------------------------------------------------------------
    # Final concatenation and sorting (single concat)
    # ------------------------------------------------------------------
    if len(df_parts) == 0:
        pdata = pd.DataFrame()
        pdata_kdtree = cKDTree(np.empty((0, 3)))
        return pdata, pdata_kdtree

    pdata = pd.concat(df_parts, ignore_index=True)
    pdata.sort_values(by="ParticleIDs", inplace=True)
    pdata.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Diagnostics: gas fraction
    # ------------------------------------------------------------------
    gas = (pdata["ParticleType"].to_numpy() == 0)
    frac_gas = np.sum(gas) / pdata.shape[0]
    print(f"Fraction of gas particles: {frac_gas:.2e}")
    log.info(f"Fraction of gas particles: {frac_gas:.2e}")

    # ------------------------------------------------------------------
    # Hydrogen partitioning (Rahmati 2013 + Blitz & Rosolowsky 2006)
    # ------------------------------------------------------------------
    log.info("Adding hydrogen partitioning...")
    fHI, fH2, fHII = partition_neutral_gas(pdata, redshift=zval, sfonly=False)

    log.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    log.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")

    pdata.loc[:, ["mfrac_HI_BR06", "mfrac_H2_BR06"]] = np.nan
    pdata.loc[gas, "mfrac_HI_BR06"] = fHI
    pdata.loc[gas, "mfrac_H2_BR06"] = fH2

    # ------------------------------------------------------------------
    # KDTree (comoving coordinates)
    # ------------------------------------------------------------------
    log.info("Creating KDTree for particle data...")
    xyz = pdata.loc[:, coord_cols].to_numpy(dtype=np.float64, copy=False)
    pdata_kdtree = cKDTree(xyz)

    return pdata, pdata_kdtree
