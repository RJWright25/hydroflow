# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)
#
# src_sims/eaglesnap/particle.py:
# Routines to read and convert particle data from EAGLE snapshot outputs.
# Uses the EagleSnapshot class from the `pyread_eagle` package.

import numpy as np
import pandas as pd
import h5py
import logging

from scipy.spatial import KDTree
from pyread_eagle import EagleSnapshot

from hydroflow.src_physics.utils import (
    get_limits, partition_neutral_gas,
    constant_gpmsun, constant_spyr
)

# --------------------------------------------------------------------------------------
# READ PARTICLE DATA (EAGLE)
# --------------------------------------------------------------------------------------
def read_subvol(path, ivol, nslice, metadata, logfile=None, verbose=False):
    """
    Read particle data belonging to a spatial subvolume from an EAGLE snapshot (single file)
    using `pyread_eagle.EagleSnapshot`, and return a unified pandas catalogue plus KDTree.

    The routine selects particles inside the requested subvolume, reads required
    particle columns (IDs, types, coordinates, velocities, masses), converts additional baryonic
    fields into physical/cgs units using the dataset conversion attributes, and concatenates
    gas + DM + stars into a single DataFrame required for subsequent processing.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    path : str
        Path to the EAGLE snapshot file.

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
      `EagleSnapshot.select_region` expects coordinates in cMpc/h, so the HYDROFLOW
      cMpc limits are multiplied by h before selection.
    • A downsampling stride is applied:
            gas  (ptype 0): keep all
            DM   (ptype 1): keep every 2nd particle
            star (ptype 4): keep every 2nd particle
      Masses are re-weighted so total mass is conserved statistically.

    ---------------------------------------------------------------------------
    Units of returned quantities
    ---------------------------------------------------------------------------
    Coordinates_* : comoving Mpc
    Velocities_*  : peculiar km/s  (scaled by sqrt(a) to match HYDROFLOW conventions)
    Masses        : Msun

    Extra baryonic fields are converted using EAGLE dataset attributes:
        value_physical_cgs = value_raw * (h^hexp) * (a^aexp) * CGSConversionFactor

    Gas-only:
        StarFormationRate is converted from g/s to Msun/yr.

    Hydrogen partitioning:
        mfrac_HI_BR06, mfrac_H2_BR06
        computed using Rahmati (2013) + Blitz & Rosolowsky (2006)

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    pdata : pandas.DataFrame
        Unified particle catalogue sorted by ParticleIDs.
        Contains gas, dark matter, and stellar particles. For star/DM particles, missing gas-only fields are
        present and set to NaN.

    pdata_kdtree : scipy.spatial.KDTree
        KDTree built from (Coordinates_x, Coordinates_y, Coordinates_z).
        `boxsize` is passed to enable periodic distance calculations.

    ---------------------------------------------------------------------------
    Notes
    ---------------------------------------------------------------------------
    - This routine uses KDTree (not cKDTree) to preserve existing behaviour.
    - The KDTree uses comoving coordinates.
    - EAGLESnapshot reads only the region previously selected with select_region.
    """

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if logfile is not None:
        logging.basicConfig(filename=logfile, level=logging.INFO)
    log = logging.getLogger(__name__)

    if logfile is not None:
        log.info(f"Reading EAGLE subvolume {ivol} from {path}...")

    # ------------------------------------------------------------------
    # Metadata / snapshot scalars
    # ------------------------------------------------------------------
    boxsize = metadata.boxsize
    hval = metadata.hval

    snap_idx_in_metadata = np.where(metadata.snapshots_flist == path)[0][0]
    afac = float(metadata.snapshots_afac[snap_idx_in_metadata])
    zval = float(metadata.snapshots_z[snap_idx_in_metadata])

    # Precompute common conversion factors
    dconv = 1/hval #to cMpc
    mconv = 1e10/hval #to Msun
    vconv = np.sqrt(afac)

    # ------------------------------------------------------------------
    # Subvolume limits (comoving Mpc) and EAGLE region selection (cMpc/h)
    # ------------------------------------------------------------------
    lims = get_limits(ivol, nslice, boxsize)  # [xmin,xmax,ymin,ymax,zmin,zmax] in cMpc
    xmin, xmax, ymin, ymax, zmin, zmax = lims

    # EAGLESnapshot expects cMpc/h limits
    xmin_h, xmax_h = xmin * hval, xmax * hval
    ymin_h, ymax_h = ymin * hval, ymax * hval
    zmin_h, zmax_h = zmin * hval, zmax * hval

    # ------------------------------------------------------------------
    # Fields and downsampling configuration
    # ------------------------------------------------------------------
    # Extra fields to read per ptype (ParticleIDs/Coordinates/Velocity/Mass are always read)
    ptypes = {
        0: ["Temperature", "Metallicity", "Density", "StarFormationRate"],
        1: [],
        4: ["Metallicity"],
    }

    # Stride: keep every Nth particle. Masses are re-weighted by stride.
    ptype_subset = {
        0: 1,
        1: 2,
        4: 2,
    }

    coord_cols = [f"Coordinates_{d}" for d in "xyz"]
    vel_cols = [f"Velocities_{d}" for d in "xyz"]

    # ------------------------------------------------------------------
    # Open HDF5 file for unit conversion attributes (hexp/aexp/cgs)
    # ------------------------------------------------------------------
    with h5py.File(path, "r") as f:
        # Used only for DM constant mass (MassTable) and conversion attrs
        mass_table = f["Header"].attrs["MassTable"]

        # ------------------------------------------------------------------
        # Set up EagleSnapshot and select region
        # ------------------------------------------------------------------
        snapshot = EagleSnapshot(path)
        snapshot.select_region(
            xmin=xmin_h, xmax=xmax_h,
            ymin=ymin_h, ymax=ymax_h,
            zmin=zmin_h, zmax=zmax_h
        )

        # Collect per-ptype DataFrames then concat once
        df_parts = []

        # ------------------------------------------------------------------
        # Loop over particle types
        # ------------------------------------------------------------------
        for ptype in ptypes.keys():
            stride = int(ptype_subset[ptype])

            # --------------------------------------------------------------
            # 1) Always-read datasets (IDs, coords, velocities, masses)
            # --------------------------------------------------------------
            log.info(f"Reading ptype {ptype} particle IDs, coordinates & velocities...")

            # These arrays are already region-selected by EagleSnapshot
            pids = snapshot.read_dataset(ptype, "ParticleIDs")[::stride].astype(np.int64, copy=False)
            coords = snapshot.read_dataset(ptype, "Coordinates")[::stride] * dconv  # -> cMpc
            vxyz = snapshot.read_dataset(ptype, "Velocity")[::stride, :] * vconv  # peculiar km/s

            n = pids.shape[0]
            if n == 0:
                continue

            # Masses in Msun, re-weighted by stride
            if ptype == 1:
                # DM: constant mass from MassTable (in 1e10 Msun/h)
                mconst = float(mass_table[1]) * mconv
                m = np.full(n, mconst * stride, dtype=np.float64)
            else:
                m = snapshot.read_dataset(ptype, "Mass")[::stride] * mconv
                if stride > 1:
                    m = m * stride

            # ParticleType
            ptype_arr = np.full(n, ptype, dtype=np.uint16)

            out = {
                "ParticleIDs": pids,
                "ParticleType": ptype_arr,
                coord_cols[0]: coords[:, 0],
                coord_cols[1]: coords[:, 1],
                coord_cols[2]: coords[:, 2],
                vel_cols[0]: vxyz[:, 0],
                vel_cols[1]: vxyz[:, 1],
                vel_cols[2]: vxyz[:, 2],
                "Masses": m,
            }

            # --------------------------------------------------------------
            # 2) Extra baryonic properties in physical/cgs units
            # --------------------------------------------------------------
            if ptypes[ptype]:
                log.info("Reading extra baryonic properties...")

                for field in ptypes[ptype]:
                    try:
                        # Read conversion attributes from the raw HDF5 dataset
                        dset = f[f"PartType{ptype}/{field}"]
                        hexp = dset.attrs["h-scale-exponent"]
                        aexp = dset.attrs["aexp-scale-exponent"]
                        cgs = dset.attrs["CGSConversionFactor"]

                        # Read the region-selected raw values and apply the same conversion
                        raw = snapshot.read_dataset(ptype, field)[::stride]
                        out[field] = raw * (hval ** hexp) * (afac ** aexp) * cgs
                    except Exception as e:
                        log.info(f"Trouble reading/converting field {field} for ptype {ptype}. Skipping. ({e})")
                        continue

            df_pt = pd.DataFrame(out)

            # --------------------------------------------------------------
            # 3) Post-processing: gas SFR conversion and star NaN padding
            # --------------------------------------------------------------
            if ptype == 0 and "StarFormationRate" in df_pt.columns:
                # Convert SFR from g/s -> Msun/yr
                df_pt["StarFormationRate"] = df_pt["StarFormationRate"] * (1.0 / constant_gpmsun) * constant_spyr

            df_parts.append(df_pt)

    # ------------------------------------------------------------------
    # Ensure star particles have the same extra columns as gas (NaN fill)
    # ------------------------------------------------------------------
    if len(df_parts) == 0:
        pdata = pd.DataFrame()
        pdata_kdtree = KDTree(np.empty((0, 3)), boxsize=boxsize)
        return pdata, pdata_kdtree

    # Identify gas and star frames (if present)
    # We do this after creation to avoid repeated per-row assignments.
    df_gas = None
    df_star = None
    df_other = []

    for df in df_parts:
        # each df is single ptype, so ParticleType unique
        ptype_unique = int(df["ParticleType"].iloc[0])
        if ptype_unique == 0:
            df_gas = df
        elif ptype_unique == 4:
            df_star = df
        else:
            df_other.append(df)

    if (df_gas is not None) and (df_star is not None):
        # Any gas-only extra fields missing in stars should exist and be NaN
        gas_fields = set(ptypes[0])
        star_fields = set(ptypes[4])
        missing = list(gas_fields.difference(star_fields))

        # Add missing columns as NaN without changing existing values
        for col in missing:
            if col not in df_star.columns:
                df_star[col] = np.nan

        # Rebuild df_parts in a deterministic order
        df_parts = [df_gas] + df_other + [df_star]
    else:
        # If stars not present, just proceed
        df_parts = df_parts

    # ------------------------------------------------------------------
    # Concatenate, sort, and reset index (single concat)
    # ------------------------------------------------------------------
    log.info("Concatenating particle data...")
    pdata = pd.concat(df_parts, ignore_index=True)
    pdata.sort_values(by="ParticleIDs", inplace=True)
    pdata.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Hydrogen partitioning (Rahmati 2013 + Blitz & Rosolowsky 2006)
    # ------------------------------------------------------------------
    log.info("Adding hydrogen partitioning...")
    gas = (pdata["ParticleType"].to_numpy() == 0)
    fHI, fH2, fHII = partition_neutral_gas(pdata, redshift=zval, sfonly=False)

    log.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    log.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")

    pdata.loc[:, ["mfrac_HI_BR06", "mfrac_H2_BR06"]] = np.nan
    pdata.loc[gas, "mfrac_HI_BR06"] = fHI
    pdata.loc[gas, "mfrac_H2_BR06"] = fH2

    # ------------------------------------------------------------------
    # Diagnostics: gas fraction
    # ------------------------------------------------------------------
    frac_gas = np.sum(gas) / pdata.shape[0]
    print(f"Fraction of gas particles: {frac_gas:.2e}")
    log.info(f"Fraction of gas particles: {frac_gas:.2e}")

    # ------------------------------------------------------------------
    # KDTree (comoving coordinates, periodic box)
    # ------------------------------------------------------------------
    log.info("Creating KDTree for particle data...")
    xyz = pdata.loc[:, coord_cols].to_numpy(dtype=np.float64, copy=False)
    pdata_kdtree = KDTree(xyz, boxsize=boxsize)

    return pdata, pdata_kdtree
