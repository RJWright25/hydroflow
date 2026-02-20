import numpy as np
import pandas as pd
import os
import time
import logging
import unyt

from scipy.spatial import cKDTree
from swiftsimio import load as swiftsimio_loader
from swiftsimio import mask as swiftsimio_mask

from hydroflow.src_physics.utils import get_limits


# --------------------------------------------------------------------------------------
# READ PARTICLE DATA (COLIBRE)
# --------------------------------------------------------------------------------------
def read_subvol(path, ivol, nslice, metadata, logfile=None, verbose=False, gasonly=False):
    """
    Read particle data belonging to a spatial subvolume from a COLIBRE snapshot using swiftsimio,
    and return a unified pandas catalogue plus KDTree for spatial queries.

    The routine optionally applies a spatial mask (nslice > 1), then reads the requested particle
    types (gas-only or gas+DM+stars), converting all quantities into the HYDROFLOW analysis
    conventions. Gas fields include thermodynamic properties and (when present) on-the-fly
    hydrogen species fractions.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    path : str
        Path to the COLIBRE snapshot file readable by swiftsimio.

    ivol : int
        Index of the subvolume to read (0 ≤ ivol < nslice^3). The simulation box is
        divided evenly into nslice × nslice × nslice cubes.

    nslice : int
        Number of subdivisions along each axis defining the spatial tiling.

    metadata : object
        Metadata container produced by HYDROFLOW initialisation. Must contain:
            boxsize  : comoving box size [cMpc]
            snapshots_flist : list of snapshot filenames
            snapshots_z     : redshift per snapshot

    logfile : str, optional
        If provided, progress and diagnostics are written to this log file.

    verbose : bool, optional
        Print progress information to stdout in addition to logging.

    gasonly : bool, optional
        If True, only gas particles are read (ParticleType==0). If False, read gas+DM+stars.

    ---------------------------------------------------------------------------
    Particle selection behaviour
    ---------------------------------------------------------------------------
    • If nslice > 1, a spatial mask is applied using `swiftsimio.mask.constrain_spatial`
      with limits in comoving Mpc.
    • Subsampling is applied to reduce memory:
          DM:    keep ~10% (subset=10) and re-weight Masses by subset
          stars: keep 50%  (subset=2)  and re-weight Masses by subset
          gas:   keep all  (subset=1)

    ---------------------------------------------------------------------------
    Units of returned quantities
    ---------------------------------------------------------------------------
    Coordinates_*  : comoving Mpc
    Velocities_*   : physical km/s (peculiar), returned by swiftsimio after convert_to_physical()
    Masses         : Msun (re-weighted for subsampling)
    Gas-only:
        Temperature     : K
        SmoothingLength : comoving Mpc
        Density         : physical g/cm^3
        StarFormationRate : Msun/yr
        Metallicity     : (dimensionless mass fraction)
        mfrac_HI, mfrac_H2 : hydrogen species *mass fractions* if present in the snapshot
                             (H2 multiplied by 2 and then scaled by hydrogen element fraction)

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    pdata : pandas.DataFrame
        Unified particle catalogue sorted by ParticleIDs.

    pdata_kdtree : scipy.spatial.cKDTree
        KDTree built from (Coordinates_x, Coordinates_y, Coordinates_z).
        The tree is periodic using `boxsize` (with a small buffer to avoid edge artefacts).


    """

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    if logfile is not None:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        logging.basicConfig(filename=logfile, level=logging.INFO)
    log = logging.getLogger(__name__)

    if logfile is not None:
        log.info(f"Reading COLIBRE subvolume {ivol} from {path}...")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Metadata / snapshot scalars
    # ------------------------------------------------------------------
    boxsize = float(metadata.boxsize)  # comoving Mpc
    zval = float(metadata.snapshots_z[np.where(metadata.snapshots_flist == path)[0][0]])

    log.info(f"Boxsize: {boxsize}")
    if verbose:
        print(f"Boxsize: {boxsize}")

    # ------------------------------------------------------------------
    # Common unit targets 
    # ------------------------------------------------------------------
    # convert each field to these target units before extracting `.value`.
    unit_mass = "Msun"
    unit_dist = "Mpc"       # comoving unless convert_to_physical() called
    unit_vel = "km/s"       # then convert_to_physical() -> physical
    unit_temp = "K"
    unit_rho = "g/cm**3"    # then convert_to_physical() -> physical density
    unit_sfr = "Msun/yr"

    # ------------------------------------------------------------------
    # Load snapshot (unmasked) once (used for metadata checks)
    # ------------------------------------------------------------------
    pdata_snap = swiftsimio_loader(path)

    # ------------------------------------------------------------------
    # Spatially mask the subregion if nslice > 1
    # ------------------------------------------------------------------
    if nslice > 1:
        limits = get_limits(ivol=ivol, nslice=nslice, boxsize=boxsize)
        log.info(f"Limits given: {limits}")

        mask = swiftsimio_mask(path)
        mask.constrain_spatial([
            [limits[0] * unyt.Mpc, limits[1] * unyt.Mpc],
            [limits[2] * unyt.Mpc, limits[3] * unyt.Mpc],
            [limits[4] * unyt.Mpc, limits[5] * unyt.Mpc],
        ])
        pdata_snap_masked = swiftsimio_loader(path, mask=mask)
    else:
        pdata_snap_masked = pdata_snap

    # ------------------------------------------------------------------
    # Particle type configuration
    # ------------------------------------------------------------------
    if gasonly:
        parttypes = [0]
        partstrs = ["gas"]
        partbuffer = [pdata_snap_masked.gas]
    else:
        parttypes = [0, 1, 4]  
        partstrs = ["gas", "dm", "stars"]
        partbuffer = [
            pdata_snap_masked.gas,
            pdata_snap_masked.dark_matter,
            pdata_snap_masked.stars,
        ]

    # Storage for per-ptype DataFrames
    dfs = []

    # ------------------------------------------------------------------
    # Loop over particle types
    # ------------------------------------------------------------------
    for iptype, ptype_name, obj in zip(parttypes, partstrs, partbuffer):
        if ptype_name == "dm":
            subset = 10  # keep ~10%
        elif ptype_name == "stars":
            subset = 2   # keep 50%
        else:
            subset = 1   # keep all gas/BH

        log.info(f"Reading {ptype_name} particles... [elapsed {time.time() - t0:.2f} s]")

        # Number of rows after subsampling
        pids = obj.particle_ids.value[::subset]
        n = pids.shape[0]

        # Start assembling columns as numpy arrays
        out = {}

        # Optional: HaloCatalogueIndex if available
        if hasattr(obj, "halo_catalogue_index"):
            log.info(f"Reading HaloCatalogueIndex for {ptype_name}... [elapsed {time.time() - t0:.2f} s]")
            out["HaloCatalogueIndex"] = obj.halo_catalogue_index.value[::subset]

        # Particle IDs and type
        out["ParticleIDs"] = pids
        out["ParticleType"] = np.full(n, iptype, dtype=np.uint16)

        # Masses (re-weight by subset to preserve total mass statistically)
        log.info(f"Reading masses for {ptype_name}... [elapsed {time.time() - t0:.2f} s]")
        masses = obj.masses
        masses.convert_to_units(unit_mass)
        out["Masses"] = masses.value[::subset] * subset
        del masses

        # Coordinates (comoving Mpc)
        log.info(f"Reading coordinates for {ptype_name}... [elapsed {time.time() - t0:.2f} s]")
        coords = obj.coordinates
        coords.convert_to_units(unit_dist)  # comoving
        coords_v = coords.value[::subset, :]
        out["Coordinates_x"] = coords_v[:, 0]
        out["Coordinates_y"] = coords_v[:, 1]
        out["Coordinates_z"] = coords_v[:, 2]
        del coords, coords_v

        # Velocities (physical km/s)
        log.info(f"Reading velocities for {ptype_name}... [elapsed {time.time() - t0:.2f} s]")
        vels = obj.velocities
        vels.convert_to_units(unit_vel)
        vels.convert_to_physical()
        vels_v = vels.value[::subset, :]
        out["Velocities_x"] = vels_v[:, 0]
        out["Velocities_y"] = vels_v[:, 1]
        out["Velocities_z"] = vels_v[:, 2]
        del vels, vels_v

        # ------------------------------------------------------------------
        # Gas-only extra fields
        # ------------------------------------------------------------------
        if ptype_name == "gas":
            # Temperature (K)
            log.info(f"Reading temperature... [elapsed {time.time() - t0:.2f} s]")
            temp = obj.temperatures
            temp.convert_to_units(unit_temp)
            out["Temperature"] = temp.value[::subset]
            del temp

            # Smoothing length (comoving Mpc)
            log.info(f"Reading smoothing length... [elapsed {time.time() - t0:.2f} s]")
            hsml = obj.smoothing_lengths
            hsml.convert_to_units(unit_dist)  # comoving
            out["SmoothingLength"] = hsml.value[::subset]
            del hsml

            # Density (physical g/cm^3)
            log.info(f"Reading density... [elapsed {time.time() - t0:.2f} s]")
            dens = obj.densities
            dens.convert_to_physical()
            dens.convert_to_units(unit_rho)
            out["Density"] = dens.value[::subset] 
            del dens

            # Star formation rate (Msun/yr)
            log.info(f"Reading SFR... [elapsed {time.time() - t0:.2f} s]")
            sfr = obj.star_formation_rates
            sfr.convert_to_units(unit_sfr)
            out["StarFormationRate"] = sfr.value[::subset]
            del sfr

            # On-the-fly species fractions
            if "species_fractions" in pdata_snap.metadata.gas_properties.field_names:
                log.info(f"Reading species fractions... [elapsed {time.time() - t0:.2f} s]")

                hydrogen_frac = pdata_snap_masked.gas.element_mass_fractions.hydrogen.value
                mfrac_hi = pdata_snap_masked.gas.species_fractions.HI.value
                mfrac_h2 = pdata_snap_masked.gas.species_fractions.H2.value * 2.0 #number density -> mass fraction
                zgas = pdata_snap_masked.gas.metal_mass_fractions.value

                # Convert H species fractions to *mass fractions* by scaling with hydrogen element fraction
                out["mfrac_HI"] = mfrac_hi * hydrogen_frac
                out["mfrac_H2"] = mfrac_h2 * hydrogen_frac
                out["Metallicity"] = zgas

                del hydrogen_frac, mfrac_hi, mfrac_h2, zgas

        # ------------------------------------------------------------------
        # Star-only extra fields
        # ------------------------------------------------------------------
        elif ptype_name == "stars":
            log.info(f"Reading stellar metallicity... [elapsed {time.time() - t0:.2f} s]")
            out["Metallicity"] = obj.metal_mass_fractions.value[::subset]

        log.info(f"Appending {ptype_name} particles to pdata... [elapsed {time.time() - t0:.2f} s]")
        dfs.append(pd.DataFrame(out))

    # ------------------------------------------------------------------
    # Final concatenation and sorting
    # ------------------------------------------------------------------
    pdata = pd.concat(dfs, ignore_index=True)
    pdata.sort_values("ParticleIDs", inplace=True)
    pdata.reset_index(drop=True, inplace=True)

    # Diagnostics: gas fraction
    gas = (pdata["ParticleType"].to_numpy() == 0)
    log.info(f"Fraction of particles that are gas: {np.sum(gas) / pdata.shape[0] * 100:.2f}%")

    # ------------------------------------------------------------------
    # KDTree (periodic)
    # ------------------------------------------------------------------

    log.info(f"Generating KDTree... [elapsed {time.time() - t0:.2f} s]")
    xyz = pdata.loc[:, ["Coordinates_x", "Coordinates_y", "Coordinates_z"]].to_numpy(dtype=np.float64, copy=False)
    pdata_kdtree = cKDTree(xyz,boxsize=boxsize+1e-5)

    return pdata, pdata_kdtree
