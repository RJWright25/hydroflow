import os
import numpy as np
import pandas as pd
import h5py

from hydroflow.run.tools_catalogue import dump_hdf
from hydroflow.run.initialise import load_metadata

import illustris_python as tng_tools


def extract_subhaloes(path, mcut=1e10, metadata=None):
    """
    Build a subhalo catalogue from IllustrisTNG / Illustris Subfind group catalogues
    using the `illustris_python` loader.

    This routine:
      1) loads the Subfind group catalogue for one or more snapshots,
      2) builds a host-halo table (FoF groups) and a galaxy table (subhaloes),
      3) attaches host-halo properties to each subhalo (vectorised),
      4) assigns SubGroupNumber within each FoF group (0 = most-massive subhalo by construction),
      5) computes Group_Rrel (distance to host group centre),
      6) applies mass cuts and writes an HDF5 catalogue.

    Parameters
    ----------
    path : str or list[str]
        One or more paths that include "groups_XXX" (directory or file).
        The snapshot number is parsed from the substring after "groups_".

    mcut : float, optional
        Minimum host-halo mass cut applied to FoF groups and downstream subhaloes (Msun).

    metadata : str or object, optional
        If a string: path to the HYDROFLOW metadata pickle.
        If None: searches the current working directory for a ".pkl" metadata file.

    Returns
    -------
    subcat : pandas.DataFrame
        Subhalo catalogue sorted by (SnapNum desc, Group_M_Crit200 desc, SubGroupNumber asc),
        and written to "./catalogues/subhaloes.hdf5".

    Notes on units
    -------------
    Illustris/TNG group catalogues store:
      - masses in units of 1e10 Msun/h
      - positions/radii in ckpc/h

    HYDROFLOW conventions used here:
      - masses in Msun
      - distances in comoving Mpc (cMpc)
    """

    # ------------------------------------------------------------------
    # Normalise inputs
    # ------------------------------------------------------------------
    if isinstance(path, str):
        path = [path]

    if len(path) == 0:
        print("No catalogue paths given. Exiting...")
        return None

    # ------------------------------------------------------------------
    # Load metadata (either provided or autodiscovered)
    # ------------------------------------------------------------------
    if metadata is not None:
        metadata_path = metadata
        metadata = load_metadata(metadata)
    else:
        metadata_path = None
        metadata = None
        for fname in os.listdir(os.getcwd()):
            if fname.endswith(".pkl"):
                metadata_path = fname
                metadata = load_metadata(metadata_path)
                print(f"Metadata file found: {metadata_path}")
                break

    if metadata is None:
        raise RuntimeError("No metadata provided and no .pkl metadata file found in the current directory.")

    hval = float(metadata.hval)

    # ------------------------------------------------------------------
    # Unit conversions
    # ------------------------------------------------------------------
    mconv = 1e10 / hval      # 1e10 Msun/h -> Msun
    dconv = 1e-3 / hval      # ckpc/h -> cMpc

    # ------------------------------------------------------------------
    # Derive snapshot numbers from input paths and fetch scale factors from metadata
    # ------------------------------------------------------------------
    snapnums, afacs = [], []
    for ipath in path:
        token = ipath.split("groups_")[-1]
        snapnum = int(token[:3])
        snapnums.append(snapnum)

        midx = np.where(metadata.snapshots_idx == snapnum)[0][0]
        afacs.append(float(metadata.snapshots_afac[midx]))

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    outpath = os.path.join(os.getcwd(), "catalogues", "subhaloes.hdf5")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Base path for illustris_python loader
    basepath = path[0].split("/groups_")[0]

    subhalo_dfs = []

    # ------------------------------------------------------------------
    # Iterate over snapshots
    # ------------------------------------------------------------------
    for isnap, snapnum in enumerate(snapnums):
        print(f"Loading snapshot {snapnum}...")

        subfind_raw = tng_tools.groupcat.load(basepath, snapNum=snapnum)
        groupcat = subfind_raw["halos"]
        subcat_raw = subfind_raw["subhalos"]

        afac = afacs[isnap]
        zval = (1.0 / afac) - 1.0

        # ------------------------------------------------------------------
        # Host-halo (FoF) table
        # ------------------------------------------------------------------
        print("Extracting group data...")
        numgroups = int(groupcat["GroupMass"].shape[0])

        group_mass = groupcat["GroupMass"][:] * mconv
        keep_group = group_mass >= mcut

        group_numbers_keep = np.flatnonzero(keep_group).astype(np.int64, copy=False)

        group_M200 = groupcat["Group_M_Crit200"][:] * mconv
        group_R200 = groupcat["Group_R_Crit200"][:] * dconv
        gpos = groupcat["GroupPos"][:] * dconv

        # ------------------------------------------------------------------
        # Subhalo (galaxy) table
        # ------------------------------------------------------------------
        print("Extracting subhalo data...")
        numsub = int(subcat_raw["SubhaloMass"].shape[0])

        sub_group = subcat_raw["SubhaloGrNr"][:].astype(np.int64, copy=False)

        # Drop subhaloes whose host group fails the host mass cut (fast boolean mask)
        # (GroupNumber indices are 0..numgroups-1, so keep_group is directly indexable)
        keep_sub = keep_group[sub_group]
        if not np.any(keep_sub):
            subhalo_dfs.append(pd.DataFrame())
            print()
            continue

        sub_group = sub_group[keep_sub]
        sub_mass = (subcat_raw["SubhaloMass"][:] * mconv)[keep_sub]
        sub_sfr = subcat_raw["SubhaloSFR"][:][keep_sub]
        sub_mstar = (subcat_raw["SubhaloMassType"][:, 4] * mconv)[keep_sub]
        spos = (subcat_raw["SubhaloPos"][:] * dconv)[keep_sub]

        # ------------------------------------------------------------------
        # SubGroupNumber assignment (vectorised)
        # ------------------------------------------------------------------
        # Subfind convention: subhaloes are stored sorted by GroupNumber, and within each group
        # they are ordered by decreasing bound mass. Under that ordering, index-within-group
        # is the SubGroupNumber (0 = most massive).
        #
        # If the raw catalogue is not sorted by GroupNumber, sort once here.
        order = np.argsort(sub_group, kind="mergesort")
        sub_group = sub_group[order]
        sub_mass = sub_mass[order]
        sub_sfr = sub_sfr[order]
        sub_mstar = sub_mstar[order]
        spos = spos[order, :]

        # Compute within-group ranks: 0..(count-1) for each contiguous group block
        # counts per group via run-length encoding
        change = np.r_[True, sub_group[1:] != sub_group[:-1]]
        first_idx = np.flatnonzero(change)
        # group sizes
        sizes = np.diff(np.r_[first_idx, sub_group.size])
        # ranks = [0..sizes[0]-1, 0..sizes[1]-1, ...]
        sub_sgn = np.concatenate([np.arange(n, dtype=np.int32) for n in sizes])

        # ------------------------------------------------------------------
        # Host property mapping (vectorised direct indexing)
        # ------------------------------------------------------------------
        host_mass = group_mass[sub_group]
        host_M200 = group_M200[sub_group]
        host_R200 = group_R200[sub_group]
        host_pos = gpos[sub_group, :]

        # Distance to group centre
        dr = spos - host_pos
        group_rrel = np.sqrt(np.sum(dr * dr, axis=1))

        # ------------------------------------------------------------------
        # Assemble DataFrame 
        # ------------------------------------------------------------------
        subhalo_df = pd.DataFrame({
            "SnapNum": np.full(sub_group.shape[0], snapnum, dtype=np.int32),
            "Redshift": np.full(sub_group.shape[0], zval, dtype=np.float64),

            "GroupNumber": sub_group,
            "GalaxyID": np.arange(sub_group.shape[0], dtype=np.int64),

            "StarFormationRate": sub_sfr,
            "StellarMass": sub_mstar,
            "Mass": sub_mass,

            "CentreOfMass_x": spos[:, 0],
            "CentreOfMass_y": spos[:, 1],
            "CentreOfMass_z": spos[:, 2],

            "SubGroupNumber": sub_sgn,

            "GroupMass": host_mass,
            "Group_M_Crit200": host_M200,
            "Group_R_Crit200": host_R200,

            "Group_CentreOfMass_x": host_pos[:, 0],
            "Group_CentreOfMass_y": host_pos[:, 1],
            "Group_CentreOfMass_z": host_pos[:, 2],

            "Group_Rrel": group_rrel,
        })

        subhalo_dfs.append(subhalo_df)
        print()

    # ------------------------------------------------------------------
    # Concatenate across snapshots
    # ------------------------------------------------------------------
    subhalo_dfs = [df for df in subhalo_dfs if df is not None and len(df) > 0]
    if len(subhalo_dfs) == 0:
        return pd.DataFrame()

    if len(subhalo_dfs) > 1:
        subcat = pd.concat(subhalo_dfs, ignore_index=True)
    else:
        subcat = subhalo_dfs[0]

    # ------------------------------------------------------------------
    # Final selection + sorting
    # ------------------------------------------------------------------
    mask = np.logical_and(
        subcat["Group_M_Crit200"].to_numpy() >= mcut,
        np.logical_or(
            subcat["SubGroupNumber"].to_numpy() == 0,
            subcat["Mass"].to_numpy() > mcut * 10 ** (-0.5),
        ),
    )
    subcat = subcat.loc[mask, :].copy()

    subcat.sort_values(
        by=["SnapNum", "Group_M_Crit200", "SubGroupNumber"],
        ascending=[False, False, True],
        inplace=True,
    )
    subcat.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Dump to HDF5
    # ------------------------------------------------------------------
    dump_hdf(outpath, subcat)

    if metadata_path is not None:
        with h5py.File(outpath, "a") as subcatfile:
            if "Header" in subcatfile:
                del subcatfile["Header"]
            header = subcatfile.create_group("Header")
            header.attrs["metadata"] = metadata_path
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    return subcat
