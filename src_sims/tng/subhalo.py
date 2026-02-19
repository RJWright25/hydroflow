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
      2) constructs a "central" halo table (groups) and a "galaxy" table (subhaloes),
      3) attaches host-halo properties to each subhalo,
      4) assigns SubGroupNumber within each FoF group (0 = central),
      5) computes subhalo distance to the host group centre (Group_Rrel),
      6) applies mass cuts and writes an HDF5 catalogue.

    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    path : str or list[str]
        One or more paths pointing to group catalogue directories/files that contain "groups_XXX".
        Example element: ".../output/groups_099/" or ".../output/groups_099/groups_099.0.hdf5".
        The function extracts snapnum from the substring after "groups_".

    mcut : float, optional
        Minimum host-halo mass cut applied to FoF groups and downstream subhaloes (Msun).

    metadata : str or object, optional
        If a string: path to the HYDROFLOW metadata pickle.
        If None: the function searches the current working directory for a ".pkl" metadata file.

    ---------------------------------------------------------------------------
    Returns
    ---------------------------------------------------------------------------
    subcat : pandas.DataFrame
        subhalo catalogue, sorted by (SnapNum desc, Group_M_Crit200 desc, SubGroupNumber asc),
        and written to "./catalogues/subhaloes.hdf5".

    ---------------------------------------------------------------------------
    Notes on units 
    ---------------------------------------------------------------------------
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
    # TNG masses: 1e10 Msun/h  ->  Msun
    mconv = 1e10 / hval

    # TNG lengths: ckpc/h  ->  cMpc
    # (ckpc/h) * (1e-3 Mpc/kpc) / h
    dconv = 1e-3 / hval

    # ------------------------------------------------------------------
    # Derive snapshot numbers from input paths and fetch scale factors from metadata
    # ------------------------------------------------------------------
    snapnums, afacs = [], []
    for ipath in path:
        # Be robust to ".../groups_099/" and ".../groups_099/groups_099.0.hdf5"
        # We just find the first 3 digits after 'groups_'.
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

    # Base path for illustris_python loader (strip anything after "/groups")
    basepath = path[0].split("/groups")[0]

    # Storage across snapshots
    subhalo_dfs = []

    # ------------------------------------------------------------------
    # Iterate over snapshots
    # ------------------------------------------------------------------
    for isnap, snapnum in enumerate(snapnums):
        print(f"Loading snapshot {snapnum}...")

        subfind_raw = tng_tools.groupcat.load(basepath, snapNum=snapnum)
        groupcat = subfind_raw["halos"]
        subcat_raw = subfind_raw["subhalos"]

        # Redshift from metadata
        afac = afacs[isnap]
        zval = (1.0 / afac) - 1.0

        # ------------------------------------------------------------------
        # Build FoF group dataframe (centrals live here)
        # ------------------------------------------------------------------
        print("Extracting group data...")
        numgroups = groupcat["GroupMass"].shape[0]

        group_df = pd.DataFrame({
            "SnapNum": np.full(numgroups, snapnum, dtype=np.int32),
            "Redshift": np.full(numgroups, zval, dtype=np.float64),
            # Use integer group indices 0..N-1 as the canonical GroupNumber
            "GroupNumber": np.arange(numgroups, dtype=np.int64),
            # For a "group central record", define SubGroupNumber=0
            "SubGroupNumber": np.zeros(numgroups, dtype=np.int16),

            # Host halo masses/radii
            "GroupMass": groupcat["GroupMass"][:] * mconv,
            "Group_M_Crit200": groupcat["Group_M_Crit200"][:] * mconv,
            "Group_R_Crit200": groupcat["Group_R_Crit200"][:] * dconv,
        })

        # Group centre (comoving Mpc)
        gpos = groupcat["GroupPos"][:] * dconv
        group_df.loc[:, ["CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z"]] = gpos

        # Apply host mass cut at the group level up-front
        group_df.sort_values("GroupNumber", inplace=True)
        group_df = group_df.loc[group_df["GroupMass"].to_numpy() >= mcut, :].copy()
        group_df.reset_index(drop=True, inplace=True)

        # For fast mapping: we will searchsorted on GroupNumber, so keep it sorted
        group_numbers_sorted = group_df["GroupNumber"].to_numpy()

        # ------------------------------------------------------------------
        # Build subhalo dataframe (one row per subhalo / galaxy)
        # ------------------------------------------------------------------
        print("Extracting subhalo data...")
        numsub = subcat_raw["SubhaloMass"].shape[0]

        subhalo_df = pd.DataFrame({
            "SnapNum": np.full(numsub, snapnum, dtype=np.int32),
            "Redshift": np.full(numsub, zval, dtype=np.float64),

            # FoF group membership for each subhalo
            "GroupNumber": subcat_raw["SubhaloGrNr"][:].astype(np.int64, copy=False),

            # A per-snapshot index 
            "GalaxyID": np.arange(numsub, dtype=np.int64),

            # Subhalo properties
            "StarFormationRate": subcat_raw["SubhaloSFR"][:],                 # Msun/yr (already)
            "StellarMass": subcat_raw["SubhaloMassType"][:, 4] * mconv,       # 1e10 Msun/h -> Msun
            "Mass": subcat_raw["SubhaloMass"][:] * mconv,                     # 1e10 Msun/h -> Msun
        })

        # Subhalo centre
        spos = subcat_raw["SubhaloPos"][:] * dconv
        subhalo_df.loc[:, ["CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z"]] = spos

        # ------------------------------------------------------------------
        # Initialise host-halo columns on the subhalo table (filled by mapping)
        # ------------------------------------------------------------------
        # SubGroupNumber = rank within FoF group, with 0 treated as central
        subhalo_df["SubGroupNumber"] = np.full(numsub, -1, dtype=np.int32)

        # Host halo properties copied down to subhaloes
        subhalo_df["GroupMass"] = np.nan
        subhalo_df["Group_M_Crit200"] = np.nan
        subhalo_df["Group_R_Crit200"] = np.nan
        subhalo_df["Group_Rrel"] = np.nan  # distance from group centre (cMpc)

        # Also store group centre as separate columns (keeps old naming style)
        subhalo_df["Group_CentreOfMass_x"] = np.nan
        subhalo_df["Group_CentreOfMass_y"] = np.nan
        subhalo_df["Group_CentreOfMass_z"] = np.nan

        # ------------------------------------------------------------------
        # Apply host mass cut by dropping subhaloes whose host group is not in group_df
        # ------------------------------------------------------------------
        # This is *much* faster than trying to apply a meaningless "Group_M_Crit200" filter before it's filled.
        host_groups = subhalo_df["GroupNumber"].to_numpy()
        # Membership test against the surviving host group list
        keep = np.isin(host_groups, group_numbers_sorted)
        subhalo_df = subhalo_df.loc[keep, :].copy()
        subhalo_df.reset_index(drop=True, inplace=True)

        # ------------------------------------------------------------------
        # Match group properties to subhaloes + assign SubGroupNumber
        # ------------------------------------------------------------------
        print("Matching group data to subhalo data...")

        # Sort subhaloes by GroupNumber so we can slice contiguous blocks quickly
        subhalo_df.sort_values("GroupNumber", inplace=True)
        subhalo_df.reset_index(drop=True, inplace=True)

        sub_groups = subhalo_df["GroupNumber"].to_numpy()

        # Precompute for fast slicing: boundaries where GroupNumber changes
        # Example: group boundaries indices [0, i1, i2, ..., N]
        change = np.flatnonzero(np.diff(sub_groups) != 0) + 1
        bounds = np.concatenate(([0], change, [subhalo_df.shape[0]]))

        # Loop over unique groups present in the filtered subhalo_df
        # (loop count = number of host haloes, typically far smaller than number of subhaloes)
        unique_groups = sub_groups[bounds[:-1]]

        for ig, gnum in enumerate(unique_groups):
            if ig % 1000 == 0:
                print(f"Group {ig+1}/{unique_groups.shape[0]}...")

            # Find matching row in group_df via searchsorted (group_df is sorted by GroupNumber)
            gidx = np.searchsorted(group_numbers_sorted, gnum)
            if gidx >= group_numbers_sorted.size or group_numbers_sorted[gidx] != gnum:
                # Should not happen due to isin filter, but keep for safety
                continue

            # Slice subhalo rows belonging to this FoF group
            i1 = bounds[ig]
            i2 = bounds[ig + 1]
            n_in_group = i2 - i1
            if n_in_group <= 0:
                continue

            # Assign SubGroupNumber within group: 0..(n_in_group-1)
            subhalo_df.loc[i1:i2-1, "SubGroupNumber"] = np.arange(n_in_group, dtype=np.int32)

            # Copy host halo properties down to these subhaloes
            subhalo_df.loc[i1:i2-1, "GroupMass"] = group_df.loc[gidx, "GroupMass"]
            subhalo_df.loc[i1:i2-1, "Group_M_Crit200"] = group_df.loc[gidx, "Group_M_Crit200"]
            subhalo_df.loc[i1:i2-1, "Group_R_Crit200"] = group_df.loc[gidx, "Group_R_Crit200"]

            gx, gy, gz = group_df.loc[gidx, ["CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z"]].to_numpy()
            subhalo_df.loc[i1:i2-1, "Group_CentreOfMass_x"] = gx
            subhalo_df.loc[i1:i2-1, "Group_CentreOfMass_y"] = gy
            subhalo_df.loc[i1:i2-1, "Group_CentreOfMass_z"] = gz

            # Compute distance to host group centre (cMpc)
            cop_sub = subhalo_df.loc[i1:i2-1, ["CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z"]].to_numpy()
            dr = cop_sub - np.array([gx, gy, gz])[None, :]
            subhalo_df.loc[i1:i2-1, "Group_Rrel"] = np.sqrt(np.sum(dr * dr, axis=1))

        # ------------------------------------------------------------------
        # Append this snapshot's subhalo table
        # ------------------------------------------------------------------
        subhalo_dfs.append(subhalo_df)
        print()

    # ------------------------------------------------------------------
    # Concatenate across snapshots
    # ------------------------------------------------------------------
    if len(subhalo_dfs) > 1:
        subcat = pd.concat(subhalo_dfs, ignore_index=True)
    else:
        subcat = subhalo_dfs[0]

    # ------------------------------------------------------------------
    # Final selection + sorting
    # ------------------------------------------------------------------
    # host mass cut AND (central OR sufficiently massive satellite)
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

    # Add metadata path into output HDF5
    if metadata_path is not None:
        with h5py.File(outpath, "a") as subcatfile:
            if "Header" in subcatfile:
                del subcatfile["Header"]
            header = subcatfile.create_group("Header")
            header.attrs["metadata"] = metadata_path
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    return subcat
