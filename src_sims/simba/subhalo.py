import os
import h5py
import numpy as np
import pandas as pd

from hydroflow.run.tools_catalogue import dump_hdf
from hydroflow.run.initialise import load_metadata


def extract_subhaloes(path, mcut=1e10, metadata=None):
    """
    Build a HYDROFLOW-style halo catalogue from CAESAR HDF5 outputs.

    This loader reads halo-level data only (one row per halo) and assigns
    SubGroupNumber = 0 for all entries.

    Parameters
    ----------
    path : str or list[str]
        Path(s) to CAESAR catalogue files.
    mcut : float, optional
        Minimum halo mass cut applied using Group_M_Crit200 (Msun).
    metadata : str or object, optional
        Metadata pickle path or loaded metadata object.

    Returns
    -------
    subcat : pandas.DataFrame
        Halo catalogue sorted by snapshot and halo mass and written to
        ./catalogues/subhaloes.hdf5

    Notes
    ------
    Only ``halo data'' is used here, only centrals included to avoid any non-trivial selection cuts.

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
    # Load metadata
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
        raise RuntimeError("No metadata provided and no .pkl metadata file found.")

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    outpath = os.path.join(os.getcwd(), "catalogues", "subhaloes.hdf5")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # ------------------------------------------------------------------
    # Unit conversions
    # ------------------------------------------------------------------
    mconv = 1.0     # Msun (no h)
    dconv = 1e-3    # kpc -> Mpc (no h)

    # ------------------------------------------------------------------
    # Snapshot numbers
    # ------------------------------------------------------------------
    snapnums = [int(p.split(".hdf5")[0][-3:]) for p in path]

    halo_dfs = []

    # ------------------------------------------------------------------
    # Loop over snapshots
    # ------------------------------------------------------------------
    for fname, snapnum in zip(path, snapnums):
        print(f"Loading snapshot {snapnum}...")

        with h5py.File(fname, mode="r") as caesarfile:
            zval = float(caesarfile["simulation_attributes"].attrs["redshift"])

            group_id = caesarfile["/halo_data/GroupID"][:].astype(np.int64, copy=False)
            numgroups = group_id.shape[0]

            m_tot = caesarfile["/halo_data/dicts/masses.total"][:] * mconv
            m200c = caesarfile["/halo_data/dicts/virial_quantities.m200c"][:] * mconv
            r200c = caesarfile["/halo_data/dicts/virial_quantities.r200c"][:] * dconv
            cop = caesarfile["/halo_data/minpotpos"][:] * dconv

            halo_df = pd.DataFrame({
                "SnapNum": np.full(numgroups, snapnum, dtype=np.int32),
                "Redshift": np.full(numgroups, zval, dtype=np.float64),
                "GroupNumber": group_id,
                "SubGroupNumber": np.zeros(numgroups, dtype=np.int16),
                "GalaxyID": (np.int64(snapnum * 1e12) + group_id),
                "Mass": m_tot,
                "GroupMass": m_tot,
                "Group_M_Crit200": m200c,
                "Group_R_Crit200": r200c,
            })

            halo_df.loc[:, ["CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z"]] = cop

            mask = np.logical_and(
                halo_df["Group_M_Crit200"].to_numpy() >= mcut,
                np.logical_or(
                    halo_df["SubGroupNumber"].to_numpy() == 0,
                    halo_df["Mass"].to_numpy() > mcut * 10 ** (-0.5),
                ),
            )

            halo_df = halo_df.loc[mask].copy()
            halo_df.reset_index(drop=True, inplace=True)

            halo_dfs.append(halo_df)

    # ------------------------------------------------------------------
    # Concatenate snapshots
    # ------------------------------------------------------------------
    if len(halo_dfs) > 1:
        subcat = pd.concat(halo_dfs, ignore_index=True)
    else:
        subcat = halo_dfs[0]

    # ------------------------------------------------------------------
    # Sort catalogue
    # ------------------------------------------------------------------
    subcat.sort_values(
        by=["SnapNum", "Group_M_Crit200", "SubGroupNumber"],
        ascending=[False, False, True],
        inplace=True,
    )
    subcat.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    dump_hdf(outpath, subcat)

    if metadata_path is not None:
        with h5py.File(outpath, "a") as subcatfile:
            if "Header" in subcatfile:
                del subcatfile["Header"]
            header = subcatfile.create_group("Header")
            header.attrs["metadata"] = metadata_path
    else:
        print("No metadata file found. Metadata path not added.")

    return subcat
