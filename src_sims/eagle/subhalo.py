import os
import h5py
import numpy as np
import pandas as pd
import eagleSqlTools as sql

from hydroflow.run.initialise import load_metadata
from hydroflow.run.tools_catalogue import dump_hdf


def extract_subhaloes(simname="RefL0100N1504", snapnums=None, uname=None, pw=None, mcut=1e11, metadata=None):
    """
    Build a HYDROFLOW-style subhalo catalogue from the EAGLE public SQL database (eagleSqlTools).

    Parameters
    ----------
    simname : str
        EAGLE simulation name, e.g. "RefL0100N1504".
    snapnums : list[int]
        Snapshot numbers to query.
    uname : str
        Database username.
    pw : str
        Database password.
    mcut : float, optional
        Minimum host halo mass cut applied using Group_M_Crit200 (Msun).
    metadata : str or object, optional
        Metadata pickle path or loaded metadata object.

    Returns
    -------
    data_pd : pandas.DataFrame
        Subhalo catalogue sorted by (SnapNum desc, Group_M_Crit200 desc, SubGroupNumber asc),
        written to ./catalogues/subhaloes.hdf5
    """

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    if snapnums is None:
        snapnums = []

    if len(snapnums) == 0:
        print("No snapshot numbers given. Exiting...")
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

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------
    outpath = os.path.join(os.getcwd(), "catalogues", "subhaloes.hdf5")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # ------------------------------------------------------------------
    # Unit conversions
    # ------------------------------------------------------------------
    # Group_R_Crit200 is returned in physical kpc; convert to comoving Mpc:
    #   pkpc -> pMpc ( /1e3 ), then to cMpc ( * (1+z) )
    r200_pkpc_to_cMpc = 1e-3

    # R_halfmass30 returned in physical kpc; convert to cMpc
    rhalf_pkpc_to_pMpc = 1e-3

    # ------------------------------------------------------------------
    # SQL connection
    # ------------------------------------------------------------------
    con = sql.connect(f"{uname}", password=f"{pw}")

    # ------------------------------------------------------------------
    # Query templates
    # ------------------------------------------------------------------
    snapnum_strs = [f"Subhalo.SnapNum={int(snapnum)}" for snapnum in snapnums]

    myQueries = [
        f"""
        SELECT
            Subhalo.Redshift as Redshift,
            Subhalo.SnapNum as SnapNum,
            Subhalo.GalaxyID as GalaxyID,
            Subhalo.Mass as Mass,
            Subhalo.SubGroupNumber as SubGroupNumber,
            Subhalo.CentreOfPotential_x as CentreOfMass_x,
            Subhalo.CentreOfPotential_y as CentreOfMass_y,
            Subhalo.CentreOfPotential_z as CentreOfMass_z,
            Subhalo.GasSpin_x as subhalodashgas_alldashL_totdashsubfexcl_x,
            Subhalo.GasSpin_y as subhalodashgas_alldashL_totdashsubfexcl_y,
            Subhalo.GasSpin_z as subhalodashgas_alldashL_totdashsubfexcl_z,
            Subhalo.MassType_Gas as subhalodashgas_alldashm_totdashsubfexcl,
            Subhalo.MassType_Star as subhalodashstardashm_totdashsubfexcl,
            Subhalo.StarFormationRate as subhalodashgas_alldashSFRdashsubfexcl,
            Aperture.Mass_Star as flag030pkpc_spheredashstardashm_totdashsubfexcl,
            Aperture.Mass_Gas as flag030pkpc_spheredashgas_alldashm_totdashsubfexcl,
            Aperture.Mass_BH as flag030pkpc_spheredashbhdashm_totdashsubfexcl,
            Aperture.Mass_DM as flag030pkpc_spheredashdmdashm_totdashsubfexcl,
            Aperture.SFR as flag030pkpc_spheredashgas_alldashSFRdashsubfexcl,
            Sizes.R_halfmass30 as flag030pkpc_spheredashstardashr_halfdashsubfexcl,
            FOF.GroupMass as GroupMass,
            FOF.Group_M_Crit200 as Group_M_Crit200,
            FOF.Group_R_Crit200 as Group_R_Crit200,
            FOF.GroupCentreOfPotential_x as GroupCentreOfMass_x,
            FOF.GroupCentreOfPotential_y as GroupCentreOfMass_y,
            FOF.GroupCentreOfPotential_z as GroupCentreOfMass_z
        FROM
            {simname}_Subhalo as Subhalo,
            {simname}_Aperture as Aperture,
            {simname}_Sizes as Sizes,
            {simname}_FOF as FOF
        WHERE
            {snapnum_str} and
            Subhalo.GroupID = FOF.GroupID and
            Subhalo.SnapNum = FOF.SnapNum and
            Subhalo.GalaxyID = Aperture.GalaxyID and
            Subhalo.GalaxyID = Sizes.GalaxyID and
            Aperture.ApertureSize = 30 and
            FOF.Group_M_Crit200 >= {mcut:2e}
        ORDER BY
            Subhalo.SnapNum desc,
            Subhalo.Mass desc
        """
        for snapnum_str in snapnum_strs
    ]

    # ------------------------------------------------------------------
    # Execute queries and concatenate results
    # ------------------------------------------------------------------
    subcats = []
    for isnap, myQuery in enumerate(myQueries):
        print("Executing query: ", snapnum_strs[isnap])
        data = sql.execute_query(con, myQuery)
        df = pd.DataFrame(data, columns=list(data.dtype.names))
        df.reset_index(drop=True, inplace=True)
        subcats.append(df)

    print("Concatenating subhalo dataframes...")
    data_pd = pd.concat(subcats, ignore_index=True)
    data_pd.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Column name cleanup
    # ------------------------------------------------------------------
    data_pd.columns = data_pd.columns.str.replace("flag", "", regex=False)
    data_pd.columns = data_pd.columns.str.replace("dash", "-", regex=False)

    # ------------------------------------------------------------------
    # Derived quantities and unit conversions
    # ------------------------------------------------------------------
    # Convert GasSpin -> angular momentum vector by multiplying by gas mass
    # (keeps the same output naming already in use)
    data_pd["subhalo-gas_all-L_tot-subfexcl_x"] = (
        data_pd["subhalo-gas_all-L_tot-subfexcl_x"] * data_pd["subhalo-gas_all-m_tot-subfexcl"]
    )
    data_pd["subhalo-gas_all-L_tot-subfexcl_y"] = (
        data_pd["subhalo-gas_all-L_tot-subfexcl_y"] * data_pd["subhalo-gas_all-m_tot-subfexcl"]
    )
    data_pd["subhalo-gas_all-L_tot-subfexcl_z"] = (
        data_pd["subhalo-gas_all-L_tot-subfexcl_z"] * data_pd["subhalo-gas_all-m_tot-subfexcl"]
    )

    # Group_R_Crit200: pkpc -> cMpc
    data_pd["Group_R_Crit200"] = data_pd["Group_R_Crit200"] * r200_pkpc_to_cMpc * (1.0 + data_pd["Redshift"])

    # 030pkpc_sphere-star-r_half-subfexcl: pkpc -> pMpc
    if "030pkpc_sphere-star-r_half-subfexcl" in data_pd.columns:
        data_pd["030pkpc_sphere-star-r_half-subfexcl"] = (
            data_pd["030pkpc_sphere-star-r_half-subfexcl"] * rhalf_pkpc_to_pMpc * (1.0 + data_pd["Redshift"])
        )

    # Group_Rrel: distance between subhalo and group centre 
    data_pd["Group_Rrel"] = np.sqrt(
        (data_pd["CentreOfMass_x"] - data_pd["GroupCentreOfMass_x"]) ** 2
        + (data_pd["CentreOfMass_y"] - data_pd["GroupCentreOfMass_y"]) ** 2
        + (data_pd["CentreOfMass_z"] - data_pd["GroupCentreOfMass_z"]) ** 2
    )

    # ------------------------------------------------------------------
    # Final selection + sorting
    # ------------------------------------------------------------------
    mask = np.logical_and(
        data_pd["Group_M_Crit200"].to_numpy() >= mcut,
        np.logical_or(
            data_pd["SubGroupNumber"].to_numpy() == 0,
            data_pd["Mass"].to_numpy() > mcut * 10 ** (-0.5),
        ),
    )
    data_pd = data_pd.loc[mask].copy()
    data_pd.reset_index(drop=True, inplace=True)

    data_pd.sort_values(
        by=["SnapNum", "Group_M_Crit200", "SubGroupNumber"],
        ascending=[False, False, True],
        inplace=True,
    )
    data_pd.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    dump_hdf(outpath, data_pd)

    if metadata_path is not None:
        with h5py.File(outpath, "a") as subcatfile:
            if "Header" in subcatfile:
                del subcatfile["Header"]
            header = subcatfile.create_group("Header")
            header.attrs["metadata"] = metadata_path
    else:
        print("No metadata file found. Metadata path not added.")

    return data_pd
