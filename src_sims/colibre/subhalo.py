import os
import h5py
import numpy as np
import pandas as pd

from hydroflow.run.tools_catalogue import dump_hdf
from hydroflow.run.initialise import load_metadata

from swiftsimio import load as swiftsimio_loader


def extract_subhaloes(path, mcut=1e10, metadata=None, flowrates=True):
    """
    Build a HYDROFLOW-style subhalo catalogue from COLIBRE SOAP outputs using swiftsimio.

    Parameters
    ----------
    path : str or list[str]
        Path(s) to SOAP catalogue file(s).
    mcut : float, optional
        Minimum host halo mass cut applied using Group_M_Crit200 (Msun).
    metadata : str or object, optional
        Metadata pickle path or loaded metadata object.
    flowrates : bool, optional
        If True, add SOAP spherical overdensity mass flow rates (when available).

    Returns
    -------
    subcat : pandas.DataFrame
        Subhalo catalogue sorted by (SnapNum desc, Group_M_Crit200 desc, SubGroupNumber asc),
        written to ./catalogues/subhaloes.hdf5
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
    outdir = os.path.join(os.getcwd(), "catalogues")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "subhaloes.hdf5")

    # ------------------------------------------------------------------
    # Units (used by swiftsimio conversions)
    # ------------------------------------------------------------------
    munit = "Msun"
    dunit = "Mpc"
    vunit = "km/s"

    # ------------------------------------------------------------------
    # Accumulate per-snapshot DataFrames
    # ------------------------------------------------------------------
    subcat_parts = []

    # ------------------------------------------------------------------
    # Loop over SOAP catalogues
    # ------------------------------------------------------------------
    for ipath in path:
        if not os.path.exists(ipath):
            print(f"Path {ipath} does not exist. Skipping...")
            continue

        print(f"Reading subhalo catalogue from {ipath}...")
        halodata = swiftsimio_loader(ipath)

        # Detect whether subfind-style group/subgroup numbering exists
        try:
            _ = halodata.input_halos_subfind.sub_group_number
            subfind = True
        except Exception:
            subfind = False

        # ------------------------------------------------------------------
        # Snapshot scalars
        # ------------------------------------------------------------------
        redshift = float(halodata.metadata.redshift)
        snapnum = int(halodata.metadata.filename.split("/")[-1].split("_")[-1].split(".")[0])

        # ------------------------------------------------------------------
        # Row count
        # ------------------------------------------------------------------
        numhaloes = int(halodata.input_halos.halo_catalogue_index.shape[0])

        # ------------------------------------------------------------------
        # Base table
        # ------------------------------------------------------------------
        out = pd.DataFrame()
        out["Redshift"] = np.full(numhaloes, redshift, dtype=np.float64)
        out["SnapNum"] = np.full(numhaloes, snapnum, dtype=np.int32)

        # IDs / central flag -> SubGroupNumber
        out["HaloCatalogueIndex"] = halodata.input_halos.halo_catalogue_index.value

        central_flag = halodata.input_halos.is_central.value  # 1 for central, 0 for non-central
        sgn = np.zeros(numhaloes, dtype=np.int16)
        sgn[central_flag == 0] = 1
        out["SubGroupNumber"] = sgn

        # Group / galaxy IDs
        if subfind:
            out["GroupNumber"] = halodata.input_halos_subfind.group_number.value
            out["GalaxyID"] = out["GroupNumber"].to_numpy(dtype=np.float64) * 1e12 + out["SubGroupNumber"].to_numpy(dtype=np.float64)
        else:
            out["GroupNumber"] = np.arange(numhaloes, dtype=np.int64)
            out["HostHaloID"] = halodata.soap.host_halo_index.value
            out["GalaxyID"] = halodata.input_halos_hbtplus.track_id.value
            out["GalaxyID_unique"] = np.int64(snapnum * 1e12) + out["GalaxyID"].to_numpy(dtype=np.int64)
            out["DescendantID"] = halodata.input_halos_hbtplus.descendant_track_id
            out["ParentID"] = halodata.input_halos_hbtplus.nested_parent_track_id
            out["SubhaloRank"] = halodata.soap.subhalo_rank_by_bound_mass.value

        # ------------------------------------------------------------------
        # Host halo properties
        # ------------------------------------------------------------------
        if not subfind:
            mfof = halodata.input_halos_fof.masses
            mfof.convert_to_units(munit)
            out["GroupMass"] = np.asarray(mfof.value)

        for od_str, od_data in zip(
            ["Crit200", "Crit500"],
            [halodata.spherical_overdensity_200_crit, halodata.spherical_overdensity_500_crit],
        ):
            mod = od_data.total_mass
            mod.convert_to_units(munit)
            out[f"Group_M_{od_str}"] = np.asarray(mod.value)

            rod = od_data.soradius
            rod.convert_to_units(dunit)
            out[f"Group_R_{od_str}"] = np.asarray(rod.value)

        vmax = halodata.bound_subhalo.maximum_circular_velocity
        vmax.convert_to_units(vunit)
        out["Subhalo_V_max"] = np.asarray(vmax.value)

        # ------------------------------------------------------------------
        # Centres
        # ------------------------------------------------------------------
        cop_halo = halodata.exclusive_sphere_30kpc.centre_of_mass
        cop_halo.convert_to_units("Mpc")
        out["CentreOfMass_x"] = np.asarray(cop_halo[:, 0].value)
        out["CentreOfMass_y"] = np.asarray(cop_halo[:, 1].value)
        out["CentreOfMass_z"] = np.asarray(cop_halo[:, 2].value)

        # ------------------------------------------------------------------
        # Subhalo mass
        # ------------------------------------------------------------------
        subhalomass = halodata.bound_subhalo.total_mass
        subhalomass.convert_to_units(munit)
        out["Mass"] = np.asarray(subhalomass.value)

        # ------------------------------------------------------------------
        # Aperture (exclusive sphere 30 kpc) properties
        # ------------------------------------------------------------------
        mstar_30kpc = halodata.exclusive_sphere_30kpc.stellar_mass
        mstar_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-star-m_tot-soapexcl"] = np.asarray(mstar_30kpc.value)

        mstar_30kpc = halodata.inclusive_sphere_30kpc.stellar_mass
        mstar_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-star-m_tot-soapincl"] = np.asarray(mstar_30kpc.value)

        mgas_30kpc = halodata.exclusive_sphere_30kpc.gas_mass
        mgas_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_tot-soapexcl"] = np.asarray(mgas_30kpc.value)

        mgas_30kpc = halodata.inclusive_sphere_30kpc.gas_mass
        mgas_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_tot-soapincl"] = np.asarray(mgas_30kpc.value)

        mHI_30kpc = halodata.exclusive_sphere_30kpc.atomic_hydrogen_mass
        mHI_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_HI-soapexcl"] = np.asarray(mHI_30kpc.value)

        mHI_30kpc = halodata.inclusive_sphere_30kpc.atomic_hydrogen_mass
        mHI_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_HI-soapincl"] = np.asarray(mHI_30kpc.value)

        mH2_30kpc = halodata.exclusive_sphere_30kpc.molecular_hydrogen_mass
        mH2_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_H2-soapexcl"] = np.asarray(mH2_30kpc.value)

        mH2_30kpc = halodata.inclusive_sphere_30kpc.molecular_hydrogen_mass
        mH2_30kpc.convert_to_units(munit)
        out["030pkpc_sphere-gas_all-m_H2-soapincl"] = np.asarray(mH2_30kpc.value)

        sfr_30kpc = halodata.exclusive_sphere_30kpc.star_formation_rate
        sfr_30kpc.convert_to_units(f"{munit}/yr")
        out["030pkpc_sphere-gas_all-SFR-soapexcl"] = np.asarray(sfr_30kpc.value)

        rstar = halodata.exclusive_sphere_30kpc.half_mass_radius_stars
        rstar.convert_to_units(dunit)
        out["030pkpc_sphere-star-r_half-soapexcl"] = np.asarray(rstar.value)

        rgas = halodata.exclusive_sphere_30kpc.half_mass_radius_gas
        rgas.convert_to_units(dunit)
        out["030pkpc_sphere-gas_all-r_half-soapexcl"] = np.asarray(rgas.value)

        out["030pkpc_sphere-star-disk_to_total-soapexcl"] = np.asarray(
            halodata.exclusive_sphere_30kpc.disc_to_total_stellar_mass_fraction
        )
        out["030pkpc_sphere-gas_all-disk_to_total-soapexcl"] = np.asarray(
            halodata.exclusive_sphere_30kpc.disc_to_total_gas_mass_fraction
        )
        out["030pkpc_sphere-star-kappa_corot-soapexcl"] = np.asarray(
            halodata.exclusive_sphere_30kpc.kappa_corot_stars
        )
        out["030pkpc_sphere-gas_all-kappa_corot-soapexcl"] = np.asarray(
            halodata.exclusive_sphere_30kpc.kappa_corot_gas
        )

        if not subfind:
            aveSFR_30kpc = halodata.exclusive_sphere_30kpc.averaged_star_formation_rate
            aveSFR_30kpc.convert_to_units(f"{munit}/yr")
            out["030pkpc_sphere-gas_all-ave_SFR_10Myr-soapexcl"] = np.asarray(aveSFR_30kpc.value[:, 0])
            out["030pkpc_sphere-gas_all-ave_SFR_100Myr-soapexcl"] = np.asarray(aveSFR_30kpc.value[:, 1])

            stellarluminosities = halodata.exclusive_sphere_30kpc.stellar_luminosity
            stellarluminosities.convert_to_units("1")
            for iband, band in enumerate(["u", "g", "r", "i", "z", "Y", "J", "H", "K"]):
                out[f"030pkpc_sphere-star-L_{band}-soapexcl"] = np.asarray(stellarluminosities[:, iband].value)

        angmom = halodata.exclusive_sphere_30kpc.angular_momentum_baryons
        angmom.convert_to_units("Msun*Mpc*km/s")
        angmom.convert_to_physical()
        out["030pkpc_sphere-baryon-L_tot-soapexcl_x"] = np.asarray(angmom.value[:, 0])
        out["030pkpc_sphere-baryon-L_tot-soapexcl_y"] = np.asarray(angmom.value[:, 1])
        out["030pkpc_sphere-baryon-L_tot-soapexcl_z"] = np.asarray(angmom.value[:, 2])

        # ------------------------------------------------------------------
        # Black hole properties
        # ------------------------------------------------------------------
        out["030pkpc_sphere-BH-n_tot-soapexcl"] = np.asarray(
            halodata.exclusive_sphere_30kpc.number_of_black_hole_particles
        )

        mbh_total = halodata.exclusive_sphere_30kpc.most_massive_black_hole_mass
        mbh_total.convert_to_units(munit)
        out["030pkpc_sphere-BH-m_tot-soapexcl"] = np.asarray(mbh_total.value)

        if not subfind:
            bh_aveaccretion = halodata.exclusive_sphere_30kpc.most_massive_black_hole_averaged_accretion_rate
            bh_aveaccretion.convert_to_units(f"{munit}/yr")
            out["030pkpc_sphere-BH-ave_accretion_10Myr-soapexcl"] = np.asarray(bh_aveaccretion.value[:, 0])
            out["030pkpc_sphere-BH-ave_accretion_100Myr-soapexcl"] = np.asarray(bh_aveaccretion.value[:, 1])

            bh_thermal_energy = halodata.exclusive_sphere_30kpc.most_massive_black_hole_injected_thermal_energy
            bh_thermal_energy.convert_to_units("erg")
            out["030pkpc_sphere-BH-thermal_energy_soapexcl"] = np.asarray(bh_thermal_energy.value)

            bh_accreted_mass = halodata.exclusive_sphere_30kpc.most_massive_black_hole_total_accreted_mass
            bh_accreted_mass.convert_to_units(munit)
            out["030pkpc_sphere-BH-accreted_mass_soapexcl"] = np.asarray(bh_accreted_mass.value)

        if hasattr(halodata.exclusive_sphere_30kpc, "most_massive_black_hole_injected_jet_energy_by_mode"):
            bh_jet_energy_modes = halodata.exclusive_sphere_30kpc.most_massive_black_hole_injected_jet_energy_by_mode
            bh_jet_energy_modes.convert_to_units("erg")
            jem = bh_jet_energy_modes.value
            for imode, mode in enumerate(["thin", "thick", "slim"]):
                out[f"030pkpc_sphere-BH-jet_energy_{mode}_soapexcl"] = np.asarray(jem[:, imode])

        if hasattr(halodata.exclusive_sphere_30kpc, "most_massive_black_hole_accretion_mode"):
            out["030pkpc_sphere-BH-accdisc_mode_soapexcl"] = np.asarray(
                halodata.exclusive_sphere_30kpc.most_massive_black_hole_accretion_mode.value
            )

        if hasattr(halodata.exclusive_sphere_30kpc, "most_massive_black_hole_number_of_mergers"):
            out["030pkpc_sphere-BH-n_mergers-soapexcl"] = np.asarray(
                halodata.exclusive_sphere_30kpc.most_massive_black_hole_number_of_mergers
            )

        # ------------------------------------------------------------------
        # Assign host properties / satellite distances
        # ------------------------------------------------------------------
        satellites = out["SubGroupNumber"].to_numpy() > 0

        if not subfind:
            hosthaloidxs = np.searchsorted(out["GroupNumber"].to_numpy(), out["HostHaloID"].to_numpy()[satellites])

            out.loc[satellites, "GroupMass"] = out["GroupMass"].to_numpy()[hosthaloidxs]
            out.loc[satellites, "Group_M_Crit200"] = out["Group_M_Crit200"].to_numpy()[hosthaloidxs]
            out.loc[satellites, "Group_R_Crit200"] = out["Group_R_Crit200"].to_numpy()[hosthaloidxs]

            dx = out["CentreOfMass_x"].to_numpy()[satellites] - out["CentreOfMass_x"].to_numpy()[hosthaloidxs]
            dy = out["CentreOfMass_y"].to_numpy()[satellites] - out["CentreOfMass_y"].to_numpy()[hosthaloidxs]
            dz = out["CentreOfMass_z"].to_numpy()[satellites] - out["CentreOfMass_z"].to_numpy()[hosthaloidxs]
            out.loc[satellites, "Group_Rrel"] = np.sqrt(dx * dx + dy * dy + dz * dz)

        else:
            is_cen = (out["SubGroupNumber"].to_numpy() == 0)
            is_sat = ~is_cen

            central_cols = [
                "GroupNumber",
                "Group_M_Crit200", "Group_R_Crit200",
                "Group_M_Crit500", "Group_R_Crit500",
                "CentreOfMass_x", "CentreOfMass_y", "CentreOfMass_z",
            ]

            centrals_df = (
                out.loc[is_cen, central_cols]
                .drop_duplicates(subset="GroupNumber", keep="first")
                .rename(columns={
                    "Group_M_Crit200": "Host_Group_M_Crit200",
                    "Group_R_Crit200": "Host_Group_R_Crit200",
                    "Group_M_Crit500": "Host_Group_M_Crit500",
                    "Group_R_Crit500": "Host_Group_R_Crit500",
                    "CentreOfMass_x": "Host_CentreOfMass_x",
                    "CentreOfMass_y": "Host_CentreOfMass_y",
                    "CentreOfMass_z": "Host_CentreOfMass_z",
                })
            )

            out = out.merge(centrals_df, on="GroupNumber", how="left", copy=False)

            for prop in ["Group_M_Crit200", "Group_R_Crit200", "Group_M_Crit500", "Group_R_Crit500"]:
                out.loc[is_sat, prop] = out.loc[is_sat, f"Host_{prop}"].to_numpy()

            dx = out["CentreOfMass_x"].to_numpy() - out["Host_CentreOfMass_x"].to_numpy()
            dy = out["CentreOfMass_y"].to_numpy() - out["Host_CentreOfMass_y"].to_numpy()
            dz = out["CentreOfMass_z"].to_numpy() - out["Host_CentreOfMass_z"].to_numpy()
            out.loc[is_sat, "Group_Rrel"] = np.sqrt(dx[is_sat] ** 2 + dy[is_sat] ** 2 + dz[is_sat] ** 2)

            out.drop(
                columns=[
                    "Host_Group_M_Crit200", "Host_Group_R_Crit200",
                    "Host_Group_M_Crit500", "Host_Group_R_Crit500",
                    "Host_CentreOfMass_x", "Host_CentreOfMass_y", "Host_CentreOfMass_z",
                ],
                inplace=True,
            )

        # ------------------------------------------------------------------
        # Flow rates
        # ------------------------------------------------------------------
        if flowrates:
            scales = ["0p10r200", "0p30r200", "1p00r200"]
            scale_idx = {"0p10r200": 0, "0p30r200": 1, "1p00r200": 2}

            for scale in scales:
                for key, fr in zip(
                    ["cold", "cool", "warm", "hot"],
                    [
                        halodata.spherical_overdensity_200_crit.cold_gas_mass_flow_rate,
                        halodata.spherical_overdensity_200_crit.cool_gas_mass_flow_rate,
                        halodata.spherical_overdensity_200_crit.warm_gas_mass_flow_rate,
                        halodata.spherical_overdensity_200_crit.hot_gas_mass_flow_rate,
                    ],
                ):
                    fr.convert_to_units(f"{munit}/Gyr")
                    frv = fr.value
                    for iflow, flowtype in enumerate(
                        ["mdot_tot_inflow_vbpseudo_vc000kmps", "mdot_tot_outflow_vbpseudo_vc000kmps", "mdot_tot_outflow_vbpseudo_vc0p25vmx"]
                    ):
                        out[f"{scale}_shellp10_full-gas_{key}-{flowtype}-soap"] = frv[:, iflow * 3 + scale_idx[scale]]

                frdm = halodata.spherical_overdensity_200_crit.dark_matter_mass_flow_rate
                frdm.convert_to_units(f"{munit}/Gyr")
                frdmv = frdm.value
                for iflow, flowtype in enumerate(["mdot_tot_inflow_vbpseudo_vc000kmps", "mdot_tot_outflow_vbpseudo_vc000kmps"]):
                    out[f"{scale}_shellp10_full-dm-{flowtype}-soap"] = frdmv[:, iflow * 3 + scale_idx[scale]]

        # ------------------------------------------------------------------
        # Mass cut and bookkeeping
        # ------------------------------------------------------------------
        mask = np.logical_and(
            out["Group_M_Crit200"].to_numpy() >= mcut,
            np.logical_or(
                out["SubGroupNumber"].to_numpy() == 0,
                out["Mass"].to_numpy() > mcut * 10 ** (-0.5),
            ),
        )
        out = out.loc[mask].copy()
        out.reset_index(drop=True, inplace=True)

        subcat_parts.append(out)

    if len(subcat_parts) == 0:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Final concatenation + sorting
    # ------------------------------------------------------------------
    subcat = pd.concat(subcat_parts, ignore_index=True)
    subcat.sort_values(["SnapNum", "Group_M_Crit200", "SubGroupNumber"], ascending=[False, False, True], inplace=True)
    subcat.reset_index(drop=True, inplace=True)

    dump_hdf(outpath, subcat)

    if metadata_path is not None:
        with h5py.File(outpath, "a") as subcatfile:
            if "Header" in subcatfile:
                del subcatfile["Header"]
            header = subcatfile.create_group("Header")
            header.attrs["metadata"] = str(metadata_path)
    else:
        print("No metadata file found. Metadata path not added.")

    return subcat
