import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import (
    constant_G,
    compute_cylindrical_ztheta,
    calc_halfmass_radius,
    weighted_nanpercentile)

from hydroflow.src_physics.gasflow import calculate_flow_rate


# --------------------------------------------------------------------------------------
# 1) Retrieve particle candidates for a single galaxy
# --------------------------------------------------------------------------------------
def retrieve_galaxy_candidates(galaxy, pdata_subvol, kdtree_subvol, maxrad=None, boxsize=None):
    """
    Retrieve the baryonic / DM particle candidates for a galaxy within a given radius,
    and compute positions / velocities relative to a baryonic centre of mass.

    Parameters
    ----------
    galaxy : dict or pd.Series
        Properties of the galaxy, including:
        - CentreOfMass_x/y/z (comoving Mpc)
        - Redshift
        - Group_R_Crit200 (comoving Mpc) – used if maxrad is None

    pdata_subvol : pd.DataFrame
        Particle data for the relevant subvolume. Must contain at least:
        - Coordinates_x/y/z (comoving Mpc)
        - Velocities_x/y/z (km/s)
        - Masses
        - ParticleType

    kdtree_subvol : scipy.spatial.cKDTree
        KDTree built from the particle coordinates in `pdata_subvol`,
        used to efficiently find particles within `maxrad` of the galaxy.

    maxrad : float, optional
        Maximum *comoving* radius to search around the galaxy (in Mpc).
        If None, defaults to 3.5 * Group_R_Crit200.

    boxsize : float, optional
        Comoving box size (Mpc). If provided, periodic wrapping is applied
        when the search sphere intersects a box edge.

    Returns
    -------
    pdata_candidates : pd.DataFrame
        Subset of `pdata_subvol` containing particles within `maxrad` of
        the galaxy. Extra columns are added:
        - Relative_x/y/z_comoving (Mpc, relative to baryonic COM)
        - Relative_r_comoving (Mpc, 3D radius from baryonic COM)
        - Relative_vx/vy/vz_pec (km/s, relative peculiar velocity)
        - Relative_vrad_pec (km/s, radial peculiar velocity)
        - Membership (0=central, 1=satellite, -1=unbound) IF HaloCatalogueIndex present
        Sorted by Relative_r_comoving (ascending).

        If *no* particles are found, returns an empty DataFrame.
    """

    # ------------------------------------------------------------------
    # 1. Galaxy centre and scale factor
    # ------------------------------------------------------------------
    # COM from catalogue (comoving Mpc)
    com = np.array([galaxy[f"CentreOfMass_{ax}"] for ax in "xyz"], dtype=float)

    # Cosmological scale factor a = 1 / (1 + z)
    afac = 1.0 / (1.0 + galaxy["Redshift"])

    # ------------------------------------------------------------------
    # 2. Determine maximum radius (comoving)
    # ------------------------------------------------------------------
    # If no explicit maxrad is passed, use 3.5 * R200,crit (comoving Mpc)
    if maxrad is None:
        maxrad = galaxy["Group_R_Crit200"] * 3.5

    # ------------------------------------------------------------------
    # 3. Query KDTree for all particles within maxrad of the catalogue COM
    # ------------------------------------------------------------------
    pidx_candidates = kdtree_subvol.query_ball_point(com, maxrad)

    # Subset the particle DataFrame to those indices
    pdata_candidates = pdata_subvol.loc[pidx_candidates, :].copy()
    pdata_candidates.reset_index(drop=True, inplace=True)

    num_candidates = pdata_candidates.shape[0]

    if num_candidates == 0:
        # No particles within the search radius
        return pd.DataFrame()

 
    # ------------------------------------------------------------------
    # 4. Compute relative radii (using catalogue COM, comoving coordinates)
    # ------------------------------------------------------------------
    coords = pdata_candidates[[f"Coordinates_{ax}" for ax in "xyz"]].values
    rel_pos_cat = coords - com[np.newaxis, :]
    # Minimal-image displacement from catalogue COM in *comoving* Mpc
    if boxsize is not None:
        rel_pos_cat -= boxsize * np.round(rel_pos_cat / boxsize)
    # Comoving radii (Mpc) from catalogue COM
    radii_relative = np.linalg.norm(rel_pos_cat, axis=1) 
    
	# ------------------------------------------------------------------
    # 5. Membership classification (central / satellite / unbound)
    # ------------------------------------------------------------------
    membership_present= "HaloCatalogueIndex" in pdata_candidates.columns
    pdata_candidates["Membership"] = np.zeros(pdata_candidates.shape[0]) - 1.0
    if membership_present:
        particle_haloidx = pdata_candidates["HaloCatalogueIndex"].values.astype(float)
        # Initially: mark unbound particles (-1)
        pdata_candidates.loc[particle_haloidx < 0, "Membership"] = -1
        # Bound to this halo
        pdata_candidates.loc[np.abs(particle_haloidx - float(galaxy["HaloCatalogueIndex"])) < 0.1,"Membership"] = 0
        # Satellites (bound but not to this halo)
        pdata_candidates.loc[np.logical_and(particle_haloidx >= 0,np.abs(particle_haloidx - float(galaxy["HaloCatalogueIndex"])) >= 0.1,),"Membership"] = 1

    # ------------------------------------------------------------------
    # 6. Compute baryonic COM and VCOM for recentering
    # ------------------------------------------------------------------

    particle_type = pdata_candidates["ParticleType"].values
    mass = pdata_candidates["Masses"].values
    vxyz = pdata_candidates[[f"Velocities_{ax}" for ax in "xyz"]].values

    # Iteratively find the baryonic centre of mass and velocity
    com_ref = com.copy()  # start from catalogue centre
    L = boxsize

    print(f'COM ref: {com_ref} Mpc')
    recentering_spheres = [30., 10.] #ckpc
    for radius in recentering_spheres:
        # mask in Mpc (scale is ckpc)
        mask = (radii_relative) < radius/1e3

        # Impose membership if present
        if membership_present:
            mask = np.logical_and(mask, pdata_candidates["Membership"].values == 0)

        # Baryons mask (everything except DM=1)
        baryons = (particle_type != 1)

        # If enough baryons *in the current selection*, use them only
        if np.nansum(mask & baryons) > 10:
            mask = np.logical_and(mask, baryons)

        if np.nansum(mask) == 0:
            continue

        rel = coords - com_ref[None, :]
        rel -= L * np.round(rel / L)
        msel = mass[mask]
        rel_sel = rel[mask]

        # COM is reference centre plus mass-weighted mean of wrapped offsets
        com_updated = com_ref + (np.nansum(msel[:, None] * rel_sel, axis=0) / np.nansum(msel))

        # Update radii for next iteration (still using minimal image about new centre)
        rel_pos_updated = coords - com_updated[None, :]
        rel_pos_updated -= L * np.round(rel_pos_updated / L)
        radii_relative = np.linalg.norm(rel_pos_updated, axis=1) 

        # Move reference centre forward (keeps offsets small + stable)
        com_ref = com_updated

    # Use 10pkpc as the final scale
    mask_final = radii_relative < recentering_spheres[-1]/1e3
    if membership_present:
        mask_final = np.logical_and(mask_final, pdata_candidates["Membership"].values == 0)

    # Periodic-safe final COM (reference + mean wrapped offsets)
    rel = coords - com_ref[None, :]
    rel -= L * np.round(rel / L)

    msel = mass[mask_final]
    rel_sel = rel[mask_final]

    com_final = com_ref + (np.nansum(msel[:, None] * rel_sel, axis=0) / np.nansum(msel))
    vcom_final = (np.nansum(msel[:, None] * vxyz[mask_final], axis=0) / np.nansum(msel))

    com_offset=np.linalg.norm(com_final-com)
    print(f"COM final: {com_final}")
    print(f"Offset: {com_offset*1e3} ckpc")

    # ------------------------------------------------------------------
    # 7. Recompute relative positions / velocities using iterative COM
    # ------------------------------------------------------------------
    rel_pos = coords - com_final[np.newaxis, :]
    rel_pos -= boxsize * np.round(rel_pos_updated / boxsize)
    rel_r = np.linalg.norm(rel_pos, axis=1)  # still comoving Mpc

    pdata_candidates[[f"Relative_{ax}_comoving" for ax in "xyz"]] = rel_pos
    pdata_candidates["Relative_r_comoving"] = rel_r

    rel_v = vxyz - vcom_final[np.newaxis, :]
    pdata_candidates[[f"Relative_v{ax}_pec" for ax in "xyz"]] = rel_v

    # Radial unit vector r̂ = r / |r|
    # stack radii three times to match shape (N, 3)
    rhat = rel_pos / np.stack(3 * [rel_r], axis=1)

    # Radial component of peculiar velocity
    pdata_candidates["Relative_vrad_pec"] = np.sum(rel_v * rhat, axis=1)

    # ------------------------------------------------------------------
    # 8. Sort candidates by increasing radius
    # ------------------------------------------------------------------
    pdata_candidates.sort_values(by="Relative_r_comoving", inplace=True)
    pdata_candidates.reset_index(drop=True, inplace=True)

    return pdata_candidates


# --------------------------------------------------------------------------------------
# 2) Main analysis routine for a single galaxy
# --------------------------------------------------------------------------------------
def analyse_galaxy(
    galaxy,
    pdata_candidates,
    metadata,
    r200_shells=[0.1, 0.3, 1],
    kpc_shells=[10, 30, 100],
    rstar_shells=[1, 2, 4],
    zslab_radii={"rmx2reff": "2reff", "rmx10pkpc": 10, "rmxzheight": 1},
    Tbins={"cold": [0, 1e3], "cool": [1e3, 1e5], "warm": [1e5, 1e7], "hot": [1e7, 1e15]},
    theta_bins={"full": [0, 90], "minax": [60, 90], "majax": [0, 30]},
    vcuts={"vc0p25vmx": "0.25Vmax", "vc1p00vmx": "1.00Vmax", "vc050kmps": 50, "vc250kmps": 250},
    drfacs=[0.1],
    dzfacs=[0.4],
    logfile=None,
):
    """
    Analyse a galaxy and its surrounding baryonic / DM reservoirs.

    This function computes:
      - Spherical enclosed masses, metallicities, SFRs etc. within several radii
      - Spherical shell properties (mass, temperature, metallicity)
      - Mass flow rates (inflow / outflow) for DM, stars and gas
      - Phase- and species-resolved gas properties
      - Vertical slab (cylindrical) flow rates above/below the disk

    All outputs are stored in a flat dictionary whose keys encode geometry,
    tracer, phase, species, velocity cuts etc.


    Parameters
    ----------
    galaxy : dict or pd.Series
        Galaxy properties. Must include:
        - Group_M_Crit200, Group_R_Crit200
        - SubGroupNumber
        - Redshift
        - possibly Subhalo_V_max (for Vmax)

    pdata_candidates : pd.DataFrame
        Output from `retrieve_galaxy_candidates`. Must include:
        - Relative_r_comoving, Relative_vrad_pec
        - Relative_x/y/z_comoving, Relative_vx/vy/vz_pec
        - Temperature, StarFormationRate, Metallicity
        - Masses, ParticleType
        - and optional species mass fraction columns: 'mfrac_<spec>'

    metadata : object
        Must provide `metadata.cosmology` with methods:
        - Ogamma(z), Om(z), H(z).value

    r200_shells, kpc_shells, rstar_shells : lists
        Radii at which to compute spherical/shell properties, expressed as:
        - multiples of R200,crit (comoving)
        - physical kpc (pkpc)
        - multiples of the stellar half-mass radius

    zslab_radii : dict
        Radii for cylindrical slabs in the disk plane:
        - numeric values are in pkpc (physical kpc)
        - strings '2reff', 'zheight' are interpreted later as multiples of the stellar reff OR the z slab height

    Tbins : dict
        Gas temperature phase bins, mapping name -> [T_min, T_max] (Kelvin).

    theta_bins : dict
        Polar angle ranges (deg) for separating outflows relative to the
        angular momentum vector (e.g. 'minax', 'full').

    vcuts : dict
        Velocity thresholds (km/s or fractions of Vmax) for defining outflows.

    drfacs : list
        Fractional shell widths (Δr = drfac * r). Usually small (e.g. 0.1).
    
    dzfacs : list
        Fractional zslab shell widths (Δz = dzfac * z). Usually small (e.g. 0.1).

    logfile : str or None
        Currently unused, kept for compatibility.

    Returns
    -------
    galaxy_output : dict
        
        Flat dictionary of all computed quantities. Keys are unchanged
        relative to the original implementation.
        
    """

    # ------------------------------------------------------------------
    # 0. Initialise output with all galaxy catalogue properties
    # ------------------------------------------------------------------
    galaxy_output = {}
    for key in galaxy.keys():
        galaxy_output[key] = galaxy[key]

    # ------------------------------------------------------------------
    # 1. Cosmology and pseudo-evolution velocity (vpseudo)
    # ------------------------------------------------------------------
    # Cosmology at this redshift
    z = galaxy["Redshift"]
    omegag = metadata.cosmology.Ogamma(z)
    omegam = metadata.cosmology.Om(z)
    Hz = metadata.cosmology.H(z).value  # Hubble rate [km/s/Mpc]
    afac = 1.0 / (1.0 + z)              # scale factor

    # Pseudo-evolution velocity
    # Using Group_M_Crit200 in Msun, Group_R_Crit200 in Mpc, G in suitable units.
    M200 = galaxy["Group_M_Crit200"]

    vpseudo=(2 / 3) * (constant_G * M200 * Hz / 100) ** (1 / 3)
    vpseudo *= (2 *omegag + (3 / 2) * omegam)
    # print(f"z={z}, Hz={Hz} km/s, omegag={omegag}, omegam={omegam}")
    # print(f"vpseudo= {vpseudo} km/s")

    # R_dot = (2 / 3) * (G * self.SO_mass * self.cosmology["H"] / 100) ** (
    #                 1 / 3
    #             )
    #             R_dot *= (
    #                 2 * self.cosmology["Omega_g"] + (3 / 2) * self.cosmology["Omega_m"]
    #             )
    #             R_dot *= R_frac
 


    galaxy_output["1p00r200-vpdoev"] = vpseudo  # km/s

    # ------------------------------------------------------------------
    # 2. Save baryonic COM position (from closest particle)
    # ------------------------------------------------------------------
    # The "COM" used for the 30 pkpc sphere is stored from particle [0]
    # (since pdata_candidates is sorted by Relative_r_comoving).
    for i_dim, dim in enumerate(["x", "y", "z"]):
        galaxy_output[f"hydroflow-com_{dim}"] = pdata_candidates.loc[
            0, f"Coordinates_{dim}"
        ]

    # ------------------------------------------------------------------
    # 3. Shell-width bookkeeping (drfacs in frac, string labels)
    # ------------------------------------------------------------------
    drfacs_pc = [drfac * 100.0 for drfac in drfacs]  # convert 0.1 Mpc => 10 pc style scaling
    drfacs_str = ["p" + f"{val:.0f}".zfill(2) for val in drfacs_pc]

    dzfacs_pc=[dzfac * 100.0 for dzfac in dzfacs]
    dzfacs_str=["p" + f"{val:.0f}".zfill(2) for val in dzfacs_pc]

    # ------------------------------------------------------------------
    # 4. Cylindrical coordinates and disk orientation
    # ------------------------------------------------------------------
    # Compute:
    #   Lbar   : total baryonic angular momentum vector
    #   thetapos : polar angle of positions relative to Lbar
    #   thetavel : polar angle of velocities relative to Lbar
    #   zheight  : height above the disk plane (Mpc)

    Lbar, thetapos, thetavel, zheight = compute_cylindrical_ztheta(
        pdata=pdata_candidates, afac=afac, baryons=True, aperture=0.03
    )

    pdata_candidates["Relative_theta_pos"] = thetapos
    pdata_candidates["Relative_theta_vel"] = thetavel
    pdata_candidates["Relative_zheight"] = zheight

    for i_dim, dim in enumerate(["x", "y", "z"]):
        galaxy_output[f"030pkpc_sphere-Lbar{dim}"] = Lbar[i_dim]

    # ------------------------------------------------------------------
    # 5. Pre-load particle data into NumPy arrays for speed
    # ------------------------------------------------------------------
    mass = pdata_candidates["Masses"].values
    rrel = pdata_candidates["Relative_r_comoving"].values  # comoving Mpc
    vrad = pdata_candidates["Relative_vrad_pec"].values    # km/s, radial pec vel
    thetapos = pdata_candidates["Relative_theta_pos"].values
    thetavel = pdata_candidates["Relative_theta_vel"].values
    temp = pdata_candidates["Temperature"].values
    sfr = pdata_candidates["StarFormationRate"].values
    vxyz = pdata_candidates[[f"Relative_v{ax}_pec" for ax in "xyz"]].values

    # Velocity component along Lbar (z-axis of disk)
    # vradz: sign convention flipped below the plane
    Lbar_norm = np.linalg.norm(Lbar)
    vradz = np.dot(vxyz, Lbar / Lbar_norm)
    vradz[zheight < 0.0] *= -1.0

    # In-plane radius (projected radius in the disk plane)
    rrel_inplane = np.sqrt(rrel ** 2 - zheight ** 2)

    # Particle type masks
    ptype = pdata_candidates["ParticleType"].values
    gas = ptype == 0.0
    star = ptype == 4.0
    dm = ptype == 1.0

    # ------------------------------------------------------------------
    # 6. Membership classification (central / satellite / unbound)
    # ------------------------------------------------------------------
    # Membership values:
    #   -1 : unbound
    #    0 : bound to the main halo
    #    1 : satellite (bound to another halo)
    
    membership_masks = {"incl": np.ones_like(mass, dtype=bool)}
    if "Membership" in pdata_candidates.columns:
        memberships = pdata_candidates["Membership"].values
        membership_masks["excl"] = memberships <= 0.0
        # "excl" removes satellites (keeps bound + unbound)

    # ------------------------------------------------------------------
    # 7. Temperature-phase masks (gas only)
    # ------------------------------------------------------------------
    Tmasks = {
        # "all" gas
        "all": gas,
        # star-forming gas (gas with sfr > 0)
        "sf": np.logical_and(gas, sfr > 0.0),
    }

    if Tbins is not None:
        # Add named temperature bins (cold/cool/warm/hot etc.)
        for Tstr, Tbin in Tbins.items():
            Tmasks[Tstr] = np.logical_and.reduce(
                [gas, temp > Tbin[0], temp < Tbin[1]]
            ).astype(bool)

    # ------------------------------------------------------------------
    # 8. Species mass arrays (e.g. HI, H2, metals, etc.)
    # ------------------------------------------------------------------
    # Expect columns named 'mfrac_<spec>' which store mass fraction for that spec.
    specmass = {}
    mfrac_columns = [col for col in pdata_candidates.columns if "mfrac" in col]

    for mfrac_col in mfrac_columns:
        spec_name = mfrac_col.split("mfrac_")[1]
        specmass[spec_name] = pdata_candidates[mfrac_col].values * mass

    # Metal mass (Z) and total mass as extra "species"
    specmass["Z"] = pdata_candidates["Metallicity"].values * mass
    specmass["tot"] = np.ones_like(specmass["Z"]) * mass

    # ------------------------------------------------------------------
    # 9. Velocity cuts and Vmax estimate
    # ------------------------------------------------------------------
    # Estimate halo circular velocity
    galaxy_output["Group_V_Crit200"] = np.sqrt(
        constant_G * galaxy["Group_M_Crit200"] / (galaxy["Group_R_Crit200"]*afac)
	)  # km/s

    # Don't use vmax cut if satellite and no vmax output
    if galaxy_output['SubGroupNumber']>0:
        galaxy_output["Group_V_Crit200"] = 0

    vmins = []
    vminstrs = list(vcuts.keys())

    # Vmax: prefer Subhalo_V_max if present, else approximate from V_circ
    if "Subhalo_V_max" in galaxy.keys():
        vmax = galaxy["Subhalo_V_max"]
        # print(f"Using Subhalo_V_max for Vmax: val = {vmax:.2f} km/s")
    elif "Group_V_Crit200" in galaxy_output.keys() and galaxy_output['SubGroupNumber']==0:
        # NFW with c ~ 10 => Vmax ~ 1.33 V_circ,200
        vmax = 1.33 * galaxy_output["Group_V_Crit200"]

    # Convert vcuts (possibly string fractions of Vmax) to km/s
    for vcut_key in vcuts.keys():
        vcut_val = vcuts[vcut_key]
        if isinstance(vcut_val, str) and "Vmax" in vcut_val:
            # e.g. "0.25Vmax"
            factor = float(vcut_val.split("Vmax")[0])
            vcut_val = vmax * factor
        vmins.append(vcut_val)

    # BERNOUILLI VELOCITY CUTS (not currently used; kept for reference)
    # potential_infinity = -constant_G * np.nansum(mass) / (np.nanmax(rrel) * afac)
    # potential_profile = -constant_G * np.cumsum(mass) / (rrel * afac)
    # indices_3r = np.searchsorted(rrel, 3 * rrel); indices_3r = np.clip(indices_3r, 0, len(rrel) - 1)
    # potential_atxrrel = potential_profile[indices_3r]
    # mu = estimate_mu(x_H=ionised_frac_H, T=temp, y=0.08)
    # cs = 0.129 * np.sqrt(temp / mu); gamma = 5 / 3
    # vb_to3r = np.sqrt(2 * (potential_atxrrel - potential_profile) - 2 * cs ** 2 / (gamma - 1))
    # vmins.append(vb_to3r); vminstrs.append('vcbnto3rr')

    # ------------------------------------------------------------------
    # 10. Stellar and gas half-mass radii (spherical and vertical)
    # ------------------------------------------------------------------
    star_r_half = np.nan
    star_rz_half = np.nan
    gas_r_half = np.nan
    gas_rz_half = np.nan

    # Stars within 0.03 Mpc physical (~30 pkpc) for r_half computation -- exclusive if membership present
    star_mask = np.logical_and(star, rrel * afac < 0.03)
    if 'excl' in membership_masks:
        star_mask = np.logical_and(star_mask, pdata_candidates["Membership"].values == 0.0)
        
    if np.nansum(star_mask):
        star_r_half = calc_halfmass_radius(mass[star_mask], rrel[star_mask])
        star_rz_half = calc_halfmass_radius(
            mass[star_mask], np.abs(zheight[star_mask])
        )

    # Gas within 0.03 Mpc physical (~30 pkpc) for r_half computation -- exclusive if membership present
    gas_mask = np.logical_and(gas, rrel * afac < 0.03)
    if 'excl' in membership_masks:
        gas_mask = np.logical_and(gas_mask, pdata_candidates["Membership"].values == 0.0)
    if np.nansum(gas_mask):
        gas_r_half = calc_halfmass_radius(mass[gas_mask], rrel[gas_mask])
        gas_rz_half = calc_halfmass_radius(
            mass[gas_mask], np.abs(zheight[gas_mask])
        )

    galaxy_output["030pkpc_sphere-star-r_half"] = star_r_half
    galaxy_output["030pkpc_sphere-star-rz_half"] = star_rz_half
    galaxy_output["030pkpc_sphere-gas-r_half"] = gas_r_half
    galaxy_output["030pkpc_sphere-gas-rz_half"] = gas_rz_half

    # ------------------------------------------------------------------
    # 11. Theta masks (gas polar angle selections)
    # ------------------------------------------------------------------
    thetamasks = {}
    for theta_str, theta_bin in theta_bins.items():
        if theta_str == "full":
            thetamasks[theta_str] = np.logical_and.reduce([gas])
        else:
            # By position angle
            thetamasks[theta_str + "pos"] = np.logical_and.reduce(
                [gas, thetapos > theta_bin[0], thetapos < theta_bin[1]]
            )
            # # By velocity angle
            # thetamasks[theta_str + "vel"] = np.logical_and.reduce(
            #     [gas, thetavel > theta_bin[0], thetavel < theta_bin[1]]
            # )

    # Disk / non-disk masks for theta selections (currently commented out)
    #
    # nondisc_mask = np.logical_and.reduce([
    #     gas,
    #     np.logical_not(np.logical_and(
    #         np.abs(zheight) < (gas_rz_half * 2),
    #         rrel_inplane < (gas_r_half * 2)
    #     ))
    # ])
    # for theta_str in list(thetamasks.keys()):
    #     thetamasks[theta_str + 'nd'] = np.logical_and.reduce(
    #         [nondisc_mask, thetamasks[theta_str]]
    #     )

    # ------------------------------------------------------------------
    # 12. z-slab maximum radii (co-planar region sizes)
    # ------------------------------------------------------------------
    # rmax is either based on stellar half-mass radii (reff), on absolute
    # zheight, or on a fixed pkpc radius converted to comoving Mpc.
    
    zslab_radii_vals = []
    zslab_radii_strs = list(zslab_radii.keys())

    for zslab_radius in zslab_radii.values():
        if isinstance(zslab_radius, str) and "reff" in zslab_radius:
            # e.g. '2reff' => 2 * r_half of stars, from the 10 pkpc sphere
            zslab_radius_val = (
                galaxy_output["030pkpc_sphere-star-r_half"]
                * float(zslab_radius.split("reff")[0])
            )
        elif isinstance(zslab_radius, str) and "zheight" in zslab_radius:
            # Special case handled later: rmax = rhi in the slab
            zslab_radius_val = "zheight"
        else:
            # Numeric pkpc => convert to comoving Mpc
            zslab_radius_val = zslab_radius / 1e3 / afac

        zslab_radii_vals.append(zslab_radius_val)

    # ------------------------------------------------------------------
    # 13. Define all radial shells (R200, pkpc, stellar reff)
    # ------------------------------------------------------------------
    # Radii in COMOVING Mpc
    radial_shells_R200 = [fR200 * galaxy["Group_R_Crit200"] for fR200 in r200_shells]
    radial_shells_pkpc = [fpkpc / 1e3 / afac for fpkpc in kpc_shells]
    radial_shells_rstar = [fstar * star_r_half for fstar in rstar_shells]

    radial_shells_R200_str = [f"{fR200:.2f}".replace(".", "p") + "r200" for fR200 in r200_shells]
    radial_shells_pkpc_str = [str(int(fpkpc)).zfill(3) + "pkpc" for fpkpc in kpc_shells]
    radial_shells_rstar_str = [f"{fstar:.2f}".replace(".", "p") + "reff" for fstar in rstar_shells]

    radial_shells = radial_shells_R200 + radial_shells_pkpc + radial_shells_rstar
    radial_shells_str = radial_shells_R200_str + radial_shells_pkpc_str + radial_shells_rstar_str

    # ------------------------------------------------------------------
    # 14. Loop over all spherical radii and compute sphere + shell quantities
    # ------------------------------------------------------------------
    for rshell, rshell_str in zip(radial_shells, radial_shells_str):

        # Pseudo-evolution velocity boundary (vbdef):
        # only applied to R200 shells and only for centrals
        if ("r200" in rshell_str) and (galaxy["SubGroupNumber"] == 0):
            vbpseudo = vpseudo * (rshell / galaxy["Group_R_Crit200"])
        else:
            vbpseudo = 0.0

        vsboundary = [vbpseudo,0]
        vsboundary_str = ["vbpseudo",'vbstatic']

        # Skip R200-based shells for satellite galaxies
        if ("r200" in rshell_str) and (galaxy["SubGroupNumber"] > 0):
            continue

        # ------------------------------------------------------------------
        # 14.1 Sphere (r < rshell)
        # ------------------------------------------------------------------
        # rrel is sorted, so we can use searchsorted to find the upper index
        rshell_maxidx = np.searchsorted(rrel, rshell)
        base_sphere_mask = np.zeros_like(rrel, dtype=bool)
        base_sphere_mask[:rshell_maxidx] = True

        # Volume of the sphere in physical pkpc^3
        galaxy_output[f"{rshell_str}_sphere-vol"] = (
            4.0 / 3.0 * np.pi * (rshell * afac * 1e3) ** 3)
        
		# Only 'incl' membership for spheres
        for mem_str, mem_mask in membership_masks.items():
            # Combine geometric sphere selection with membership selection
            mask_sphere = np.logical_and(base_sphere_mask, mem_mask)

            # ---------------- DM inside sphere ----------------
            dm_mask_sphere = np.logical_and(mask_sphere, dm)
            galaxy_output[f"{rshell_str}_sphere-dm-m_tot_{mem_str}"] = np.nansum(
                mass[dm_mask_sphere]
            )
            galaxy_output[f"{rshell_str}_sphere-dm-n_tot_{mem_str}"] = np.nansum(
                dm_mask_sphere
            )

            # ---------------- Stars inside sphere ----------------
            star_mask_sphere = np.logical_and(mask_sphere, star)
            galaxy_output[f"{rshell_str}_sphere-star-m_tot_{mem_str}"] = np.nansum(
                mass[star_mask_sphere]
            )
            galaxy_output[f"{rshell_str}_sphere-star-n_tot_{mem_str}"] = np.nansum(
                star_mask_sphere
            )
            galaxy_output[f"{rshell_str}_sphere-star-Z_{mem_str}"] = (
                np.nansum(specmass["Z"][star_mask_sphere])
                / np.nansum(mass[star_mask_sphere])
            )

            # ---------------- Gas inside sphere ----------------
            for Tstr, Tmask in Tmasks.items():
                Tmask_sphere = np.logical_and.reduce([mask_sphere, gas, Tmask])

                # Total number and mass in this phase
                galaxy_output[
                    f"{rshell_str}_sphere-gas_{Tstr}-n_tot_{mem_str}"
                ] = np.nansum(Tmask_sphere.astype(bool).astype(float))
                galaxy_output[
                    f"{rshell_str}_sphere-gas_{Tstr}-m_tot_{mem_str}"
                ] = np.nansum(mass[Tmask_sphere])

                # Species masses within this phase
                for spec in specmass.keys():
                    galaxy_output[
                        f"{rshell_str}_sphere-gas_{Tstr}-m_{spec}_{mem_str}"
                    ] = np.nansum(specmass[spec][Tmask_sphere])

                # SFR and metallicity in this phase
                galaxy_output[
                    f"{rshell_str}_sphere-gas_{Tstr}-SFR_{mem_str}"
                ] = np.nansum(sfr[Tmask_sphere])
                galaxy_output[
                    f"{rshell_str}_sphere-gas_{Tstr}-Z_{mem_str}"
                ] = np.nansum(specmass["Z"][Tmask_sphere]) / np.nansum(
                    mass[Tmask_sphere]
                )

        # ------------------------------------------------------------------
        # 14.2 Spherical shells (r between r - dr/2 and r + dr/2)
        # ------------------------------------------------------------------
        for drfac, drfac_str in zip(drfacs, drfacs_str):
            # Inner/outer edges in comoving Mpc
            rhi = rshell + (drfac * rshell) / 2.0
            rlo = rshell - (drfac * rshell) / 2.0

            # Indices in sorted radial array
            rshell_minidx = np.searchsorted(rrel, rlo)
            rshell_maxidx = np.searchsorted(rrel, rhi)

            base_shell_mask = np.zeros_like(rrel, dtype=bool)
            base_shell_mask[rshell_minidx:rshell_maxidx] = True

            # Convert shell radii to physical Mpc for volumes/areas
            rhi_phys = rhi * afac
            rlo_phys = rlo * afac
            dr_phys = rhi_phys - rlo_phys

            # Volume and area in pkpc^3 and pkpc^2
            galaxy_output[f"{rshell_str}_shell{drfac_str}_full-vol"] = (
                4.0 / 3.0
                * np.pi
                * ((rhi_phys * 1e3) ** 3 - (rlo_phys * 1e3) ** 3)
            )
            galaxy_output[f"{rshell_str}_shell{drfac_str}_full-area"] = (
                4.0 * np.pi * ((rhi_phys * 1e3) ** 2 - (rlo_phys * 1e3) ** 2)
            )

            for mem_str, mem_mask in membership_masks.items():
                mask_shell = np.logical_and(base_shell_mask, mem_mask)

                # ---------------- DM in shell ----------------
                dm_shell_mask = np.logical_and(mask_shell, dm)
                galaxy_output[
                    f"{rshell_str}_shell{drfac_str}_full-dm-m_tot_{mem_str}"
                ] = np.nansum(mass[dm_shell_mask])
                galaxy_output[
                    f"{rshell_str}_shell{drfac_str}_full-dm-n_tot_{mem_str}"
                ] = np.nansum(dm_shell_mask)

                for vboundary, vkey in zip(vsboundary, vsboundary_str):
                    dm_flow_rates = calculate_flow_rate(
                        masses=mass[dm_shell_mask],
                        vrad=vrad[dm_shell_mask],
                        dr=dr_phys,
                        vboundary=vboundary,
                        vmin=[],
                    )
                    galaxy_output[
                        f"{rshell_str}_shell{drfac_str}_full-dm-mdot_tot_inflow_{vkey}_vc000kmps_{mem_str}"
                    ] = dm_flow_rates[0]
                    galaxy_output[
                        f"{rshell_str}_shell{drfac_str}_full-dm-mdot_tot_outflow_{vkey}_vc000kmps_{mem_str}"
                    ] = dm_flow_rates[1]

                # ---------------- Stars in shell ----------------
                stars_shell_mask = np.logical_and(mask_shell, star)
                galaxy_output[
                    f"{rshell_str}_shell{drfac_str}_full-star-m_tot_{mem_str}"
                ] = np.nansum(mass[stars_shell_mask])
                galaxy_output[
                    f"{rshell_str}_shell{drfac_str}_full-star-n_tot_{mem_str}"
                ] = np.nansum(stars_shell_mask)
                galaxy_output[
                    f"{rshell_str}_shell{drfac_str}_full-star-Z_{mem_str}"
                ] = np.nansum(specmass["Z"][stars_shell_mask]) / np.nansum(
                    mass[stars_shell_mask]
                )

                for vboundary, vkey in zip(vsboundary, vsboundary_str):
                    stars_flow_rates = calculate_flow_rate(
                        masses=mass[stars_shell_mask],
                        vrad=vrad[stars_shell_mask],
                        dr=dr_phys,
                        vboundary=vboundary,
                        vmin=[],
                    )
                    galaxy_output[
                        f"{rshell_str}_shell{drfac_str}_full-star-mdot_tot_inflow_{vkey}_vc000kmps_{mem_str}"
                    ] = stars_flow_rates[0]
                    galaxy_output[
                        f"{rshell_str}_shell{drfac_str}_full-star-mdot_tot_outflow_{vkey}_vc000kmps_{mem_str}"
                    ] = stars_flow_rates[1]

                # ---------------- Gas in shell (phase- and theta-resolved) ----------------
                for theta_str, thetamask in thetamasks.items():
                    mask_shell_theta = np.logical_and(mask_shell, thetamask)

                    for Tstr, Tmask in Tmasks.items():
                        Tmask_shell = np.logical_and.reduce(
                            [mask_shell_theta, gas, Tmask]
                        )

                        # Number of gas elements in this (theta, T) selection
                        galaxy_output[
                            f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-n_tot_{mem_str}"
                        ] = np.nansum(Tmask_shell.astype(bool).astype(float))
                        # Total mass (phase-resolved)
                        galaxy_output[
							f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-m_tot_{mem_str}"
						] = np.nansum(mass[Tmask_shell])
                        
						# Mass-weighted mean temperature (phase-resolved)
                        galaxy_output[
							f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-T_mean_{mem_str}"
						] = np.nansum(
							temp[Tmask_shell] * mass[Tmask_shell]) / np.nansum(mass[Tmask_shell])

                        # Median temperature (phase-resolved)
                        galaxy_output[
                            f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-T_{mem_str}"
                        ] = np.nanmedian(temp[Tmask_shell])

                        # Total SFR and metallicity
                        galaxy_output[
                            f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-SFR_{mem_str}"
                        ] = np.nansum(sfr[Tmask_shell])
                        galaxy_output[
                            f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-Z_{mem_str}"
                        ] = np.nansum(specmass["Z"][Tmask_shell]) / np.nansum(
                            mass[Tmask_shell]
                        )
                        
						# Add outflow velocity statistics
                        outmask = np.logical_and(Tmask_shell, vrad > 0.0)
                        galaxy_output[
							f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_mean_{mem_str}"
						] = np.nanmean(vrad[outmask])
                        galaxy_output[
							f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_50P_{mem_str}"
						] = float(weighted_nanpercentile(
							vrad[outmask], mass[outmask], 50))
                        galaxy_output[
							f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_90P_{mem_str}"
						] = float(weighted_nanpercentile(
							vrad[outmask], mass[outmask], 90))

                        # For Tstr == 'all', also track species masses and vrad stats
                        if Tstr == "all":
                            for spec in specmass.keys():
                                spec_mass_shell = specmass[spec][Tmask_shell]
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-m_{spec}_{mem_str}"
                                ] = np.nansum(spec_mass_shell)
                                
								# Add mass-weighted temperature for that species
                                galaxy_output[
									f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-T_{spec}_mean_{mem_str}"
								] = np.nansum(
									temp[Tmask_shell] * spec_mass_shell
								) / np.nansum(spec_mass_shell)

                                # Mass-weighted mean vrad for that species
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vrad_{spec}_mean_{mem_str}"
                                ] = np.nansum(
                                    vrad[Tmask_shell] * spec_mass_shell
                                ) / np.nansum(spec_mass_shell)

                                # Outflow-only selection for that species - for velocity stats
                                outmask = np.logical_and(Tmask_shell, vrad > 0.0)
                                spec_mass_out = specmass[spec][outmask]
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_{spec}_mean_{mem_str}"
                                ] = np.nansum(
                                    vrad[outmask] * spec_mass_out
                                ) / np.nansum(spec_mass_out)

                                # Mass-weighted percentiles of vrad for outflowing gas
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_{spec}_50P_{mem_str}"
                                ] = float(
                                    weighted_nanpercentile(
                                        vrad[outmask], spec_mass_out, 50
                                    )
                                )
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-vradout_{spec}_90P_{mem_str}"
                                ] = float(
                                    weighted_nanpercentile(
                                        vrad[outmask], spec_mass_out, 90
                                    )
                                )

                        # ---------------- Gas flow rates (all phases) ----------------
                        for vboundary, vkey in zip(vsboundary, vsboundary_str):

                            # Build vmins_use, masking per-particle arrays if present
                            vmins_use = []
                            for iv, vminstr in enumerate(vminstrs):
                                if isinstance(vmins[iv], np.ndarray):
                                    vmins_use.append(vmins[iv][Tmask_shell])
                                else:
                                    vmins_use.append(vmins[iv])

                            gas_flow_rates = calculate_flow_rate(
                                masses=mass[Tmask_shell],
                                vrad=vrad[Tmask_shell],
                                dr=dr_phys,
                                vboundary=vboundary,
                                vmin=vmins_use,
                            )

                            galaxy_output[
                                f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_tot_inflow_{vkey}_vc000kmps_{mem_str}"
                            ] = gas_flow_rates[0]
                            galaxy_output[
                                f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_tot_outflow_{vkey}_vc000kmps_{mem_str}"
                            ] = gas_flow_rates[1]

                            # Additional outflow rates for each vcut in vmins
                            for iv, vminstr in enumerate(vminstrs):
                                galaxy_output[
                                    f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_tot_outflow_{vkey}_{vminstr}_{mem_str}"
                                ] = gas_flow_rates[2 + iv]

                            # Species-resolved gas flow rates (only for Tstr == 'all')
                            if Tstr == "all":
                                for spec in specmass.keys():
                                    spec_flow_rates = calculate_flow_rate(
                                        masses=specmass[spec][Tmask_shell],
                                        vrad=vrad[Tmask_shell],
                                        dr=dr_phys,
                                        vboundary=vboundary,
                                        vmin=vmins_use,
                                    )
                                    galaxy_output[
                                        f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_{spec}_inflow_{vkey}_vc000kmps_{mem_str}"
                                    ] = spec_flow_rates[0]
                                    galaxy_output[
                                        f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_{spec}_outflow_{vkey}_vc000kmps_{mem_str}"
                                    ] = spec_flow_rates[1]

                                    for iv, vminstr in enumerate(vminstrs):
                                        galaxy_output[
                                            f"{rshell_str}_shell{drfac_str}_{theta_str}-gas_{Tstr}-mdot_{spec}_outflow_{vkey}_{vminstr}_{mem_str}"
                                        ] = spec_flow_rates[2 + iv]

    # ------------------------------------------------------------------
    # 15. Cylindrical z-slab calculations (vertical flows through slabs)
    # ------------------------------------------------------------------
    for rshell, rshell_str in zip(radial_shells, radial_shells_str):

        # Only perform slab analysis for certain "inner" shells
        flag_innershell = (
            ("kpc" in rshell_str)
            or ("0p10r200" in rshell_str)
            or ("0p30r200" in rshell_str)
            or ("1p00r200" in rshell_str)
            or ("reff" in rshell_str)
        )

        for dzfac, dzfac_str in zip(dzfacs, dzfacs_str):
            if not flag_innershell:
                continue

            # z-height limits (in comoving Mpc)
            zhi = rshell + (dzfac * rshell) / 2.0
            zlo = rshell - (dzfac * rshell) / 2.0

            # Particles whose |z| lies in [rlo, rhi)
            zmask = np.logical_and(np.abs(zheight) >= zlo, np.abs(zheight) < zhi)

            for rmax_str, rmax_val in zip(zslab_radii_strs, zslab_radii_vals):

                rmax = rmax_val
                if "zheight" in rmax_str:
                    # For this case, the radial extent equals the slab half-height
                    rmax = rhi

                # Cylindrical selection: radius_in_plane < rmax
                mask_shell = np.logical_and.reduce([zmask, rrel_inplane < rmax])

                # Convert slab height limits to physical Mpc
                rhi_phys = rhi * afac
                rlo_phys = rlo * afac
                dr_phys = rhi_phys - rlo_phys

                # Gas phase breakdown within the slab
                for Tstr, Tmask in Tmasks.items():
                    Tmask_shell = np.logical_and.reduce([mask_shell, gas, Tmask])

                    galaxy_output[
                        f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-n_tot"
                    ] = np.nansum(Tmask_shell)

                    # Flow rates perpendicular to the disk use vradz
                    for vboundary, vkey in zip([0], ["vbstatic"]):
                        vmins_use = []
                        for iv, vminstr in enumerate(vminstrs):
                            if isinstance(vmins[iv], np.ndarray):
                                vmins_use.append(vmins[iv][Tmask_shell])
                            else:
                                vmins_use.append(vmins[iv])

                        gas_flow_rates = calculate_flow_rate(
                            masses=mass[Tmask_shell],
                            vrad=vradz[Tmask_shell],
                            dr=dr_phys,
                            vboundary=vboundary,
                            vmin=vmins_use,
                        )

                        galaxy_output[
                            f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_tot_inflow_{vkey}_vc000kmps"
                        ] = gas_flow_rates[0]
                        galaxy_output[
                            f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_tot_outflow_{vkey}_vc000kmps"
                        ] = gas_flow_rates[1]

                        for iv, vminstr in enumerate(vminstrs):
                            galaxy_output[
                                f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_tot_outflow_{vkey}_{vminstr}"
                            ] = gas_flow_rates[2 + iv]

                        # Species-resolved slab flows (only for Tstr == 'all')
                        if Tstr == "all":
                            for spec in specmass.keys():
                                spec_flow_rates = calculate_flow_rate(
                                    masses=specmass[spec][Tmask_shell],
                                    vrad=vradz[Tmask_shell],
                                    dr=dr_phys,
                                    vboundary=vboundary,
                                    vmin=vmins_use,
                                )
                                galaxy_output[
                                    f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_{spec}_inflow_{vkey}_vc000kmps"
                                ] = spec_flow_rates[0]
                                galaxy_output[
                                    f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_{spec}_outflow_{vkey}_vc000kmps"
                                ] = spec_flow_rates[1]

                                for iv, vminstr in enumerate(vminstrs):
                                    galaxy_output[
                                        f"{rshell_str}_zslab{dzfac_str}_{rmax_str}-gas_{Tstr}-mdot_{spec}_outflow_{vkey}_{vminstr}"
                                    ] = spec_flow_rates[2 + iv]

    # ------------------------------------------------------------------
    # Done – return all accumulated outputs
    # ------------------------------------------------------------------
    return galaxy_output
