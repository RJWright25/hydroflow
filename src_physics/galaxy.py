# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells). Both 
import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import constant_G, compute_relative_phi
from hydroflow.src_physics.gasflow import calculate_flow_rate

def retrieve_galaxy_candidates(galaxy,pdata_subvol,kdtree_subvol,maxrad=None): 
	"""
	
	retrieve_galaxy_candidates: Retrieve the baryonic candidates for a galaxy within a specified radius.


	Input:
	-----------
	galaxy: dict or pd.Series
		Dictionary containing the properties of the galaxy, including the virial radius.

	pdata_subvol: pd.DataFrame
		DataFrame containing the particle data for the relevant subvolume.

	kdtree_subvol: scipy.spatial.cKDTree
		KDTree containing the particle data for the relevant subvolume -- used to speed up the search.

	maxrad: float
		Maximum radius to search for candidates (in comoving Mpc).

	Output:
	-----------
	pdata_candidates: pd.DataFrame
		DataFrame containing the particle for the galaxy within the specified radius. 
		Additional columns are added to the DataFrame to include the relative position and radial distance of each particle from the galaxy.
	

	"""

	# Define the centre of mass and scale factor
	com=np.array([galaxy[f'CentreOfPotential_{x}'] for x in 'xyz'])
	afac=1/(1+galaxy['Redshift'])

	# Define the maximum radius to search for candidates if not provided
	if maxrad is None:
		maxrad=galaxy['Group_R_Crit200']*3.5

	# Retrieve the candidates from the KDTree
	pidx_candidates=kdtree_subvol.query_ball_point(com,maxrad)
	pdata_candidates=pdata_subvol.loc[pidx_candidates,:]
	pdata_candidates.reset_index(drop=True,inplace=True)
	numcdt=pdata_candidates.shape[0]

	# Get derived quantities if there are elements within the radius
	if numcdt>0:
		# Compute relative position (comoving) based on catalogue centre
		positions_relative=pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com
		radii_relative=np.linalg.norm(positions_relative,axis=1)

		# Calculate 0p10r200 centre of mass
		mask=(radii_relative<0.1*galaxy['Group_R_Crit200'])
		
		# Calculate the com, and vcom
		com_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Coordinates_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
		vcom_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Velocities_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
	
		# Renormalise 	
		pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=(pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com_0p10r200)
		pdata_candidates['Relative_r_comoving']=np.linalg.norm(pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values,axis=1)
		pdata_candidates['Relative_r_physical']=pdata_candidates['Relative_r_comoving']*afac

		# Calculate the relative velocity
		pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']]=pdata_candidates.loc[:,[f'Velocities_{x}' for x in 'xyz']].values-vcom_0p10r200

		# Calculate the relative radial velocity
		rhat = pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values / np.stack(3 * [pdata_candidates['Relative_r_comoving']], axis=1)
		pdata_candidates['Relative_vrad_pec'] = np.sum(pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']].values * rhat, axis=1)

		return pdata_candidates
	else:
		return pd.DataFrame()


# Main function to analyse the galaxy and surrounding baryonic reservoirs

def analyse_galaxy(galaxy,pdata_candidates,metadata,r200_shells=None,ckpc_shells=None,Tbins=None,drfac=None,logfile=None):

	"""
	analyse_galaxy: Main function to analyse the galaxy and baryonic reservoirs. Computes properties within spheres and shells of the galaxy.

	Input:
	-----------
	galaxy: dict or pd.Series
		Dictionary containing the properties of the galaxy, including the virial radius and scale factor.
	pdata: pd.DataFrame
		DataFrame containing the particle data for the relevant galaxy. Should include the particle properties out to the largest radius of interest.
	r200_shells: list
		List of radii at which to calculate properties (multiples of R200).
	ckpc_shells: list
		List of radii at which to calculate properties (in ckpc, but will also calculate for pkpc). 
	Tbins: list
		Dict of temperature bins to use for gas properties.
	drfac: float
		Fractional width of the shell.
	

	Output:
	-----------
	
	galaxy_output: dict
		Dictionary containing the properties of the galaxy's baryonic reservoirs. Each key is a string describing the reservoir, and the value is a float or array of floats containing the properties of the reservoir.

	"""	

	galaxy_output={}

	# Add existing galaxy properties
	for key in galaxy.keys():
		galaxy_output[key]=galaxy[key]

	# Masks
	gas=pdata_candidates['ParticleType'].values==0.
	star=pdata_candidates['ParticleType'].values==4.
	dm=pdata_candidates['ParticleType'].values==1.

	# Fields
	mass=pdata_candidates['Masses'].values
	rrel=pdata_candidates['Relative_r_comoving'].values #relative position to the halo catalogue centre
	coordinates=pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values #comoving coordinates
	velocities=pdata_candidates.loc[:,[f'Velocities_{x}' for x in 'xyz']].values #peculiar velocity in km/s

	# Gas properties
	temp=pdata_candidates['Temperature'].values
	sfr=pdata_candidates['StarFormationRate'].values

	# Gas selections by temperature
	Tmasks={'all':gas}
	if Tbins is not None:
		for Tstr,Tbin in Tbins.items():
			Tmasks[Tstr]=np.logical_and.reduce([gas,temp>Tbin[0],temp<Tbin[1]])

	# Species fractions (if available)
	specfrac={}
	mfrac_columns=[col for col in pdata_candidates.columns if 'mfrac' in col]
	for mfrac_col in mfrac_columns:
		specfrac[mfrac_col.split('mfrac_')[0]]=pdata_candidates[mfrac_col].values
	specfrac['Z']=pdata_candidates['Metallicity'].values
	specfrac['tot']=np.ones_like(gas)

	# Get relative phi
	Lbar,phirel=compute_relative_phi(pdata=pdata_candidates,baryons=True,aperture=galaxy['Group_R_Crit200']*0.1)
	pdata_candidates['Relative_phi']=phirel

	# Add to the galaxy output
	galaxy_output['0p10r200-Lbartot_x']=Lbar[0]
	galaxy_output['0p10r200-Lbartot_y']=Lbar[1]
	galaxy_output['0p10r200-Lbartot_z']=Lbar[2]
	
	# Compute psuedoevolution velocity
	omegar=metadata.cosmology.Ogamma(galaxy['Redshift'])
	omegam=metadata.cosmology.Om(galaxy['Redshift'])
	Hz=metadata.cosmology.H(galaxy['Redshift']).value
	afac=1/(1+galaxy['Redshift'])
	vpseudo=2/3*(constant_G/100)**(1/3)*galaxy['Group_M_Crit200']**(1/3)*(2*omegar+3/2*omegam)*Hz**(1/3)
	galaxy_output['1p00r200-v_pdoev']=vpseudo #pseudo-evolution velocity cut in km/s
	
	# Combine all the shell radii for analysis
	radial_shells_R200=[fR200*galaxy['Group_R_Crit200'] for fR200 in r200_shells] #numerical values are comoving
	radial_shells_ckpc=[fckpc/1e3 for fckpc in ckpc_shells] #numerical values are comoving
	radial_shells_pkpc=[fpkpc/1e3/afac for fpkpc in ckpc_shells] #numerical values are comoving
	radial_shells_R200_str=[f'{fR200:.2f}'.replace('.','p')+'r200' for fR200 in r200_shells]
	radial_shells_ckpc_str=[str(int(fckpc)).zfill(3)+'ckpc' for fckpc in ckpc_shells]
	radial_shells_pkpc_str=[str(int(fpkpc)).zfill(3)+'pkpc' for fpkpc in ckpc_shells]
	radial_shells=radial_shells_R200+radial_shells_ckpc+radial_shells_pkpc
	radial_shells_str=radial_shells_R200_str+radial_shells_ckpc_str+radial_shells_pkpc_str

	# Loop over all the shells
	for rshell,rshell_str in zip(radial_shells,radial_shells_str):

		# Pseudo-evolution velocity cut (updated for each shell)
		vpseudo_ishell=vpseudo*(rshell/galaxy['Group_R_Crit200'])

		# Skip the shell if it is a multiple of r200 and the galaxy is a satellite
		if not ('r200' in rshell_str and galaxy['SubGroupNumber']>0):
			
			#### SPHERE CALCULATIONS (r<rshell) ####

			# Mask for the sphere in comoving coordinates
			mask_sphere=rrel<=rshell
			
			# Add the sphere volume in pkpc^3
			galaxy_output[f'{rshell_str}_sphere-vol']=4/3*np.pi*(rshell*afac*1e3)**3

			# Calculate the centre of mass position in the given sphere
			com_sphere=np.nansum(mass[mask_sphere,np.newaxis]*coordinates[mask_sphere],axis=0)/np.nansum(mass[mask_sphere])
			galaxy_output[f'{rshell_str}_sphere-com_x']=com_sphere[0]
			galaxy_output[f'{rshell_str}_sphere-com_y']=com_sphere[1]
			galaxy_output[f'{rshell_str}_sphere-com_z']=com_sphere[2]

			# Calculate the centre of mass velocity in the given sphere
			vcom_sphere=np.nansum(mass[mask_sphere,np.newaxis]*velocities[mask_sphere],axis=0)/np.nansum(mass[mask_sphere])
			galaxy_output[f'{rshell_str}_sphere-vcom_x']=vcom_sphere[0]
			galaxy_output[f'{rshell_str}_sphere-vcom_y']=vcom_sphere[1]
			galaxy_output[f'{rshell_str}_sphere-vcom_z']=vcom_sphere[2]

			# Calculate the relative position of particles in this sphere using the new centre of mass
			positions=coordinates-com_sphere
			radii=np.linalg.norm(positions,axis=1)

			# Calculate the relative velocity of particles in this sphere using the new centre of mass
			rhat = positions / np.stack(3 * [radii], axis=1)
			vrad = np.sum((velocities-vcom_sphere) * rhat, axis=1)			

			### DARK MATTER
			galaxy_output[f'{rshell_str}_sphere-dm-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,dm)])
			galaxy_output[f'{rshell_str}_sphere-dm-n_tot']=np.nansum(np.logical_and(mask_sphere,dm))

			### STARS
			galaxy_output[f'{rshell_str}_sphere-star-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,star)])
			galaxy_output[f'{rshell_str}_sphere-star-n_tot']=np.nansum(np.logical_and(mask_sphere,star))
			galaxy_output[f'{rshell_str}_sphere-star-Z']=np.nansum(specfrac['Z'][np.logical_and(mask_sphere,star)]*mass[np.logical_and(mask_sphere,star)])/np.nansum(mass[np.logical_and(mask_sphere,star)])

			### GAS
			# Break down the gas mass by phase
			for Tstr,Tmask in Tmasks.items():
				Tmask_sphere=np.logical_and.reduce([mask_sphere,gas,Tmask])
				galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_sphere)

				# Breakdown of mass in this phase by species
				for spec in specfrac.keys():
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-m_{spec}']=np.nansum(mass[Tmask_sphere]*specfrac[spec][Tmask_sphere])

				# If considering a galaxy-scale shell, calculate the SFR and metallicity
				if ('kpc' in rshell_str or '0p10' in rshell_str):
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_sphere])
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-Z']=np.nansum(specfrac['Z'][Tmask_sphere]*mass[Tmask_sphere])/np.nansum(mass[Tmask_sphere])
							
			#### SHELL CALCULATIONS (r between r-dr/2 and r+dr/2) ####

			# Mask for the shell in comoving coordinates (particle data is in comoving coordinates)
			r_hi=rshell+(drfac*rshell)/2
			r_lo=rshell-(drfac*rshell)/2
			mask_shell=np.logical_and(radii>=r_lo,radii<r_hi)

			# Now convert the shell values to physical units for the calculations
			r_hi=r_hi*afac
			r_lo=r_lo*afac
			dr=r_hi-r_lo

			# Shell volume and area in pkpc^3 and pkpc^2
			galaxy_output[f'{rshell_str}_shell-vol']=4/3*np.pi*((r_hi*1e3)**3-(r_lo*1e3)**3)
			galaxy_output[f'{rshell_str}_shell-area']=4*np.pi*((r_hi*1e3)**2-(r_lo*1e3)**2)

			### DM shell properties
			dm_shell_mask=np.logical_and(mask_shell,dm)
			galaxy_output[f'{rshell_str}_shell-dm-m_tot']=np.nansum(mass[dm_shell_mask])
			galaxy_output[f'{rshell_str}_shell-dm-n_tot']=np.nansum(dm_shell_mask)
			
			for vboundary, vkey in zip([0, vpseudo_ishell], ['000kmps', 'pdoev']):
				dm_flow_rates=calculate_flow_rate(masses=mass[dm_shell_mask],vrad=vrad[dm_shell_mask],dr=dr,vboundary=vboundary)
				galaxy_output[f'{rshell_str}_shell-dm-mdot_tot_inflow_{vkey}_pec']=dm_flow_rates[0]
				galaxy_output[f'{rshell_str}_shell-dm-mdot_tot_outflow_{vkey}_pec']=dm_flow_rates[1]

			### STARS shell properties
			stars_shell_mask=np.logical_and(mask_shell,star)
			galaxy_output[f'{rshell_str}_shell-star-m_tot']=np.nansum(mass[stars_shell_mask])
			galaxy_output[f'{rshell_str}_shell-star-n_tot']=np.nansum(stars_shell_mask)
			galaxy_output[f'{rshell_str}_shell-star-Z']=np.nansum(specfrac['Z'][stars_shell_mask]*mass[stars_shell_mask])/np.nansum(mass[stars_shell_mask])

			for vboundary, vkey in zip([0, vpseudo_ishell], ['000kmps', 'pdoev']):
				stars_flow_rates=calculate_flow_rate(masses=mass[stars_shell_mask],vrad=vrad[stars_shell_mask],dr=dr,vboundary=vboundary)
				galaxy_output[f'{rshell_str}_shell-star-mdot_tot_inflow_{vkey}_pec']=stars_flow_rates[0]
				galaxy_output[f'{rshell_str}_shell-star-mdot_tot_outflow_{vkey}_pec']=stars_flow_rates[1]
			

			### GAS shell properties
			# Break down the gas mass by phase
			for Tstr,Tmask in Tmasks.items():
				Tmask_shell=np.logical_and.reduce([mask_shell,gas,Tmask])
				galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_shell)

				# Breakdown of mass in this phase by species
				for spec in specfrac.keys():
					galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-m_{spec}']=np.nansum(mass[Tmask_shell]*specfrac[spec][Tmask_shell])
				
				# If considering a galaxy-scale shell, calculate the SFR and metallicity
				if ('kpc' in rshell_str or '0p10' in rshell_str):
					galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_shell])
					galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-Z']=np.nansum(specfrac['Z'][Tmask_shell]*mass[Tmask_shell])/np.nansum(mass[Tmask_shell])

				# Calculate the total flow rates for the gas
				for vboundary, vkey in zip([0, vpseudo_ishell], ['000kmps', 'pdoev']):
					gas_flow_rates=calculate_flow_rate(masses=mass[Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary)
					galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-mdot_tot_inflow_{vkey}_pec']=gas_flow_rates[0]
					galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_pec']=gas_flow_rates[1]

					# Calculate the flow rates for the gas by species
					for spec in specfrac.keys():
						gas_flow_rates_species=calculate_flow_rate(masses=mass[Tmask_shell]*specfrac[spec][Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary)
						galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-mdot_{spec}_inflow_{vkey}_pec']=gas_flow_rates_species[0]
						galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_pec']=gas_flow_rates_species[1]

			else:
				pass

	#write to log
	return galaxy_output


