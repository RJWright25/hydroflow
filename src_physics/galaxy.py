# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells). Both 
import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import constant_G, compute_relative_theta, calc_halfmass_radius
from hydroflow.src_physics.gasflow import calculate_flow_rate

def retrieve_galaxy_candidates(galaxy,pdata_subvol,kdtree_subvol,maxrad=None,boxsize=None): 
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

		# Check if this halo is near the edge of the box
		safe=True
		if boxsize:
			for idim in range(3):
				if com[idim]-maxrad<0 or com[idim]+maxrad>boxsize:
					safe=False
					break
		# If the halo is near the edge of the box, adjust the coordinates
		if not safe:
			for idim,dim in enumerate(['x','y','z']):
				if com[idim]-maxrad<0:
					mask_otherside=pdata_candidates[f'Coordinates_{dim}'].values>boxsize/2
					pdata_candidates[f'Coordinates_{dim}'][mask_otherside]-=boxsize

				elif com[idim]+maxrad>boxsize:
					mask_otherside=pdata_candidates[f'Coordinates_{dim}'].values<boxsize/2
					pdata_candidates[f'Coordinates_{dim}'][mask_otherside]+=boxsize

		# Compute relative position (comoving) based on catalogue centre
		positions_relative=pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com
		radii_relative=np.linalg.norm(positions_relative,axis=1)*afac #physical Mpc

		# Calculate 30pkpc baryonic centre of mass
		mask=np.logical_and(radii_relative<0.03,np.logical_or(pdata_candidates['ParticleType'].values==0.,pdata_candidates['ParticleType'].values==4.))
		if np.nansum(mask):
			# Calculate the com, and vcom
			com_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Coordinates_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			vcom_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Velocities_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			baryons=True
		else:
			mask=np.logical_and(radii_relative<0.03,pdata_candidates['ParticleType'].values==1.)
			com_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Coordinates_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			vcom_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Velocities_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			baryons=False
		
		# Renormalise the coordinates and velocities to the new centre of mass
		pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=(pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com_0p10r200)
		pdata_candidates['Relative_r_comoving']=np.linalg.norm(pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values,axis=1)

		# Calculate the relative velocity
		pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']]=pdata_candidates.loc[:,[f'Velocities_{x}' for x in 'xyz']].values-vcom_0p10r200

		# Calculate the relative radial velocity
		rhat = pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values / np.stack(3 * [pdata_candidates['Relative_r_comoving']], axis=1)
		pdata_candidates['Relative_vrad_pec'] = np.sum(pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']].values * rhat, axis=1)

		# Compute relative theta
		Lbar,thetarel=compute_relative_theta(pdata=pdata_candidates,afac=1/(1+galaxy['Redshift']),baryons=baryons,aperture=0.03)
		pdata_candidates['Relative_theta']=thetarel

		return pdata_candidates
	else:
		return pd.DataFrame()


# Main function to analyse the galaxy and surrounding baryonic reservoirs

def analyse_galaxy(galaxy,pdata_candidates,metadata,
				   r200_shells=[0.1,0.3,1],
				   kpc_shells=[10,30,100],
				   rstar_shells=[1,2,4],
				   Tbins={'cold':[0,1e3],'cool':[1e3,1e5],'warm':[1e5,1e7],'hot':[1e7,1e15]},
				   theta_bins={'full':[0,90],'minax':[60,90],'majax':[0,30]},
				   vcuts={'vc0p25vmx':'0.25Vmax','vc050kmps':50,'vc100kmps':100,'vc250kmps':250},
				   drfacs=[0.1],
				   logfile=None):

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
	kpc_shells: list
		List of radii at which to calculate properties (pkpc). 
	rstar_shells: list
		List of radii at which to calculate properties (multiples of the stellar half-mass radius).
	Tbins: list
		Dict of temperature bins to use for gas properties.
	drfacs: float
		Fractional width(s) of the shell.
	

	Output:
	-----------
	
	galaxy_output: dict
		Dictionary containing the properties of the galaxy's baryonic reservoirs. Each key is a string describing the reservoir, and the value is a float or array of floats containing the properties of the reservoir.

	"""	

	galaxy_output={}

	# Add existing galaxy properties
	for key in galaxy.keys():
		galaxy_output[key]=galaxy[key]

	# Compute psuedoevolution velocity
	omegar=metadata.cosmology.Ogamma(galaxy['Redshift'])
	omegam=metadata.cosmology.Om(galaxy['Redshift'])
	Hz=metadata.cosmology.H(galaxy['Redshift']).value
	afac=1/(1+galaxy['Redshift'])
	vpseudo=2/3*(constant_G/100)**(1/3)*galaxy['Group_M_Crit200']**(1/3)*(2*omegar+3/2*omegam)*Hz**(1/3)
	galaxy_output['1p00r200-vpdoev']=vpseudo #pseudo-evolution velocity cut in km/s
	

	# Velocity cuts (if any)
	galaxy_output['Group_V_Crit200']=np.sqrt(constant_G*galaxy['Group_M_Crit200']/(galaxy['Group_R_Crit200']))
	vmins=[];vminstrs=list(vcuts.keys())
	if 'Subhalo_V_max' in galaxy.keys(): vmax=galaxy['Subhalo_V_max']
	elif 'Group_V_Crit200' in galaxy_output.keys(): vmax=1.3*galaxy_output['Group_V_Crit200']# Otherwise assuming vmax=1.33*vcirc, from NFW profile with c=10
	for vcut in vcuts.keys():
		vcut_kmps=vcuts[vcut]
		if type(vcut_kmps)==str and 'Vmax' in vcut_kmps:
			vcut_kmps=vmax*np.float32(vcut_kmps.split('Vmax')[0])
		vmins.append(vcut_kmps)

	# Shell width for calculations
	drfacs_pc=[idrfac*100 for idrfac in drfacs] #convert to pc
	drfacs_str=['p'+str(f'{idrfac:.0f}').zfill(2) for idrfac in drfacs_pc]

	# Masks
	gas=pdata_candidates['ParticleType'].values==0.
	star=pdata_candidates['ParticleType'].values==4.
	dm=pdata_candidates['ParticleType'].values==1.

	# Pre-load the particle data
	mass=pdata_candidates['Masses'].values
	rrel=pdata_candidates['Relative_r_comoving'].values #relative position to the halo catalogue centre
	rrel=pdata_candidates['Relative_r_comoving'].values #relative position to the centre as per the candidate function
	vrad=pdata_candidates['Relative_vrad_pec'].values #peculiar radil velocity in km/s relative to the centre as per the candidate function
	thetarel=pdata_candidates['Relative_theta'].values #relative theta in degrees
	temp=pdata_candidates['Temperature'].values
	sfr=pdata_candidates['StarFormationRate'].values

	# Gas selections by temperature (adding sf, all)
	Tmasks={'all':gas,'sf':np.logical_and(gas,sfr>0)}
	if Tbins is not None:
		for Tstr,Tbin in Tbins.items():
			Tmasks[Tstr]=np.logical_and.reduce([gas,temp>Tbin[0],temp<Tbin[1]])

	# Species fractions (if available)
	specmass={}
	mfrac_columns=[col for col in pdata_candidates.columns if 'mfrac' in col]
	for mfrac_col in mfrac_columns:
		specmass[mfrac_col.split('mfrac_')[1]]=pdata_candidates[mfrac_col].values*mass
	specmass['Z']=pdata_candidates['Metallicity'].values*mass
	specmass['tot']=np.ones_like(specmass['Z'])*mass

	# Get relative theta masks
	thetamasks={}
	for thetarel_str,thetarel_bin in theta_bins.items():
		thetamasks[thetarel_str]=np.logical_and.reduce([gas,thetarel>thetarel_bin[0],thetarel<thetarel_bin[1]])

	# Get stellar half-mass radius
	star_r_half=np.nan
	star_mask=np.logical_and(star,pdata_candidates['Relative_r_comoving'].values*afac<0.01)
	if np.nansum(star_mask):
		star_r_half=calc_halfmass_radius(pdata_candidates.loc[star_mask,'Masses'].values,pdata_candidates.loc[star_mask,'Relative_r_comoving'].values)

	# Get gas half-mass radius
	gas_r_half=np.nan
	gas_mask=np.logical_and(gas,pdata_candidates['Relative_r_comoving'].values*afac<0.01)
	if np.nansum(gas_mask):
		gas_r_half=calc_halfmass_radius(pdata_candidates.loc[gas_mask,'Masses'].values,pdata_candidates.loc[gas_mask,'Relative_r_comoving'].values)

	# Add to the galaxy output
	galaxy_output['010pkpc_sphere-star-r_half']=star_r_half
	galaxy_output['010pkpc_sphere-gas-r_half']=gas_r_half

	# Combine all the shell radii for analysis
	radial_shells_R200=[fR200*galaxy['Group_R_Crit200'] for fR200 in r200_shells] #numerical values are comoving
	radial_shells_pkpc=[fpkpc/1e3/afac for fpkpc in kpc_shells] #numerical values are comoving
	radial_shells_rstar=[fstar*star_r_half for fstar in rstar_shells] #numerical values are comoving
	radial_shells_R200_str=[f'{fR200:.2f}'.replace('.','p')+'r200' for fR200 in r200_shells]
	radial_shells_pkpc_str=[str(int(fpkpc)).zfill(3)+'pkpc' for fpkpc in kpc_shells]
	radial_shells_rstar_str=[f'{fstar:.2f}'.replace('.','p')+'reff' for fstar in rstar_shells]
	radial_shells=radial_shells_R200+radial_shells_pkpc+radial_shells_rstar
	radial_shells_str=radial_shells_R200_str+radial_shells_pkpc_str+radial_shells_rstar_str

	# Loop over all the shells
	for rshell,rshell_str in zip(radial_shells,radial_shells_str):

		# Pseudo-evolution velocity cut (updated for each shell)
		if 'r200' in rshell_str and galaxy['SubGroupNumber']==0:
			vbdef=vpseudo*(rshell/galaxy['Group_R_Crit200'])
		else:
			vbdef=0
		vsboundary=[vbdef]
		vsboundary_str=['vbdef']

		# Skip the shell if it is a multiple of r200 and the galaxy is a satellite
		if not ('r200' in rshell_str and galaxy['SubGroupNumber']>0) and rshell>0:
			
			#### SPHERE CALCULATIONS (r<rshell) ####
			# Mask for the sphere in comoving coordinates
			mask_sphere=rrel<=rshell
			
			# Add the sphere volume in pkpc^3
			galaxy_output[f'{rshell_str}_sphere-vol']=4/3*np.pi*(rshell*afac*1e3)**3
		
			### DARK MATTER
			galaxy_output[f'{rshell_str}_sphere-dm-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,dm)])
			galaxy_output[f'{rshell_str}_sphere-dm-n_tot']=np.nansum(np.logical_and(mask_sphere,dm))

			### STARS
			galaxy_output[f'{rshell_str}_sphere-star-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,star)])
			galaxy_output[f'{rshell_str}_sphere-star-n_tot']=np.nansum(np.logical_and(mask_sphere,star))
			galaxy_output[f'{rshell_str}_sphere-star-Z']=np.nansum(specmass['Z'][np.logical_and(mask_sphere,star)])/np.nansum(mass[np.logical_and(mask_sphere,star)])

			### GAS
			# Break down the gas mass by phase
			for Tstr,Tmask in Tmasks.items():
				Tmask_sphere=np.logical_and.reduce([mask_sphere,gas,Tmask])
				galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_sphere)

				# Breakdown of mass in this phase by species
				for spec in specmass.keys():
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-m_{spec}']=np.nansum(specmass[spec][Tmask_sphere])

				# If considering a galaxy-scale shell, calculate the SFR and metallicity
				if ('kpc' in rshell_str or '0p10' in rshell_str or 'reff' in rshell_str):
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_sphere])
					galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-Z']=np.nansum(specmass['Z'][Tmask_sphere])/np.nansum(mass[Tmask_sphere])
							
			#### SPHERICAL SHELL CALCULATIONS (r between r-dr/2 and r+dr/2) ####
			# Mask for the shell in comoving coordinates (particle data is in comoving coordinates)
			for drfac,drfac_str in zip(drfacs,drfacs_str):
				rshell_str=rshell_str
				r_hi=rshell+(drfac*rshell)/2
				r_lo=rshell-(drfac*rshell)/2
				mask_shell=np.logical_and(rrel>=r_lo,rrel<r_hi)

				# Now convert the shell values to physical units for the calculations
				r_hi=r_hi*afac
				r_lo=r_lo*afac
				dr=r_hi-r_lo

				# Shell volume and area in pkpc^3 and pkpc^2
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-vol']=4/3*np.pi*((r_hi*1e3)**3-(r_lo*1e3)**3)
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-area']=4*np.pi*((r_hi*1e3)**2-(r_lo*1e3)**2)

				### DM shell properties
				dm_shell_mask=np.logical_and(mask_shell,dm)
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-dm-m_tot']=np.nansum(mass[dm_shell_mask])
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-dm-n_tot']=np.nansum(dm_shell_mask)
				for vboundary, vkey in zip(vsboundary, vsboundary_str):
					dm_flow_rates=calculate_flow_rate(masses=mass[dm_shell_mask],vrad=vrad[dm_shell_mask],dr=dr,vboundary=vbdef,vmin=[])
					galaxy_output[f'{rshell_str}_shell{drfac_str}_full-dm-mdot_tot_inflow_{vkey}_vc000kmps']=dm_flow_rates[0]
					galaxy_output[f'{rshell_str}_shell{drfac_str}_full-dm-mdot_tot_outflow_{vkey}_vc000kmps']=dm_flow_rates[1]

				### STARS shell properties
				stars_shell_mask=np.logical_and(mask_shell,star)
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-star-m_tot']=np.nansum(mass[stars_shell_mask])
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-star-n_tot']=np.nansum(stars_shell_mask)
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-star-Z']=np.nansum(specmass['Z'][stars_shell_mask])/np.nansum(mass[stars_shell_mask])
				for vboundary, vkey in zip(vsboundary, vsboundary_str):
					stars_flow_rates=calculate_flow_rate(masses=mass[stars_shell_mask],vrad=vrad[stars_shell_mask],dr=dr,vboundary=vboundary,vmin=[])
					galaxy_output[f'{rshell_str}_shell{drfac_str}_full-star-mdot_tot_inflow_{vkey}_vc000kmps']=stars_flow_rates[0]
					galaxy_output[f'{rshell_str}_shell{drfac_str}_full-star-mdot_tot_outflow_{vkey}_vc000kmps']=stars_flow_rates[1]
				
				### GAS shell properties
				# Break down by theta
				for thetarel_str,thetamask in thetamasks.items():
					mask_shell_theta=np.logical_and(mask_shell,thetamask)

					# Break down the gas mass by phase
					for Tstr,Tmask in Tmasks.items():
						Tmask_shell=np.logical_and.reduce([mask_shell_theta,gas,Tmask])
						galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_shell)
						
						# If considering a galaxy-scale shell, calculate the SFR and metallicity
						if ('kpc' in rshell_str or '0p10' in rshell_str or 'reff' in rshell_str):
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_shell])
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-Z']=np.nansum(specmass['Z'][Tmask_shell])/np.nansum(mass[Tmask_shell])

						# Breakdown of mass in this phase by species
						for spec in specmass.keys():
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-m_{spec}']=np.nansum(specmass[spec][Tmask_shell])
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-vrad_{spec}_mean']=np.nansum(vrad[Tmask_shell]*specmass[spec][Tmask_shell])/np.nansum(specmass[spec][Tmask_shell])
						
						# Calculate the total flow rates for the gas
						for vboundary, vkey in zip(vsboundary, vsboundary_str):
							gas_flow_rates=calculate_flow_rate(masses=mass[Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins)
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_inflow_{vkey}_vc000kmps']=gas_flow_rates[0]
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_vc000kmps']=gas_flow_rates[1]
							for iv,vminstr in enumerate(vminstrs):
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_{vminstr}']=gas_flow_rates[2+iv]

							# Calculate the flow rates for the gas by species
							for spec in specmass.keys():
								gas_flow_rates_species=calculate_flow_rate(masses=specmass[spec][Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins)
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_inflow_{vkey}_vc000kmps']=gas_flow_rates_species[0]
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_vc000kmps']=gas_flow_rates_species[1]
								for iv,vminstr in enumerate(vminstrs):
									galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_{vminstr}']=gas_flow_rates_species[2+iv]

	# #### CYLINDRICAL SLAB CALCULATIONS (z between r-dr/2 and r+dr/2) ####
	# # Mask for the shell in comoving coordinates (particle data is in comoving coordinates)
	# for rshell,rshell_str in zip(radial_shells,radial_shells_str):
	# 	for drfac,drfac_str in zip(drfacs,drfacs_str):
	# 		if ('kpc' in rshell_str or '0p10' in rshell_str or 'reff' in rshell_str):
	# 			# Only do for kpc, rstar and 0.1r200 shells
	# 			rshell_str=rshell_str;thetarel_str='full'
	# 			r_hi=rshell+(drfac*rshell)/2
	# 			r_lo=rshell-(drfac*rshell)/2
	# 			zheight=rrel*np.sin(np.radians(thetarel))


	# 			mask_shell=np.logical_and(zheight>=r_lo,zheight<r_hi)

	# 			# Now convert the shell values to physical units for the calculations
	# 			r_hi=r_hi*afac
	# 			r_lo=r_lo*afac
	# 			dr=r_hi-r_lo

	# 			### GAS shell properties
	# 			# Break down the gas mass by phase
	# 			for Tstr,Tmask in Tmasks.items():
	# 				Tmask_shell=np.logical_and.reduce([mask_shell_theta,gas,Tmask])
	# 				galaxy_output[f'{rshell_str}_zslab{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_shell)
					
	# 				# If considering a galaxy-scale shell, calculate the SFR and metallicity
	# 				if ('kpc' in rshell_str or '0p10' in rshell_str or 'reff' in rshell_str):
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_shell])
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-Z']=np.nansum(specmass['Z'][Tmask_shell])/np.nansum(mass[Tmask_shell])

	# 				# Breakdown of mass in this phase by species
	# 				for spec in specmass.keys():
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-m_{spec}']=np.nansum(specmass[spec][Tmask_shell])
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-vrad_{spec}_mean']=np.nansum(vrad[Tmask_shell]*specmass[spec][Tmask_shell])/np.nansum(specmass[spec][Tmask_shell])
					
	# 				# Calculate the total flow rates for the gas
	# 				for vboundary, vkey in zip(vsboundary, vsboundary_str):
	# 					gas_flow_rates=calculate_flow_rate(masses=mass[Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins)
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_inflow_{vkey}_vc000kmps']=gas_flow_rates[0]
	# 					galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_vc000kmps']=gas_flow_rates[1]
	# 					for iv,vminstr in enumerate(vminstrs):
	# 						galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_{vminstr}']=gas_flow_rates[2+iv]

	# 					# Calculate the flow rates for the gas by species
	# 					for spec in specmass.keys():
	# 						gas_flow_rates_species=calculate_flow_rate(masses=specmass[spec][Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins)
	# 						galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_inflow_{vkey}_vc000kmps']=gas_flow_rates_species[0]
	# 						galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_vc000kmps']=gas_flow_rates_species[1]
	# 						for iv,vminstr in enumerate(vminstrs):
	# 							galaxy_output[f'{rshell_str}_shell{drfac_str}_{thetarel_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_{vminstr}']=gas_flow_rates_species[2+iv]


		else:
			pass

	# Return the galaxy output dictionary
	return galaxy_output


