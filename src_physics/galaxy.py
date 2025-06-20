# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells). Both 
import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import constant_G, compute_cylindrical_ztheta, calc_halfmass_radius, weighted_nanpercentile, estimate_mu
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
		mask=np.logical_and(radii_relative<0.03,np.logical_or(pdata_candidates['ParticleType'].values==0.,np.logical_not(pdata_candidates['ParticleType'].values==1)))
		if np.nansum(mask):
			# Calculate the com, and vcom
			com_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Coordinates_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			vcom_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Velocities_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
		else:
			mask=np.logical_and(radii_relative<0.03,pdata_candidates['ParticleType'].values==1.)
			com_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Coordinates_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
			vcom_0p10r200=np.nansum(pdata_candidates.loc[mask,'Masses'].values[:,np.newaxis]*pdata_candidates.loc[mask,[f'Velocities_{x}' for x in 'xyz']].values,axis=0)/np.nansum(pdata_candidates.loc[mask,'Masses'].values)
		
		# Renormalise the coordinates and velocities to the new centre of mass
		pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=(pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com_0p10r200)
		pdata_candidates['Relative_r_comoving']=np.linalg.norm(pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values,axis=1)

		# Calculate the relative velocity
		pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']]=pdata_candidates.loc[:,[f'Velocities_{x}' for x in 'xyz']].values-vcom_0p10r200

		# Calculate the relative radial velocity
		rhat = pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values / np.stack(3 * [pdata_candidates['Relative_r_comoving']], axis=1)
		pdata_candidates['Relative_vrad_pec'] = np.sum(pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']].values * rhat, axis=1)

		# Sort by relative radius
		pdata_candidates.sort_values(by='Relative_r_comoving',inplace=True)
		pdata_candidates.reset_index(drop=True,inplace=True)

		return pdata_candidates
	else:
		return pd.DataFrame()


# Main function to analyse the galaxy and surrounding baryonic reservoirs

def analyse_galaxy(galaxy,pdata_candidates,metadata,
				   r200_shells=[0.1,0.3,1],
				   kpc_shells=[10,30,100],
				   rstar_shells=[1,2,4],
				   zslab_radii={'rmx2reff':'2r_half','rmx10pkpc':10,'rmxzheight':1},
				   Tbins={'cold':[0,1e3],'cool':[1e3,1e5],'warm':[1e5,1e7],'hot':[1e7,1e15]},
				   theta_bins={'full':[0,90],'minax':[60,90],'majax':[0,30]},
				   vcuts={'vc0p25vmx':'0.25Vmax','vc1p00vmx':'1.00Vmax','vc050kmps':50,'vc250kmps':250},
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
	
	# Save used com from the position of the closest particle
	for idim,dim in enumerate(['x','y','z']):
		galaxy_output[f'030pkpc_sphere-combar_{dim}']=pdata_candidates.loc[0,f'Coordinates_{dim}']

	# Shell width for calculations
	drfacs_pc=[idrfac*100 for idrfac in drfacs] #convert to pc
	drfacs_str=['p'+str(f'{idrfac:.0f}').zfill(2) for idrfac in drfacs_pc]

	# Compute relative zheight and theta
	Lbar,thetapos,thetavel,zheight=compute_cylindrical_ztheta(pdata=pdata_candidates,afac=afac,baryons=True,aperture=0.03)
	pdata_candidates['Relative_theta_pos']=thetapos
	pdata_candidates['Relative_theta_vel']=thetavel
	pdata_candidates['Relative_zheight']=zheight
	for idim,dim in enumerate(['x','y','z']):
		galaxy_output[f'030pkpc_sphere-Lbar{dim}']=Lbar[idim]

	# Pre-load the particle data
	mass=pdata_candidates['Masses'].values
	rrel=pdata_candidates['Relative_r_comoving'].values #relative position to the halo centre
	vrad=pdata_candidates['Relative_vrad_pec'].values #peculiar radial velocity in km/s relative to the centre as per the candidate function
	thetapos=pdata_candidates['Relative_theta_pos'].values #relative theta in degrees
	thetavel=pdata_candidates['Relative_theta_vel'].values #relative theta in degrees
	temp=pdata_candidates['Temperature'].values
	sfr=pdata_candidates['StarFormationRate'].values
	vxyz=pdata_candidates.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']].values #relative velocity in km/s
	vradz=np.dot(vxyz,Lbar/np.linalg.norm(Lbar)) # Get z radial velocity vector
	vradz[zheight<0]*=-1 # Flip sign of z radial velocity for particles below the plane
	rrel_inplane=np.sqrt(rrel**2-zheight**2) # Get the in-plane radius

	# Masks
	gas=pdata_candidates['ParticleType'].values==0.
	star=pdata_candidates['ParticleType'].values==4.
	dm=pdata_candidates['ParticleType'].values==1.

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
	key_HI= 'mfrac_HI';key_H2='mfrac_H2'
	if 'mfrac_HI_BR06' in mfrac_columns:
		key_HI='mfrac_HI_BR06';key_H2='mfrac_H2_BR06'
	print(f'Using {key_HI} and {key_H2} for ionised fractions of H')
	ionised_frac_H= np.ones_like(temp) # If no species, assume fully ionised
	if key_HI is not None and key_H2 is not None:
		ionised_frac_H=1-(pdata_candidates[key_HI][:]+ pdata_candidates[key_H2][:])/(0.76)

	# Velocity cuts (if any)
	galaxy_output['Group_V_Crit200']=np.sqrt(constant_G*galaxy['Group_M_Crit200']/(galaxy['Group_R_Crit200']))
	vmins=[];vminstrs=list(vcuts.keys())
	if 'Subhalo_V_max' in galaxy.keys():
		vmax=galaxy['Subhalo_V_max']
		print(f'Using Subhalo_V_max for Vmax: val = {vmax:.2f} km/s'.format(vmax))
	elif 'Group_V_Crit200' in galaxy_output.keys(): 
		vmax=1.33*galaxy_output['Group_V_Crit200']# Otherwise assuming vmax=1.33*vcirc, from NFW profile with c=10
	for vcut in vcuts.keys():
		vcut_kmps=vcuts[vcut]
		if type(vcut_kmps)==str and 'Vmax' in vcut_kmps:
			vcut_kmps=vmax*np.float32(vcut_kmps.split('Vmax')[0])
		vmins.append(vcut_kmps)
		
	# Extra (Bernoulli) velocity cuts
	potential_infinity=-constant_G*np.nansum(mass)/(np.nanmax(rrel)*afac)
	potential_profile=-constant_G*np.cumsum(mass)/(rrel*afac)
	indices_3r = np.searchsorted(rrel,  3 * rrel);indices_3r=np.clip(indices_3r, 0, len(rrel) - 1)
	potential_atxrrel = potential_profile[indices_3r]
	mu=estimate_mu(x_H=ionised_frac_H,T=temp,y=0.08) #Estimate the mean molecular weight based on the ionised fraction and temperature
	cs=0.129*np.sqrt(temp/mu);gamma=5/3 #ideal gas
	# vb_toinf=np.sqrt(2*(potential_infinity - potential_profile) - 2*cs**2/(gamma-1) )
	vb_to3r=np.sqrt(2*(potential_atxrrel - potential_profile) - 2*cs**2/(gamma-1)  )
	# vmins.append(vb_toinf);vminstrs.append('vcbntoinf')
	vmins.append(vb_to3r); vminstrs.append('vcbnto3rr')

	# Get stellar half-mass radius
	star_r_half=np.nan
	star_rz_half=np.nan
	star_mask=np.logical_and(star,rrel*afac<0.01)
	if np.nansum(star_mask):
		star_r_half=calc_halfmass_radius(mass[star_mask],rrel[star_mask])
		star_rz_half=calc_halfmass_radius(mass[star_mask],np.abs(zheight[star_mask]))

	# Get gas half-mass radius
	gas_r_half=np.nan
	gas_rz_half=np.nan
	gas_mask=np.logical_and(gas,rrel*afac<0.03)
	if np.nansum(gas_mask):
		gas_r_half=calc_halfmass_radius(mass[gas_mask],rrel[gas_mask])
		gas_rz_half=calc_halfmass_radius(mass[gas_mask],np.abs(zheight[gas_mask]))

	# Get relative theta masks
	thetamasks={}
	for theta_str,theta_bin in theta_bins.items():
		if theta_str=='full':
			thetamasks[theta_str]=np.logical_and.reduce([gas])
		else:
			thetamasks[theta_str+'pos']=np.logical_and.reduce([gas,thetapos>theta_bin[0],thetapos<theta_bin[1]])
			thetamasks[theta_str+'vel']=np.logical_and.reduce([gas,thetavel>theta_bin[0],thetavel<theta_bin[1]])
	
	nondisc_mask=np.logical_and.reduce([gas,np.logical_not(np.logical_and(np.abs(zheight)<(gas_rz_half*2),rrel_inplane<(gas_r_half*2)))]) #non-disk gas
	for theta_str in list(thetamasks.keys()):
		thetamasks[theta_str+'nd']=np.logical_and.reduce([nondisc_mask,thetamasks[theta_str]]) #non-disk gas with theta selection

	# Add to the galaxy output
	galaxy_output['030pkpc_sphere-star-r_half']=star_r_half
	galaxy_output['030pkpc_sphere-star-rz_half']=star_rz_half
	galaxy_output['030pkpc_sphere-gas-r_half']=gas_r_half
	galaxy_output['030pkpc_sphere-gas-rz_half']=gas_rz_half

	# Max co-planar radii for z-slab calculations
	zslab_radii_vals=[];zslab_radii_strs=list(zslab_radii.keys())
	for zslab_radius in zslab_radii.values():
		if type(zslab_radius)==str and 'r_half' in zslab_radius:
			zslab_radius=galaxy_output['010pkpc_sphere-star-r_half']*np.float32(zslab_radius.split('r_half')[0])
		elif type(zslab_radius)==str and 'zheight' in zslab_radius:
			zslab_radius='zheight'
		else:
			zslab_radius=zslab_radius/1e3/afac #convert from pkpc to comoving Mpc
		zslab_radii_vals.append(zslab_radius)

	# Combine all the shell radii for analysis
	radial_shells_R200=[fR200*galaxy['Group_R_Crit200'] for fR200 in r200_shells] #numerical values are comoving
	radial_shells_pkpc=[fpkpc/1e3/afac for fpkpc in kpc_shells] #numerical values are comoving
	radial_shells_rstar=[fstar*star_r_half for fstar in rstar_shells] #numerical values are comoving
	radial_shells_R200_str=[f'{fR200:.2f}'.replace('.','p')+'r200' for fR200 in r200_shells]
	radial_shells_pkpc_str=[str(int(fpkpc)).zfill(3)+'pkpc' for fpkpc in kpc_shells]
	radial_shells_rstar_str=[f'{fstar:.2f}'.replace('.','p')+'reff' for fstar in rstar_shells]
	radial_shells=radial_shells_R200+radial_shells_pkpc+radial_shells_rstar
	radial_shells_str=radial_shells_R200_str+radial_shells_pkpc_str+radial_shells_rstar_str

	# Loop over all the spherical shells
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
			rshell_maxidx=np.searchsorted(rrel,rshell)
			mask_sphere=np.zeros_like(rrel).astype(bool)
			mask_sphere[:rshell_maxidx]=True
			
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
				galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_sphere])
				galaxy_output[f'{rshell_str}_sphere-gas_'+Tstr+f'-Z']=np.nansum(specmass['Z'][Tmask_sphere])/np.nansum(mass[Tmask_sphere])
						
			#### SPHERICAL SHELL CALCULATIONS (r between r-dr/2 and r+dr/2) ####
			# Mask for the shell in comoving coordinates (particle data is in comoving coordinates)
			for drfac,drfac_str in zip(drfacs,drfacs_str):
				rshell_str=rshell_str
				rhi=rshell+(drfac*rshell)/2
				rlo=rshell-(drfac*rshell)/2

				rshell_minidx=np.searchsorted(rrel,rlo)
				rshell_maxidx=np.searchsorted(rrel,rhi)
				mask_shell=np.zeros_like(rrel).astype(bool)
				mask_shell[rshell_minidx:rshell_maxidx]=True

				# Now convert the shell values to physical units for the calculations
				rhi=rhi*afac
				rlo=rlo*afac
				dr=rhi-rlo

				# Shell volume and area in pkpc^3 and pkpc^2
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-vol']=4/3*np.pi*((rhi*1e3)**3-(rlo*1e3)**3)
				galaxy_output[f'{rshell_str}_shell{drfac_str}_full-area']=4*np.pi*((rhi*1e3)**2-(rlo*1e3)**2)

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
				for theta_str,thetamask in thetamasks.items():
					mask_shell_theta=np.logical_and(mask_shell,thetamask)

					# Break down the gas mass by phase
					for Tstr,Tmask in Tmasks.items():
						Tmask_shell=np.logical_and.reduce([mask_shell_theta,gas,Tmask])
						galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_shell)
						
						# If considering a galaxy-scale shell, calculate the SFR and metallicity
						galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-SFR']=np.nansum(sfr[Tmask_shell])
						galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-Z']=np.nansum(specmass['Z'][Tmask_shell])/np.nansum(mass[Tmask_shell])

						# Breakdown of mass in this phase by species
						if Tstr=='all':
							for spec in specmass.keys():
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-m_{spec}']=np.nansum(specmass[spec][Tmask_shell])
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-vrad_{spec}_mean']=np.nansum(vrad[Tmask_shell]*specmass[spec][Tmask_shell])/np.nansum(specmass[spec][Tmask_shell])
								outmask=np.logical_and(Tmask_shell,vrad>0)
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-vradout_{spec}_mean']=np.nansum(vrad[outmask]*specmass[spec][outmask])/np.nansum(specmass[spec][outmask])
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-vradout_{spec}_50P']=np.float64(weighted_nanpercentile(vrad[outmask],specmass[spec][outmask],50))
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-vradout_{spec}_90P']=np.float64(weighted_nanpercentile(vrad[outmask],specmass[spec][outmask],90))
							
						# Calculate the total flow rates for the gas
						for vboundary, vkey in zip(vsboundary, vsboundary_str):
							# Retrieve vmins -- mask the vmin values which are arrays
							vmins_use=[]
							for iv,vminstr in enumerate(vminstrs):
								if type(vmins[iv])==np.ndarray:
									vmins_use.append(vmins[iv][Tmask_shell])
								else:
									vmins_use.append(vmins[iv])
							
							gas_flow_rates=calculate_flow_rate(masses=mass[Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins_use)
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_tot_inflow_{vkey}_vc000kmps']=gas_flow_rates[0]
							galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_vc000kmps']=gas_flow_rates[1]
							for iv,vminstr in enumerate(vminstrs):
								galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_{vminstr}']=gas_flow_rates[2+iv]


							# Calculate the flow rates for the gas by species
							if Tstr=='all':
								for spec in specmass.keys():
									gas_flow_rates_species=calculate_flow_rate(masses=specmass[spec][Tmask_shell],vrad=vrad[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins_use)
									galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_{spec}_inflow_{vkey}_vc000kmps']=gas_flow_rates_species[0]
									galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_vc000kmps']=gas_flow_rates_species[1]
									for iv,vminstr in enumerate(vminstrs):
										galaxy_output[f'{rshell_str}_shell{drfac_str}_{theta_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_{vminstr}']=gas_flow_rates_species[2+iv]

	#### CYLINDRICAL SLAB CALCULATIONS (abs[z] between r-dr/2 and r+dr/2) ####
	# Mask for the shell in comoving coordinates (particle data is in comoving coordinates)
	for rshell,rshell_str in zip(radial_shells,radial_shells_str):
		# Only do for kpc, rstar and 0.10r200 shells
		flag_innershell=('kpc' in rshell_str) or ('0p10r200' in rshell_str) or ('0p30r200' in rshell_str) or ('1p00r200' in rshell_str) or ('reff' in rshell_str)
		for drfac,drfac_str in zip(drfacs,drfacs_str):
			if flag_innershell:
				rshell_str=rshell_str
				rhi=rshell+(drfac*rshell)/2
				rlo=rshell-(drfac*rshell)/2
				zmask=np.logical_and(np.abs(zheight)>=rlo,np.abs(zheight)<rhi)

				# Iterate over the z-slab max radii
				for rmax_str,rmax in zip(zslab_radii_strs,zslab_radii_vals):
					
					if 'zheight' in rmax_str:
						rmax=rhi

					# Mask for the slab in comoving coordinates
					mask_shell=np.logical_and.reduce([zmask,rrel_inplane<rmax])

					# Now convert the shell values to physical units for the calculations
					rhi=rhi*afac
					rlo=rlo*afac
					dr=rhi-rlo

					### GAS shell properties
					# Break down the gas mass by phase
					for Tstr,Tmask in Tmasks.items():
						Tmask_shell=np.logical_and.reduce([mask_shell,gas,Tmask])
						galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-n_tot']=np.nansum(Tmask_shell)

						# Calculate the total flow rates for the gas
						for vboundary, vkey in zip([0], ['vbstatic']):
							# Retrieve vmins -- mask the vmin values which are arrays
							vmins_use=[]
							for iv,vminstr in enumerate(vminstrs):
								if type(vmins[iv])==np.ndarray:
									vmins_use.append(vmins[iv][Tmask_shell])
								else:
									vmins_use.append(vmins[iv])

							gas_flow_rates=calculate_flow_rate(masses=mass[Tmask_shell],vrad=vradz[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins_use)
							galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_tot_inflow_{vkey}_vc000kmps']=gas_flow_rates[0]
							galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_vc000kmps']=gas_flow_rates[1]
							for iv,vminstr in enumerate(vminstrs):
								galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_tot_outflow_{vkey}_{vminstr}']=gas_flow_rates[2+iv]
							
							# Calculate the flow rates for the gas by species
							if Tstr=='all':
								for spec in specmass.keys():
									gas_flow_rates_species=calculate_flow_rate(masses=specmass[spec][Tmask_shell],vrad=vradz[Tmask_shell],dr=dr,vboundary=vboundary,vmin=vmins_use)
									galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_{spec}_inflow_{vkey}_vc000kmps']=gas_flow_rates_species[0]
									galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_vc000kmps']=gas_flow_rates_species[1]
									for iv,vminstr in enumerate(vminstrs):
										galaxy_output[f'{rshell_str}_zslab{drfac_str}_{rmax_str}-gas_'+Tstr+f'-mdot_{spec}_outflow_{vkey}_{vminstr}']=gas_flow_rates_species[2+iv]


	# Return the galaxy output dictionary
	return galaxy_output


