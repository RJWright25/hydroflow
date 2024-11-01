# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routines to analyse a galaxy (in shells and ) for output catalogues. 
import numpy as np
import pandas as pd

from hydroflow.src_physics.utils import constant_G
from hydroflow.src_physics.utils import MpcpGyr_to_kmps

def retrieve_galaxy_candidates(galaxy,pdata_subvol,kdtree_subvol,maxrad=None): 

	print('Getting candidates from KDtree')

	# Define the centre of mass and scale factor
	com=np.array([galaxy[f'CentreOfPotential_{x}'] for x in 'xyz'])
	afac=1/(1+galaxy['Redshift'])

	# Define the maximum radius to search for candidates if not provided
	if maxrad is None:
		maxrad=galaxy['Group_R_Crit200']*3.5

	# Retrieve the candidates from the KDTree
	pidx_candidates_snapf=kdtree_subvol.query_ball_point(com,maxrad)
	pdata_candidates=pdata_subvol.loc[pidx_candidates_snapf,:]
	pdata_candidates.reset_index(drop=True,inplace=True)
	numcdt=pdata_candidates.shape[0]

	# Get derived quantities if there are elements within the radius
	if numcdt>0:
		# Compute relative position
		pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=(pdata_candidates.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-com)
		pdata_candidates['Relative_r_comoving']=np.linalg.norm(pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values,axis=1)
		pdata_candidates['Relative_r_physical']=pdata_candidates['Relative_r_comoving']*afac

		# Using the mean velocity of the halo (WITHIN R200, all particle types) to calculate relative velocities
		vmask=pdata_candidates['Relative_r_comoving'].values<galaxy['Group_R_Crit200']
		vcom=np.array([np.nanmean(pdata_candidates[f'Velocities_{x}'].values[vmask]) for x in 'xyz'])
		
		# These velocities are all a*dx/dt (peculiar velocity)
		for idim,dim in enumerate('xyz'):
			pdata_candidates[f'Relative_v_{dim}']=pdata_candidates[f'Velocities_{dim}'].values-vcom[idim]
		
		# Magnitude of the relative velocity
		pdata_candidates[f'Relative_v_abs_pec']=np.sqrt(np.nansum(np.square(pdata_candidates.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values),axis=1))

		# Radial velocity is the dot product of the unit vector relative position and velocity (units of radial position don't matter, only the unit vector)
		pdata_candidates[f'Relative_v_rad_pec']=np.nansum(pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values*pdata_candidates.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values,axis=1)/pdata_candidates['Relative_r_comoving'].values
 
		# Define the angular momentum of the galaxy with baryonic elements within 30ckpc
		Lbarmask=np.logical_or(pdata_candidates['ParticleType'].values==0,pdata_candidates['ParticleType'].values==4)
		Lbarmask=np.logical_and(Lbarmask,pdata_candidates['Relative_r_comoving'].values<30*1e-3/afac) 
		Lbarspec=np.cross(pdata_candidates.loc[Lbarmask,[f'Relative_{x}_comoving' for x in 'xyz']].values*afac,pdata_candidates.loc[Lbarmask,[f'Relative_v_{x}' for x in 'xyz']].values)
		Lbartot=Lbarspec*pdata_candidates.loc[Lbarmask,'Masses'].values[:,np.newaxis]
		Lbartot=np.nansum(Lbartot,axis=0)#this uses comoving coordinates but only the unit vector is needed not the magnitude

		# Find the angle between the angular momentum of the galaxy and the position vector of each particle
		position = pdata_candidates.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values
		cos_theta=np.sum(Lbartot*position,axis=1)/(np.linalg.norm(Lbartot)*np.linalg.norm(position,axis=1))
		deg_theta=np.arccos(cos_theta)*180/np.pi
		deg_theta[deg_theta>90]=180-deg_theta[deg_theta>90]
		pdata_candidates['Relative_phi']=deg_theta

		# Save the angular momentum and velocity of the galaxy
		pdata_candidates.attrs['Lbartot']=Lbartot
		pdata_candidates.attrs['vcom']=vcom

		return pdata_candidates
	else:
		return pd.DataFrame()


# Main function to analyse the galaxy and surrounding baryonic reservoirs

def analyse_galaxy(galaxy,pdata_candidates,metadata,r200_shells=None,ckpc_shells=None,Tbins=None,vcuts=None,drfac=None,logfile=None):

	"""
	analyse_galaxy: Analyse the static properties of a galaxy, including the mass and properties of its baryonic reservoirs.

	Input:
	-----------
	galaxy: dict or pd.Series
		Dictionary containing the properties of the galaxy, including the virial radius and scale factor.
	pdata: pd.DataFrame
		DataFrame containing the particle data for the relevant galaxy. Should include the particle properties out to the largest radius of interest.
	r200_shells: list
		List of radii at which to calculate properties (multiples of R200).
	ckpc_shells: list
		List of radii at which to calculate properties (in ckpc).
	Tbins: list
		Dict of temperature bins to use for gas properties.
	vcuts: dict
		Dictionary containing the velocity cuts to use for inflows/outflows. Psuedo-R200 evolution added by default.
	drfac: float
		Fractional width of the shell.
	

	Output:
	-----------
	
	galaxy_output: dict
		Dictionary containing the properties of the galaxy's baryonic reservoirs. Each key is a string describing the reservoir, and the value is a float or array of floats containing the properties of the reservoir.

	"""	

	galaxy_output={}

	# Retrieve computed quantities from candidates
	galaxy_output['vcom_x']=pdata_candidates.attrs['vcom'][0]
	galaxy_output['vcom_y']=pdata_candidates.attrs['vcom'][1]
	galaxy_output['vcom_z']=pdata_candidates.attrs['vcom'][2]
	galaxy_output['Lbartot_x']=pdata_candidates.attrs['Lbartot'][0]
	galaxy_output['Lbartot_y']=pdata_candidates.attrs['Lbartot'][1]
	galaxy_output['Lbartot_z']=pdata_candidates.attrs['Lbartot'][2]

	# Add existing galaxy properties
	for key in galaxy.keys():
		galaxy_output[key]=galaxy[key]

	# Masks
	gas=pdata_candidates['ParticleType'].values==0.
	star=pdata_candidates['ParticleType'].values==4.
	dm=pdata_candidates['ParticleType'].values==1.

	# Fields
	mass=pdata_candidates['Masses'].values
	rrel=pdata_candidates['Relative_r_comoving'].values
	vrad={'pec':pdata_candidates['Relative_v_rad_pec'].values}
	# 'hub':pdata_candidates['Relative_v_rad_hub'].values}
	
	# Gas properties
	temp=pdata_candidates['Temperature'].values
	sfr=pdata_candidates['StarFormationRate'].values
	specfrac={}
	if 'mfrac_HI' in pdata_candidates.columns:
		specfrac['HI']=pdata_candidates['mfrac_HI'].values
	if 'mfrac_HII' in pdata_candidates.columns:
		specfrac['HII']=pdata_candidates['mfrac_HII'].values
	if 'mfrac_H2' in pdata_candidates.columns:
		specfrac['H2']=pdata_candidates['mfrac_H2'].values
	if 'Metallicity' in pdata_candidates.columns:
		specfrac['Z']=pdata_candidates['Metallicity'].values
	specfrac['tot']=np.ones_like(gas)

	# Gas selections by temperature
	Tmasks={'all':gas}
	if Tbins is not None:
		for Tstr,Tbin in Tbins.items():
			Tmasks[Tstr]=np.logical_and.reduce([gas,temp>Tbin[0],temp<Tbin[1]])

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

		#pseudo-evolution velocity cut (updated for each shell)
		vcuts['pdoev']=vpseudo*(rshell/galaxy['Group_R_Crit200'])

		# Skip the shell if it is a multiple of r200 and the galaxy is a satellie
		if not ('r200' in rshell_str and galaxy['SubGroupNumber']>0):
			
			#### SPHERE CALCULATIONS (r<rshell) ####
			# Mask for the sphere in comoving coordinates
			mask_sphere=rrel<=rshell

			# DARK MATTER
			galaxy_output[f'{rshell_str}_sphere-dm-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,dm)])
			galaxy_output[f'{rshell_str}_sphere-dm-n_tot']=np.nansum(np.logical_and(mask_sphere,dm))

			# STARS
			galaxy_output[f'{rshell_str}_sphere-star-m_tot']=np.nansum(mass[np.logical_and(mask_sphere,star)])
			galaxy_output[f'{rshell_str}_sphere-star-n_tot']=np.nansum(np.logical_and(mask_sphere,star))
			galaxy_output[f'{rshell_str}_sphere-star-Z']=np.nansum(specfrac['Z'][np.logical_and(mask_sphere,star)]*mass[np.logical_and(mask_sphere,star)])/np.nansum(mass[np.logical_and(mask_sphere,star)])

			# GAS
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

			# Mask for the shell in comoving coordinates
			r_hi=(rshell*(1+drfac/2))
			r_lo=(rshell*(1-drfac/2))
			mask_shell=np.logical_and(rrel<r_hi,rrel>=r_lo)

			# Convert to physical units for the calculations
			r_hi=r_hi*afac
			r_lo=r_lo*afac
			dr=r_hi-r_lo

			# Shell volume and area
			galaxy_output[f'{rshell_str}_shell-vol']=4/3*np.pi*((r_hi*1e3)**3-(r_lo*1e3)**3)
			galaxy_output[f'{rshell_str}_shell-area']=4*np.pi*((r_hi*1e3)**2-(r_lo*1e3)**2)

			# DM shell properties
			dm_shell_mask=np.logical_and(mask_shell,dm)
			galaxy_output[f'{rshell_str}_shell-dm-m_tot']=np.nansum(mass[dm_shell_mask])
			galaxy_output[f'{rshell_str}_shell-dm-n_tot']=np.nansum(dm_shell_mask)

			# DM flow rates
			for flowsign,sign in zip(['inflow','outflow'],[-1,1]):
				for flowtype in ['pec']:
					for vcut_str,vcut in vcuts.items():
						flowstr=f'{flowsign}_{vcut_str}_{flowtype}'
						if flowsign=='inflow':
							vrad_mask=vrad[flowtype][dm_shell_mask]<vcut
							vrad_vals=vrad[flowtype][dm_shell_mask][vrad_mask]
						else:
							vrad_mask=vrad[flowtype][dm_shell_mask]>vcut
							vrad_vals=vrad[flowtype][dm_shell_mask][vrad_mask]

						if vcut_str=='pdoev':
							vrad_vals+=-vcut

						galaxy_output[f'{rshell_str}_shell-dm-mdot_tot_{flowstr}']=1/dr*np.nansum(mass[dm_shell_mask][vrad_mask]*vrad_vals/MpcpGyr_to_kmps)*sign

			# STARS shell properties
			stars_shell_mask=np.logical_and(mask_shell,star)
			galaxy_output[f'{rshell_str}_shell-star-m_tot']=np.nansum(mass[stars_shell_mask])
			galaxy_output[f'{rshell_str}_shell-star-n_tot']=np.nansum(stars_shell_mask)
			galaxy_output[f'{rshell_str}_shell-star-Z']=np.nansum(specfrac['Z'][stars_shell_mask]*mass[stars_shell_mask])/np.nansum(mass[stars_shell_mask])

			# STARS flow rates
			for flowsign,sign in zip(['inflow','outflow'],[-1,1]):
				for flowtype in ['pec']:
					for vcut_str,vcut in vcuts.items():
						flowstr=f'{flowsign}_{vcut_str}_{flowtype}'
						if flowsign=='inflow':
							vrad_mask=vrad[flowtype][stars_shell_mask]<vcut
							vrad_vals=vrad[flowtype][stars_shell_mask][vrad_mask]
						else:
							vrad_mask=vrad[flowtype][stars_shell_mask]>vcut
							vrad_vals=vrad[flowtype][stars_shell_mask][vrad_mask]

						if vcut_str=='pdoev':
							vrad_vals+=-vcut

						galaxy_output[f'{rshell_str}_shell-star-mdot_tot_{flowstr}']=1/dr*np.nansum(mass[stars_shell_mask][vrad_mask]*vrad_vals/MpcpGyr_to_kmps)*sign

			# GAS shell properties
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


				# GAS flow rates		
				for flowsign,sign in zip(['inflow','outflow'],[-1,1]):
					for flowtype in ['pec']:
						for vcut_str,vcut in vcuts.items():
							flowstr=f'{flowsign}_{vcut_str}_{flowtype}'
							if flowsign=='inflow':
								vrad_mask=vrad[flowtype][Tmask_shell]<vcut
								vrad_vals=vrad[flowtype][Tmask_shell][vrad_mask]
							else:
								vrad_mask=vrad[flowtype][Tmask_shell]>vcut
								vrad_vals=vrad[flowtype][Tmask_shell][vrad_mask]
							if vcut_str=='pdoev':
								vrad_vals+=-vcut
								
							for spec in specfrac.keys():
								galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-mdot_{spec}_{flowstr}']=1/dr*np.nansum(mass[Tmask_shell][vrad_mask]*specfrac[spec][Tmask_shell][vrad_mask]*vrad_vals/MpcpGyr_to_kmps)*sign
								if spec=='tot' and '000' in vcut_str:
									galaxy_output[f'{rshell_str}_shell-gas_'+Tstr+f'-vrad_{spec}_{flowstr}']=np.nanmean(vrad_vals)
			else:
				pass

	#write to log
	return galaxy_output


