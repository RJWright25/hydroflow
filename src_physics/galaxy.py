# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routine to characterise a galaxy for output catalogues (including fitting baryonic mass profile).

import numpy as np

def analyse_galaxy(galaxy,pdata,Tcut,r200_shells=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3]):
	"""
	analyse_galaxy: Analyse the static properties of a galaxy, including the mass and properties of its baryonic reservoirs.

	Input:
	-----------
	galaxy: dict or pd.Series
		Dictionary containing the properties of the galaxy, including the virial radius and scale factor.
	pdata: pd.DataFrame
		DataFrame containing the particle data for the relevant galaxy. Should include the particle properties out to the largest radius of interest.
	Tcut: float
		Temperature cut for defining the ISM.
	r200_shells: list
		List of radii at which to calculate the gas mass profile (multiples of R200).

	Output:
	-----------
	success: bool
		True if the function completes successfully.
	
	galaxy_output: dict
		Dictionary containing the properties of the galaxy's baryonic reservoirs. Each key is a string describing the reservoir, and the value is a float or array of floats containing the properties of the reservoir.

	"""

	galaxy_output={}
	galaxy_reservoirs={}
	
	#which properties to analyse
	properties_ptype={0:['Metallicity',
			             'Temperature',
						 'Relative_v_rad',
						 'Relative_v_tan'], 
		   			  4:['Metallicity',
						 'Relative_v_rad',
						 'Relative_v_tan']}
		  
	properties_abbrev={'Metallicity':'Z',
					   'Temperature':'T',
					   'Relative_v_rad':'vrad',
					   'Relative_v_tan':'vtan'}

	hval=galaxy['hval']
	afac=galaxy['afac']

	#remove tracers if present
	if 'Flag_Tracer' in list(pdata.keys()):
		tracer=pdata['Flag_Tracer'].values>0
		pdata=pdata.loc[np.logical_not(tracer),:].copy();pdata.reset_index(drop=True,inplace=True)

	#masks
	gas=pdata['ParticleType'].values==0.
	star=pdata['ParticleType'].values==4.

	if 'StellarFormationTime' in pdata:
		gas=np.logical_or(gas,pdata.StellarFormationTime<0)
		star=np.logical_and(star,pdata.StellarFormationTime>0)
	
	mass=pdata['Mass'].values
	cool=pdata['Temperature'].values<Tcut
	sfr=pdata['StarFormationRate'].values>0
	rrel=pdata['Relative_r'].values
	r200=galaxy['Group_R_Crit200']

	#masking
	r200_mask=rrel<=r200
	galaxy_reservoirs['1p00r200_star']=np.logical_and(star,r200_mask)
	galaxy_reservoirs['1p00r200_gas']=np.logical_and(gas,r200_mask)

	#30ckpc ISM
	c30kpc_mask=rrel<=((30*1e-3)*hval)
	galaxy_reservoirs['030ckpc_star']=np.logical_and(star,c30kpc_mask)
	galaxy_reservoirs['030ckpc_gas']=np.logical_and(gas,c30kpc_mask)
	galaxy_reservoirs['030ckpc_coolgas']=np.logical_and(galaxy_reservoirs['030ckpc_gas'],np.logical_or(cool,sfr))
	galaxy_reservoirs['030ckpc_sfrgas']=np.logical_and(galaxy_reservoirs['030ckpc_gas'],sfr)

	#30pkpc ISM
	p30kpc_mask=rrel<=((30*1e-3)*hval/afac)
	galaxy_reservoirs['030pkpc_star']=np.logical_and(star,p30kpc_mask)
	galaxy_reservoirs['030pkpc_gas']=np.logical_and(gas,p30kpc_mask)
	galaxy_reservoirs['030pkpc_coolgas']=np.logical_and(galaxy_reservoirs['030pkpc_gas'],np.logical_or(cool,sfr))
	galaxy_reservoirs['030pkpc_sfrgas']=np.logical_and(galaxy_reservoirs['030pkpc_gas'],sfr)

	#0p20r200 ISM
	disk=rrel<=(0.20*r200)
	galaxy_reservoirs['0p20r200_star']=np.logical_and(star,disk)
	galaxy_reservoirs['0p20r200_gas']=np.logical_and(gas,disk)
	galaxy_reservoirs['0p20r200_coolgas']=np.logical_and(galaxy_reservoirs['0p20r200_gas'],np.logical_or(cool,sfr))
	galaxy_reservoirs['0p20r200_sfrgas']=np.logical_and(galaxy_reservoirs['0p20r200_gas'],sfr)

	#0p10r200 ISM
	disk=rrel<=(0.10*r200)
	galaxy_reservoirs['0p10r200_star']=np.logical_and(star,disk)
	galaxy_reservoirs['0p10r200_gas']=np.logical_and(gas,disk)
	galaxy_reservoirs['0p10r200_coolgas']=np.logical_and(galaxy_reservoirs['0p10r200_gas'],np.logical_or(cool,sfr))
	galaxy_reservoirs['0p10r200_sfrgas']=np.logical_and(galaxy_reservoirs['0p10r200_gas'],sfr)

	#not in 0p20r200, within r200
	halo=np.logical_and.reduce([np.logical_not(np.logical_or(galaxy_reservoirs['0p20r200_gas'],galaxy_reservoirs['0p20r200_star'])),r200_mask])
	galaxy_reservoirs['0p20r200_halostar']=np.logical_and(star,halo)
	galaxy_reservoirs['0p20r200_halogas']=np.logical_and(gas,halo)

	#not in 0p10r200, within r200
	halo=np.logical_and.reduce([np.logical_not(np.logical_or(galaxy_reservoirs['0p10r200_gas'],galaxy_reservoirs['0p10r200_star'])),r200_mask])
	galaxy_reservoirs['0p10r200_halostar']=np.logical_and(star,halo)
	galaxy_reservoirs['0p10r200_halogas']=np.logical_and(gas,halo)	

	######### PROFILES
	#gas mass profile (r200)
	reservoir_edges=np.concatenate([[0],r200_shells])
	reservoir_names_gas=[f'{fachi:.2f}'.replace('.','p')+'r200_gasprof' for fachi in reservoir_edges[1:]]
	reservoir_masks_gas=[np.logical_and.reduce([gas,rrel>faclo*r200,rrel<=fachi*r200]) for faclo,fachi in zip(reservoir_edges[:-1],reservoir_edges[1:])]

	for name, mask in zip(reservoir_names_gas,reservoir_masks_gas):
		galaxy_reservoirs[name]=mask

    #Calculate average properties of each reservoir
	for reservoir in galaxy_reservoirs:
		mask=galaxy_reservoirs[reservoir]
		partmass=mass[np.where(mask)]
		galaxy_output[f'{reservoir}-n']=np.nansum(mask)
		galaxy_output[f'{reservoir}-m']=np.nansum(partmass)

		if 'star' in reservoir:
			reservoir_props=properties_ptype[4]
		elif 'gas' in reservoir:
			reservoir_props=properties_ptype[0]
		else:
			reservoir_props=properties_ptype[1]
		
		if len(partmass)>0:
			for prop in reservoir_props:
				partprop=pdata.loc[mask,prop].values
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_median']=np.nanmedian(partprop)
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_mean']=np.average(partprop,weights=partmass)
			if 'coolgas' in reservoir or 'prof' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=np.nansum(pdata.loc[mask,'StarFormationRate'].values)
		else:
			for prop in reservoir_props:
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_median']=np.nan
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_mean']=np.nan
			if 'coolgas' in reservoir or 'prof' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=0

	return True, galaxy_output

