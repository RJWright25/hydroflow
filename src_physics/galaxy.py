# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routine to characterise a galaxy for output catalogues (including fitting baryonic mass profile).

import numpy as np

from hydroflow.src_physics.utils import calc_r200

def analyse_galaxy(galaxy,pdata):
	galaxy_output={}
	galaxy_reservoirs={}

	#which properties to analyse
	properties_ptype={0:['Metallicity',
			             'Temperature',
			             'R_rel'], 
					  1:['R_rel'],
		   			  4:['Metallicity',
					  	 'R_rel']}
		  
	properties_abbrev={'Metallicity':'Z',
					   'Temperature':'T',
					   'R_rel':'R'}

	#masks
	gas=pdata['ParticleType'].values==0
	star=pdata['ParticleType'].values==4
	dm=pdata['ParticleType'].values==1
	mass=pdata['Mass'].values
	cool=pdata['Temperature'].values<5*10**4
	sfr=pdata['StarFormationRate'].values>0
	rrel=pdata['R_rel'].values

	#within r200
	r200=rrel<=calc_r200(galaxy)
	galaxy_reservoirs['1p00r200_star']=np.logical_and(star,r200)
	galaxy_reservoirs['1p00r200_gas']=np.logical_and(gas,r200)
	galaxy_reservoirs['1p00r200_dm']=np.logical_and(dm,r200)

	#within ISM
	disk=rrel<=(0.15*calc_r200(galaxy))
	galaxy_reservoirs['0p15r200_star']=np.logical_and(star,disk)
	galaxy_reservoirs['0p15r200_gas']=np.logical_and(gas,disk)
	galaxy_reservoirs['0p15r200_coolgas']=np.logical_and(galaxy_reservoirs['0p15r200_gas'],np.logical_or(cool,sfr))

	#not in ism, within r200
	halo=np.logical_and.reduce([np.logical_not(np.logical_or(galaxy_reservoirs['0p15r200_coolgas'],galaxy_reservoirs['0p15r200_star'])),r200])
	galaxy_reservoirs['0p15r200_halostar']=np.logical_and(star,halo)
	galaxy_reservoirs['0p15r200_halogas']=np.logical_and(gas,halo)

    #metallicity, sfr, temperature
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
			if 'cool' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=np.nansum(pdata.loc[mask,'StarFormationRate'].values)
		else:
			for prop in reservoir_props:
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_median']=np.nan
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_mean']=np.nan
			if 'cool' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=0

	return True, galaxy_output

