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

	if not ('StarFormationRate' in pdata):
		pdata.loc[:,'StarFormationRate']=np.nan

	#masks
	gas=pdata['ParticleType'].values==0.
	star=pdata['ParticleType'].values==4.
	dm=pdata['ParticleType'].values==1.

	if 'StellarFormationTime' in pdata:
		gas=np.logical_or(gas,pdata.StellarFormationTime<0)
		star=np.logical_and(star,pdata.StellarFormationTime>0)
	
	mass=pdata['Mass'].values
	cool=pdata['Temperature'].values<5*10**4
	sfr=pdata['StarFormationRate'].values>0
	Zmet=pdata['Metallicity'].values>0
	rrel=pdata['R_rel'].values

	#within 
	r200=calc_r200(galaxy)
	r200_mask=rrel<=r200
	galaxy_reservoirs['1p00r200_star']=np.logical_and(star,r200_mask)
	galaxy_reservoirs['1p00r200_gas']=np.logical_and(gas,r200_mask)
	galaxy_reservoirs['1p00r200_dm']=np.logical_and(dm,r200_mask)

	#within ISM
	disk=rrel<=(0.15*r200)
	galaxy_reservoirs['0p15r200_star']=np.logical_and(star,disk)
	galaxy_reservoirs['0p15r200_gas']=np.logical_and(gas,disk)
	galaxy_reservoirs['0p15r200_coolgas']=np.logical_and(galaxy_reservoirs['0p15r200_gas'],np.logical_or(cool,sfr))

	#not in ism, within r200
	halo=np.logical_and.reduce([np.logical_not(np.logical_or(galaxy_reservoirs['0p15r200_coolgas'],galaxy_reservoirs['0p15r200_star'])),r200_mask])
	galaxy_reservoirs['0p15r200_halostar']=np.logical_and(star,halo)
	galaxy_reservoirs['0p15r200_halogas']=np.logical_and(gas,halo)

	### mass profiles
	reservoir_edges=[0.00,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0]
	reservoir_names=[f'{fachi:.2f}'.replace('.','p')+'r200_gasprof' for fachi in reservoir_edges[1:]]
	reservoir_masks=[np.logical_and.reduce([gas,rrel>faclo*r200,rrel<=fachi*r200]) for faclo,fachi in zip(reservoir_edges[:-1],reservoir_edges[1:])]

	for name, mask in zip(reservoir_names,reservoir_masks):
		galaxy_reservoirs[name]=mask

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

