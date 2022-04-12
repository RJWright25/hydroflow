# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routine to characterise a galaxy for output catalogues (including fitting baryonic mass profile).

import numpy as np

from hydroflow.src_physics.utils import calc_r200,calc_barymp

def analyse_galaxy(galaxy,pdata):
	reservoirs={}
	galaxy_output={}

	props={0:['Metallicity','Temperature','R_rel'], 4:['Metallicity','R_rel']}
	abbrev={'Metallicity':'Z','Temperature':'T','R_rel':'R'}

	z=galaxy['redshift']
	sgn=galaxy['SubGroupNumber']

	cool=pdata['Temperature'].values<5*10**4
	sfr=pdata['StarFormationRate'].values>0
	gas=pdata['ParticleType'].values==0
	star=pdata['ParticleType'].values==4
	sgm=pdata['SubGroupNumber'].values==sgn
	mass=pdata['Mass'].values
	rrel=pdata['R_rel'].values

	##### BARYMP
	r_bins=np.linspace(0,1,num=101)
	r200=calc_r200(galaxy);galaxy_output['r200_eff']=r200

	baryon_candidates=np.logical_or.reduce([star,cool,sfr])
	baryons=np.logical_and.reduce([baryon_candidates,sgm,pdata['R_rel'].values<r200])
	rrel_galaxy=rrel[baryons]/r200
	mass_galaxy=mass[baryons]

	mprofile=[np.nansum(mass_galaxy[mask]) for mask in [rrel_galaxy<bin_hi for bin_hi in r_bins[1:]]]
	mprofile=mprofile/mprofile[-1]

	try:
		barymp_fac,nfit=calc_barymp(r_bins[1:],mprofile)
		galaxy_output['bmp_factor']=barymp_fac
		galaxy_output['bmp_radius']=barymp_fac*r200
		galaxy_output['bmp_nfit']=nfit
	except:
		return False, None

	##### OTHER PROPERTIES
	
	#within r200
	r200=np.logical_and(rrel<=r200,sgm)

	reservoirs['r200_star']=np.logical_and(star,r200)
	reservoirs['r200_gas']=np.logical_and(gas,r200)

	#within barymp
	bmp=np.logical_and(rrel<=barymp_fac*r200,sgm)
	reservoirs['bmp_star']=np.logical_and(star,bmp)
	reservoirs['bmp_gas']=np.logical_and(gas,bmp)
	reservoirs['bmp_ism']=np.logical_and(reservoirs['bmp_gas'],np.logical_or(cool,sfr))

	#not in ism, within r200
	nbmp=np.logical_and.reduce([np.logical_not(np.logical_or(reservoirs['bmp_ism'],reservoirs['bmp_star'])),rrel<=r200,sgm])
	reservoirs['cgm_star']=np.logical_and(star,nbmp)
	reservoirs['cgm_gas']=np.logical_and(gas,nbmp)

    #metallicity, sfr, temperature
	for reservoir in reservoirs:
		mask=reservoirs[reservoir]
		partmass=mass[np.where(mask)]
		galaxy_output[f'{reservoir}-n']=np.nansum(mask)
		galaxy_output[f'{reservoir}-m']=np.nansum(partmass)

		if 'star' in reservoir:
			reservoir_props=props[4]
		else:
			reservoir_props=props[0]
		
		if len(partmass)>0:
			for prop in reservoir_props:
				partprop=pdata.loc[mask,prop].values
				galaxy_output[f'{reservoir}-{abbrev[prop]}_median']=np.nanmedian(partprop)
				galaxy_output[f'{reservoir}-{abbrev[prop]}_mean']=np.average(partprop,weights=partmass)
			if reservoir=='bmp_ism':
				galaxy_output[f'bmp_ism-SFR']=np.nansum(pdata.loc[mask,'StarFormationRate'].values)
		else:
			for prop in reservoir_props:
				galaxy_output[f'{reservoir}-{abbrev[prop]}_median']=np.nan
				galaxy_output[f'{reservoir}-{abbrev[prop]}_mean']=np.nan
			if reservoir=='bmp_ism':
				galaxy_output[f'bmp_ism-SFR']=0

	return True, galaxy_output