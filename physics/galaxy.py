import sys
sys.path.append('/home/rwright/Software')

import numpy as np
import pandas as pd

from hydroflow.physics.math import weighted_median,r200_eff,bary_mp

def find_progidx(subcat,galid,depth):
	fields=list(subcat.columns)
	if 'nodeIndex' in fields:
		galid_key='nodeIndex'
		progid_key='mainProgenitorIndex'
		descid_key='descendantIndex'
	else:#eagle snaps
		galid_key='GalaxyID'
		progid_key='LastProgID'
		descid_key='DescendantID'

	galid_idepth=galid
	nmerger_min=0;nmerger_maj=0
	for idepth in range(depth):
		match_idepth=subcat[galid_key].values==galid_idepth
		try:
			mainprogen_id=subcat.loc[match_idepth,progid_key].values[0]
		except:
			return 0,0,0

		match_idepth_progen=subcat[descid_key].values==galid_idepth
		numprogen=np.nansum(match_idepth_progen)        
		if numprogen>1:
			match_idepth_masses=subcat.loc[match_idepth_progen,'ApertureMeasurements/Mass/030kpc'].values
			mratio=10**(np.abs(np.log10(match_idepth_masses[0]/match_idepth_masses[1])))
			if mratio<=5:
				nmerger_maj+=1
			elif mratio>5:
				nmerger_min+=1

		galid_idepth=mainprogen_id
	return nmerger_min,nmerger_maj,galid_idepth

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
	r200=r200_eff(galaxy);galaxy_output['r200_eff']=r200

	baryons=np.logical_or.reduce([star,cool,sfr])
	baryons=np.logical_and.reduce([baryons,sgm,pdata['R_rel'].values<r200])
	rrel_galaxy=rrel[baryons]/r200
	mass_galaxy=mass[baryons]

	mprofile=[np.nansum(mass_galaxy[mask]) for mask in [rrel_galaxy<bin_hi for bin_hi in r_bins[1:]]]
	mprofile=mprofile/mprofile[-1]

	try:
		barymp_fac,nfit=bary_mp(r_bins[1:],mprofile)
		galaxy_output['bmp_factor']=barymp_fac
		galaxy_output['bmp_radius']=barymp_fac*r200
		galaxy_output['bmp_nfit']=nfit
	except:
		return False, None

	##### OTHER PROPERTIES
	#within r200
	r200=rrel<=r200
	reservoirs['r200_star']=np.logical_and(star,r200)
	reservoirs['r200_gas']=np.logical_and(gas,r200)

	#within barymp
	bmp=rrel<=barymp_fac*r200
	reservoirs['bmp_star']=np.logical_and(star,bmp)
	reservoirs['bmp_gas']=np.logical_and(gas,bmp)
	reservoirs['bmp_ism']=np.logical_and(reservoirs['bmp_gas'],np.logical_or(cool,sfr))

	#not in ism, within r200
	nbmp=np.logical_and(np.logical_not(np.logical_or(reservoirs['bmp_ism'],reservoirs['bmp_star'])),rrel<=r200)
	reservoirs['cgm_star']=np.logical_and(star,nbmp)
	reservoirs['cgm_gas']=np.logical_and(gas,nbmp)

    #metallicity, sfr, temperature
	for reservoir,reservoir_mask in reservoirs.items():
		galaxy_output[f'{reservoir}-n']=np.nansum(reservoir_mask)
		galaxy_output[f'{reservoir}-m']=np.nansum(mass[reservoir_mask])

		if 'star' in reservoir:
			reservoir_props=props[4]
		else:
			reservoir_props=props[0]
		
		partmass=mass[reservoir_mask]
		if len(partmass)>0:
			for prop in reservoir_props:
				partprop=pdata.loc[reservoir_mask,prop].values
				galaxy_output[f'{reservoir}-{abbrev[prop]}_median']=weighted_median(partprop,weights=partmass)
				galaxy_output[f'{reservoir}-{abbrev[prop]}_mean']=np.average(partprop,weights=partmass)
			if reservoir=='bmp_ism':
				galaxy_output[f'bmp_ism-SFR']=np.nansum(pdata.loc[reservoirs['bmp_ism'],'StarFormationRate'].values)
		else:
			for prop in reservoir_props:
				galaxy_output[f'{reservoir}-{abbrev[prop]}_median']=np.nan
				galaxy_output[f'{reservoir}-{abbrev[prop]}_mean']=np.nan

	return True, pd.Series(galaxy_output)