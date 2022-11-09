# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/galaxy.py: routine to characterise a galaxy for output catalogues (including fitting baryonic mass profile).

from telnetlib import DM
import numpy as np

from hydroflow.src_physics.utils import calc_r200

def analyse_galaxy(galaxy,pdata,hval=0.67):
	galaxy_output={}
	galaxy_reservoirs={}
	galaxy_reservoirs_vol={}
	
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
	cool=pdata['Temperature'].values<5*10**4
	sfr=pdata['StarFormationRate'].values>0
	rrel=pdata['Relative_r'].values
	r200=galaxy['Group_R_Crit200']

	#masking
	r200_mask=rrel<=r200
	galaxy_reservoirs['1p00r200_star']=np.logical_and(star,r200_mask)
	galaxy_reservoirs['1p00r200_gas']=np.logical_and(gas,r200_mask)

	c30kpc_mask=rrel<=((30*1e-3)*hval)
	galaxy_reservoirs['030ckpc_star']=np.logical_and(star,c30kpc_mask)
	galaxy_reservoirs['030ckpc_gas']=np.logical_and(gas,c30kpc_mask)

	#within ISM
	disk=rrel<=(0.20*r200)
	galaxy_reservoirs['0p20r200_star']=np.logical_and(star,disk)
	galaxy_reservoirs['0p20r200_gas']=np.logical_and(gas,disk)
	galaxy_reservoirs['0p20r200_coolgas']=np.logical_and(galaxy_reservoirs['0p20r200_gas'],np.logical_or(cool,sfr))
	galaxy_reservoirs['0p20r200_sfrgas']=np.logical_and(galaxy_reservoirs['0p20r200_gas'],sfr)

	#not in 0p15r200, within r200
	halo=np.logical_and.reduce([np.logical_not(np.logical_or(galaxy_reservoirs['0p20r200_gas'],galaxy_reservoirs['0p20r200_star'])),r200_mask])
	galaxy_reservoirs['0p20r200_halostar']=np.logical_and(star,halo)
	galaxy_reservoirs['0p20r200_halogas']=np.logical_and(gas,halo)

	######### PROFILES
	#gas mass profile (r200)
	reservoir_edges=np.concatenate([[0,0.02,0.04,0.06,0.08],np.linspace(0.1,1,10),np.linspace(1.2,2,5),np.linspace(2.2,2,5)])
	reservoir_names_gas=[f'{fachi:.2f}'.replace('.','p')+'r200_gasprof' for fachi in reservoir_edges[1:]]
	reservoir_masks_gas=[np.logical_and.reduce([gas,rrel>faclo*r200,rrel<=fachi*r200]) for faclo,fachi in zip(reservoir_edges[:-1],reservoir_edges[1:])]
	reservoir_volume={name:4/3*np.pi*((fachi*r200/hval)**3-(faclo*r200/hval)**3) for name,faclo,fachi in zip(reservoir_names_gas,reservoir_edges[:-1],reservoir_edges[1:])}

	for name, mask in zip(reservoir_names_gas,reservoir_masks_gas):
		galaxy_reservoirs[name]=mask
		galaxy_reservoirs_vol[name]=reservoir_volume[name]


	#gas mass profile (ckpc)
	reservoir_edges=np.concatenate([np.linspace(0,25,6),np.linspace(30,100,8)])
	reservoir_names_gas=[f'{str(fachi).zfill(3)}'.replace('.','p')+'ckpc_gasprof' for fachi in reservoir_edges[1:]]
	reservoir_masks_gas=[np.logical_and.reduce([gas,rrel>(faclo*hval*1e-3),rrel<=(fachi*hval*1e-3)]) for faclo,fachi in zip(reservoir_edges[:-1],reservoir_edges[1:])]
	reservoir_volume={name:4/3*np.pi*((fachi*1e-3)**3-(faclo*1e-3)**3) for name,faclo,fachi in zip(reservoir_names_gas,reservoir_edges[:-1],reservoir_edges[1:])}

	for name, mask in zip(reservoir_names_gas,reservoir_masks_gas):
		galaxy_reservoirs[name]=mask
		galaxy_reservoirs_vol[name]=reservoir_volume[name]

    #Calculate average properties of each reservoir
	for reservoir in galaxy_reservoirs:
		mask=galaxy_reservoirs[reservoir]
		partmass=mass[np.where(mask)]
		galaxy_output[f'{reservoir}-n']=np.nansum(mask)
		galaxy_output[f'{reservoir}-m']=np.nansum(partmass)

		if 'prof' in reservoir:
			galaxy_output[f'{reservoir}-vol']=galaxy_reservoirs_vol[reservoir]
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
			if ('kpc' in reservoir or 'coolgas' in reservoir) and 'gas' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=np.nansum(pdata.loc[mask,'StarFormationRate'].values)
		else:
			for prop in reservoir_props:
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_median']=np.nan
				galaxy_output[f'{reservoir}-{properties_abbrev[prop]}_mean']=np.nan
			if ('kpc' in reservoir or 'coolgas' in reservoir) and 'gas' in reservoir:
				galaxy_output[f'{reservoir}-SFR']=0

	return True, galaxy_output

