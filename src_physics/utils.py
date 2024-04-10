# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: lower level mathematical functions for the repository.

import numpy as np
from astropy import units
from astropy.cosmology import Planck13 as cosmology

def ivol_gen(ix,iy,iz,nslice):

	"""
	ivol_gen: Generate a unique subvolume index from the subvolume coordinates. This is used to split the simulation volume into subvolumes for parallel processing.

	Input:
	-----------
	ix: int
		x-coordinate of the subvolume.
	iy: int
		y-coordinate of the subvolume.
	iz: int
		z-coordinate of the subvolume.
	nslice: int
		Number of subvolumes in each dimension.
	
	Output:
	-----------
	ivol_str: str
		String containing the subvolume index. Ranges from (from 0-nslice**3)

	"""
    
	ivol=ix*nslice**2+iy*nslice+iz
	ivol_str=str(ivol).zfill(3)
	return ivol_str

def ivol_idx(ivol,nslice):
	
	"""
	ivol_idx: Generate the subvolume coordinates from the unique subvolume index.

	Input:
	-----------
	ivol: int
		Unique subvolume index.
	nslice: int
		Number of subvolumes in each dimension.
	
	Output:
	-----------
	ix: int
		x-coordinate of the subvolume.
	iy: int
		y-coordinate of the subvolume.
	iz: int
		z-coordinate of the subvolume.
	"""

	if type(ivol)==str:
		ivol=int(ivol)
	ix=int(np.floor(ivol/nslice**2))
	iz=int(ivol%nslice)
	iy=int((ivol-ix*nslice**2-iz)/nslice)
	return (ix,iy,iz)


def get_limits(ivol,nslice,boxsize,buffer=0.2):
	"""
	get_limits: Generate the limits of a subvolume in the simulation volume.

	Input:
	-----------
	ivol: int
		Unique subvolume index.
	nslice: int
		Number of subvolumes in each dimension.
	boxsize: float
		Size of the simulation volume.
	buffer: float
		Buffer size around the subvolume.
	
	Output:
	-----------
	xmin: float
		Minimum x-coordinate of the subvolume.
	xmax: float
		Maximum x-coordinate of the subvolume.
	ymin: float
		Minimum y-coordinate of the subvolume.
	ymax: float
		Maximum y-coordinate of the subvolume.
	zmin: float
		Minimum z-coordinate of the subvolume.
	zmax: float
		Maximum z-coordinate of the subvolume.
	"""

	subvol_ix,subvol_iy,subvol_iz=ivol_idx(ivol,nslice)
	subvol_L=boxsize/nslice
	subvol_buffer=subvol_L*buffer
	xmin=subvol_ix*subvol_L-subvol_buffer
	ymin=subvol_iy*subvol_L-subvol_buffer
	zmin=subvol_iz*subvol_L-subvol_buffer
	xmax=(subvol_ix+1)*subvol_L+subvol_buffer
	ymax=(subvol_iy+1)*subvol_L+subvol_buffer
	zmax=(subvol_iz+1)*subvol_L+subvol_buffer
	
	return xmin,xmax,ymin,ymax,zmin,zmax

def get_progidx(subcat,galid,depth):
	"""
	get_progidx: Find the main progenitor of a galaxy at a given redshift.

	Input:
	-----------
	subcat: pd.DataFrame
		DataFrame containing the subhalo catalogue.
	galid: int
		Galaxy ID of the galaxy of interest.
	depth: int
		Depth to search for the main progenitor (in terms of number of snapshots).
	
	Output:
	-----------
	nmerger_min: int
		Number of minor mergers.
	nmerger_maj: int
		Number of major mergers.
	galid_idepth: int
		Galaxy ID of the main progenitor at the given depth.
	
	
	"""
	galid_key='GalaxyID'
	descid_key='DescendantID'

	galid_idepth=galid
	nmerger_min=0;nmerger_maj=0

	for idepth in range(depth):
		#find main progenitor
		match_idepth_progen=subcat[descid_key].values==galid_idepth
		if np.nansum(match_idepth_progen):
			progens=subcat.loc[match_idepth_progen,:].copy()
			progens.sort_values('Mass',ignore_index=True,inplace=True,ascending=False)
			progens_mainid=progens[galid_key].values[0]
		else:
			return 0,0,0

		#check for mergers
		numprogen=np.nansum(match_idepth_progen)        
		if numprogen>1:
			match_idepth_masses=progens['Mass'].values
			mratio=10**(np.abs(np.log10(match_idepth_masses[0]/match_idepth_masses[1])))
			if mratio<=5:
				nmerger_maj+=1
			elif mratio>5:
				nmerger_min+=1

		#advance current id to progenitor
		galid_idepth=progens_mainid

	return nmerger_min,nmerger_maj,galid_idepth

def calc_r200(galaxy):
	
	"""
	calc_r200: Calculate an effective R200 of a galaxy if it is a satellite (to avoid using the whole group R200).

	Input:
	-----------
	galaxy: dict
		Dictionary containing the properties of the galaxy, including the group R200 and subgroup number.
	
	Output:
	-----------
	r200: float
		Effective R200 of the galaxy. Is the group R200 if the galaxy is a central, or is calculated from the galaxy mass if the galaxy is a satellite.
	
	"""

	#if satellite, find approximate r200 from subfind "Mass" field 
	if galaxy['SubGroupNumber']==0:
		return galaxy['Group_R_Crit200']
	else:
		h=cosmology.H0.value/100;a=(1/(1+galaxy['Redshift']))
		rhocrit=cosmology.critical_density(galaxy['Redshift'])
		rhocrit=rhocrit.to(units.Msun/units.Mpc**3)
		rhocrit=rhocrit.value
		if galaxy['SubGroupNumber']>0:
			m200_eff=galaxy['Mass']
			r200_cubed=3*m200_eff/(800*np.pi*rhocrit)
			r200=r200_cubed**(1/3)*h/a #return in comoving units
			return r200


##### CONSTANTS #####

constant_G=4.30073691e-09 #(km/s)^2*Mpc/Msun

vel_conversion=1*units.Mpc/units.Gyr #from pMpc/Gyr to km/s
vel_conversion=vel_conversion.to(units.km/units.s)
MpcpGyr_to_kmps=vel_conversion.value

distance_conversion=1*units.Mpc
distance_conversion=distance_conversion.to(units.km)
Mpc_to_km=distance_conversion.value
