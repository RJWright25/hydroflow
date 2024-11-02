# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: lower level mathematical functions for the repository.

import numpy as np
from astropy import constants
from astropy import units

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


def get_limits(ivol,nslice,boxsize,buffer=0.1):
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


def calc_temperature(pdata,XH=0.76,gamma=5/3):

	"""
	
	calc_temperature: Calculate the temperature of the gas particles in a subvolume 
	                  based on the internal energy and electron abundance.

	Input:
	-----------
	pdata: pd.DataFrame
		DataFrame containing the particle data.

	Output:
	-----------
	T: np.array
		Array containing the temperature of the gas particles. 

	"""

	#### Equation: T = (gamma-1)*(u/k_B)*mu
	#### where mu = 4/(1+3*XH+4*XH*ne)*m_p is the mean molecular weight

	u=pdata['InternalEnergy'].values*units.km**2/(units.s**2)
	ne=pdata['ElectronAbundance'].values*units.dimensionless_unscaled

	mu=4/(1+3*XH+4*XH*ne)*constant_mp*units.g
	T=(gamma-1)*u*mu/(constant_kB*units.km**2*units.g/(units.K*units.s**2))

	T=T.to(units.K).value

	return T

##### CONSTANTS #####

# Gravitational constant in (km/s)^2*Mpc/Msun
constant_G=constants.G.to(units.km**2*units.Mpc/(units.Msun*units.s**2)).value

# Boltzmann constant in (km/s)^2*g/K
constant_kB=constants.k_B.to(units.km**2*units.g/(units.K*units.s**2)).value

# Mass of the proton in g
constant_mp=constants.m_p.to(units.g).value

# Solar mass in g
constant_gpmsun=constants.M_sun.to(units.g).value

# Seconds in a year
constant_spyr=1*units.yr.to(units.s)

# Conversion factor from pMpc/Gyr to km/s -- used to align the units of particle velocities for the gas flow calculations
vel_conversion=1*units.Mpc/units.Gyr #from pMpc/Gyr to km/s
vel_conversion=vel_conversion.to(units.km/units.s)
constant_MpcpGyrtokmps=vel_conversion.value

# Conversion factor from Mpc to km
distance_conversion=1*units.Mpc
distance_conversion=distance_conversion.to(units.km)
constant_Mpcpkm=distance_conversion.value
