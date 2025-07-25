# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: lower level mathematical functions for the repository.

import numpy as np
from astropy import constants
from astropy import units
import time 

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


def compute_cylindrical_ztheta(pdata,baryons=True,aperture=30*1e-3,afac=1):
    """
    compute_cylindrical_ztheta: Calculate the angular momentum of a system of particles and the angle between the angular momentum and the position vector of each particle.

    Input:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data.
    baryons: bool
        Flag to only consider baryonic particles.
    aperture: float
        Aperture radius to mask the particles (in physical Mpc). 
    
    Output:
    -----------
    Lbar: np.array
        Array containing the angular momentum of the system.

    theta: np.array
        Array containing the angle between the angular momentum of the system and the position vector of each particle.
        The output is in degrees and ranges from [0,90] degrees -- 0 degrees corresponds to particles aligned with the angular momentum vector/minor axis.

    z : np.array
        Array containing the z-coordinate of the particles relative to the disk plane.
        The output is in physical Mpc and can be positive or negative.

    """

    # Mask the particles within the aperture and only baryonic particles
    ptypes=pdata['ParticleType'].values
    masses=pdata['Masses'].values
    radii=pdata['Relative_r_comoving'].values*afac
    positions=pdata.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values*afac
    velocities=pdata.loc[:,[f'Relative_v{x}_pec' for x in 'xyz']].values

    if baryons:
        mask=np.logical_or(ptypes==0,ptypes==4) & (radii<aperture)
    else:
        mask=(radii<aperture)

	# Define the angular momentum of the galaxy with baryonic elements within aperture
    Lbar=np.nansum(np.cross(positions[mask],masses[mask][:,np.newaxis]*velocities[mask]),axis=0)
    Lbarhat=Lbar/np.linalg.norm(Lbar)

    # Find the z-coordinate of the particles relative to the disk plane
    zheight=np.dot(positions,Lbarhat)
    zheight=zheight/afac #convert back to comoving units

    ## Position angles
	# Find the angle between the angular momentum of the galaxy and the position vector of each particle
    cos_theta=np.sum(Lbar*positions,axis=1)/(np.linalg.norm(Lbar)*np.linalg.norm(positions,axis=1))
    deg_theta=np.arccos(cos_theta)*180/np.pi
    deg_theta[deg_theta>90]=180-deg_theta[deg_theta>90] # particles with e.g. theta=180 degrees (opposite minor axis) are re-assigned to 0 degrees (mirrored)
    
    # Now make 90 degrees the minor axis
    theta_pos=90-deg_theta

    ## Velocity angles
    # Find the angle between the angular momentum of the galaxy and the velocity vector of each particle
    cos_theta_vel=np.sum(Lbar*velocities,axis=1)/(np.linalg.norm(Lbar)*np.linalg.norm(velocities,axis=1))
    deg_theta_vel=np.arccos(cos_theta_vel)*180/np.pi
    deg_theta_vel[deg_theta_vel>90]=180-deg_theta_vel[deg_theta_vel>90] # particles with e.g. theta=180 degrees (opposite minor axis) are re-assigned to 0 degrees (mirrored)
    # Now make 90 degrees the minor axis
    theta_vel=90-deg_theta_vel



    return Lbar, theta_pos, theta_vel, zheight

def rahmati2013_neutral_fraction(nH,T,redshift=0):
    """
     
    rahmati2013_neutral_fraction: Calculate the neutral fraction of the gas particles in a subvolume based on the gas density and temperature.
        						  Adapted from Rahmati et al. (2013) - https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2427R/abstract. From 
        						  Rob Crain, Leiden, March 2014, adapted to python by Michelle Furlong. 

     
    Input:
    -----------
    
    nH: np.array
		Array containing the gas density of the gas particles.
	T: np.array
		Array containing the temperature of the gas particles.
	redshift: float
		Redshift of the simulation snapshot.
          
	Output:
	-----------
     
	f_neutral: np.array
		Array containing the neutral fraction of the gas particles.
          

    """
    t0=time.time()
    if redshift>5:
        redshift = 5.0

    if redshift < 1.0:
        dlogz = (np.log10(1+redshift) - 0.0)/np.log10(2.)
        lg_n0_lo     = -2.56
        gamma_uvb_lo =  8.34e-14
        alpha1_lo    = -1.86
        alpha2_lo    = -0.51
        beta_lo      =  2.83
        f_lo         =  0.01
        lg_n0_hi     = -2.29
        gamma_uvb_hi =  7.39e-13
        alpha1_hi    = -2.94
        alpha2_hi    = -0.90
        beta_hi      =  1.21
        f_hi         =  0.03

    elif (redshift >= 1.0 and redshift < 2.0):
        dlogz = (np.log10(1+redshift) - np.log10(2.))/(np.log10(3.)-np.log10(2.))
        lg_n0_lo     = -2.29
        gamma_uvb_lo =  7.39e-13 
        alpha1_lo    = -2.94
        alpha2_lo    = -0.90
        beta_lo      =  1.21
        f_lo         =  0.03
        lg_n0_hi     = -2.06
        gamma_uvb_hi =  1.50e-12
        alpha1_hi    = -2.22
        alpha2_hi    = -1.09
        beta_hi      =  1.75
        f_hi         =  0.03
    
    elif (redshift >= 2.0 and redshift < 3.0):
        dlogz = (np.log10(1+redshift) - np.log10(3.))/(np.log10(4.)-np.log10(3.))
        lg_n0_lo     = -2.06
        gamma_uvb_lo =  1.50e-12
        alpha1_lo    = -2.22
        alpha2_lo    = -1.09
        beta_lo      =  1.75
        f_lo         =  0.03
        lg_n0_hi     = -2.13
        gamma_uvb_hi =  1.16e-12
        alpha1_hi    = -1.99
        alpha2_hi    = -0.88
        beta_hi      =  1.72
        f_hi         =  0.04

    elif (redshift >= 3.0 and redshift < 4.0):
        dlogz = (np.log10(1+redshift) - np.log10(4.))/(np.log10(5.)-np.log10(4.))
        lg_n0_lo     = -2.13
        gamma_uvb_lo =  1.16e-12
        alpha1_lo    = -1.99
        alpha2_lo    = -0.88
        beta_lo      =  1.72
        f_lo         =  0.04
        lg_n0_hi     = -2.23
        gamma_uvb_hi =  7.91e-13
        alpha1_hi    = -2.05
        alpha2_hi    = -0.75
        beta_hi      =  1.93
        f_hi         =  0.02

    elif (redshift >= 4.0 and redshift <= 5.0):
        dlogz = (np.log10(1+redshift) - np.log10(5.))/(np.log10(6.)-np.log10(5.))
        lg_n0_lo     = -2.23
        gamma_uvb_lo =  7.91e-13 
        alpha1_lo    = -2.05
        alpha2_lo    = -0.75
        beta_lo      =  1.93
        f_lo         =  0.02
        
        lg_n0_hi     = -2.35
        gamma_uvb_hi =  5.43e-13
        alpha1_hi    = -2.63
        alpha2_hi    = -0.57
        beta_hi      =  1.77
        f_hi         =  0.01

    lg_n0     = lg_n0_lo     + dlogz*(lg_n0_hi     - lg_n0_lo)
    n0        = 10.**lg_n0
    lg_gamma_uvb_lo, lg_gamma_uvb_hi = np.log10(gamma_uvb_lo), np.log10(gamma_uvb_lo)
    gamma_uvb = 10**(lg_gamma_uvb_lo + dlogz*(lg_gamma_uvb_hi - lg_gamma_uvb_lo))
    alpha1    = alpha1_lo    + dlogz*(alpha1_hi    - alpha1_lo)
    alpha2    = alpha2_lo    + dlogz*(alpha2_hi    - alpha2_lo)
    beta      = beta_lo      + dlogz*(beta_hi      - beta_lo)
    f         = f_lo         + dlogz*(f_hi         - f_lo)

    gamma_ratio = (1.-f) * (1. + (nH / n0)**beta)**alpha1 + f*(1. + (nH / n0))**alpha2
    gamma_phot  = gamma_uvb * gamma_ratio

    lambda_T  = 315614.0 / T
    AlphaA    = 1.269e-13 * (lambda_T)**(1.503) / ((1. + (lambda_T / 0.522)**0.470)**1.923)
    LambdaT   = 1.17e-10 * (np.sqrt(T) * np.exp(-157809.0 / T) / (1.0 + np.sqrt(T / 1.0e5)))

    A = AlphaA + LambdaT
    B = 2.0 * AlphaA + (gamma_phot / nH) + LambdaT
    sqrt_term = np.sqrt(np.maximum(B**2 - 4.0 * A * AlphaA, 0))
    f_neutral = (B - sqrt_term) / (2.0 * A)
    f_neutral = np.maximum(f_neutral, 1e-30)  # negative values seem to arise from rounding errors - AlphaA and A are both positive, so B-sqrt_term should be positive!
    
    t1 = time.time()
    print(f"Time taken for Rahmati neutral fraction: {t1-t0:.3f} sec")
    
    return f_neutral


def partition_neutral_gas(pdata,redshift,xH=0.76,sfonly=True):
    """
	neutral_partitioning: Calculate the neutral, ionized and molecular hydrogen fractions of the gas particles in a subvolume based on the gas density and temperature.
						  Uses the neutral fraction calculation from Rahmati et al. (2013) - https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2427R/abstract.
                          Subsequently calculates the molecular, atomic and ionized hydrogen fractions based Blitz & Rosolowsky (2006) - https://ui.adsabs.harvard.edu/abs/2006ApJ...650..933B/abstract.
                                
     
    """

	# Mask for gas particles
    gas=pdata['ParticleType'].values==0

	# Initialize arrays
    fHI=np.zeros(pdata.shape[0])
    fH2=np.zeros(pdata.shape[0])
    fHII=np.zeros(pdata.shape[0])
	
	# Get gas properties in correct units
    nH=pdata.loc[gas,'Density'].values*xH/constant_mp
    T=pdata.loc[gas,'Temperature'].values
    sfr=pdata.loc[gas,'StarFormationRate'].values

	# Mask for star-forming gas particles (if required)
    if not sfonly:
        sfr=np.ones(nH.shape[0])
    sfrmask=np.where(sfr>0)
	 
    # Neutral fraction
    fneutral=rahmati2013_neutral_fraction(nH,T,redshift=redshift)
    fHII=(1-fneutral)*xH
    t0=time.time()

    # H2 fraction
    midplane_pressure = T*nH # true pressure have to multiply this by kB, this is in cm^-3 K

    # Partition function
    Rmol=(midplane_pressure/4.3e4)**0.92 # Blitz & Rosolowsky (2006) 
    fH2=np.zeros(nH.shape[0])
    fH2[sfrmask]=1/(1+Rmol[sfrmask])
    fHI=1-fH2
    fH2*=fneutral*xH # convert from fraction of neutral mass to fraction of total mass
    fHI*=fneutral*xH # convert from fraction of neutral mass to fraction of total mass
    t1=time.time()
    print(f"Time taken for neutral hydrogen partitioning: {t1-t0:.3f} sec")
	
    return fHI,fH2,fHII


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


def calc_halfmass_radius(masses,radius):
    totalmass=np.nansum(masses)
    halfmass=totalmass/2
    rsort=np.argsort(radius)
    msort=np.cumsum(masses[rsort])
    halfmass_idx=np.searchsorted(msort,halfmass)
    halfmass_radius=radius[rsort][halfmass_idx]

    return halfmass_radius


def estimate_mu(x_H, T, y=0.08):
    """
    Vectorized estimate of mean molecular weight μ for given hydrogen ionisation and temperature.
    
    Parameters:
    -----------
    x_H : float or np.ndarray
        Hydrogen ionisation fraction (mass-weighted), shape (...,)
    T : float or np.ndarray
        Temperature in K, shape (...,)
    y : float
        Helium-to-hydrogen number ratio (default = 0.08 for primordial)

    Returns:
    --------
    mu : np.ndarray
        Mean molecular weight in units of proton mass, shape (...,)
    """
    x_H = np.asarray(x_H)
    T = np.asarray(T)
    
    x_He = np.zeros_like(T)
    x_He[(T >= 1e4) & (T < 1e5)] = 1
    x_He[T >= 1e5] = 2

    numerator = 1 + 4 * y
    denominator = 1 + x_H + y * (1 + x_He)
    mu = numerator / denominator

    return mu

def weighted_nanpercentile(data, weights, percentiles):
    """
    Compute weighted percentiles ignoring NaNs.

    Parameters
    ----------
    data : array_like
        Input data array, can contain NaNs.
    weights : array_like
        Weights for each data point. Must be same shape as `data`.
    percentiles : scalar or array_like
        Percentile or sequence of percentiles to compute (0-100).

    Returns
    -------
    percentiles : scalar or ndarray
        Weighted percentile(s) of the data.
    """
    data = np.asarray(data)
    weights = np.asarray(weights)
    percentiles = np.atleast_1d(percentiles)

    # Mask NaNs
    mask = ~np.isnan(data)
    data = data[mask]
    weights = weights[mask]

    if len(data) == 0:
        return np.full_like(percentiles, np.nan, dtype=np.float64)

    # Sort data and associated weights
    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]

    # Compute the cumulative sum of weights and normalize
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    # Interpolate to find the percentiles
    return np.interp(percentiles / 100.0, cdf, data) if len(percentiles) > 1 else np.interp(percentiles[0] / 100.0, cdf, data)





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

# cm in a kpc
constant_cmpkpc=1*units.kpc.to(units.cm)

# Conversion factor from pMpc/Gyr to km/s -- used to align the units of particle velocities for the gas flow calculations
vel_conversion=1*units.Mpc/units.Gyr #from pMpc/Gyr to km/s
vel_conversion=vel_conversion.to(units.km/units.s)
constant_MpcpGyrtokmps=vel_conversion.value

# Conversion factor from Mpc to km
distance_conversion=1*units.Mpc
distance_conversion=distance_conversion.to(units.km)
constant_Mpcpkm=distance_conversion.value




