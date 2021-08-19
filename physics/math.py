from __future__ import division
import numpy as np
from astropy import units
from astropy.cosmology import Planck13 as cosmology

MpcGyr_kms=1*units.Mpc/units.Gyr
MpcGyr_kms=MpcGyr_kms.to(units.km/units.s)
MpcGyr_kms=MpcGyr_kms.value

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    if len(data)==1:
        return data[0]

    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median

def ivol_gen(ix,iy,iz,nslice):
    ivol=ix*nslice**2+iy*nslice+iz
    ivol_str=str(ivol).zfill(3)
    return ivol_str

def ivol_idx(ivol,nslice):
    if type(ivol)==str:
        ivol=int(ivol)
    ix=int(np.floor(ivol/nslice**2))
    iz=int(ivol%nslice)
    iy=int((ivol-ix*nslice**2-iz)/nslice)
    return (ix,iy,iz)

def get_limits(ivol,nslice,boxsize,buffer=0.1):
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

def r200_eff(gal):
	h=cosmology.H0.value/100;a=(1/(1+gal['redshift']))
	rhocrit=cosmology.critical_density(gal['redshift'])
	rhocrit=rhocrit.to(units.Msun/units.Mpc**3)
	rhocrit=rhocrit.value
	if gal['SubGroupNumber']>0:
		m200_eff=gal['Mass']
	else:
		m200_eff=gal['Group_M_Crit200']

	r200_cubed=3*m200_eff/(800*np.pi*rhocrit)
	r200=r200_cubed**(1/3)*h/a #return in comoving units
	return r200

def bary_mp(x,y,eps=0.01,grad=1):

	"""
	Find the radius for a galaxy from the BaryMP method
	x = r/r_200
	y = cumulative baryonic mass profile
	eps = epsilon, if data 
	"""
	dydx = np.diff(y)/np.diff(x)
	
	maxarg = np.argwhere(dydx==np.max(dydx))[0][0] # Find where the gradient peaks
	xind = np.argwhere(dydx[maxarg:]<=grad)[0][0] + maxarg # The index where the gradient reaches 1
	
	x2fit_new, y2fit_new = x[xind:], y[xind:] # Should read as, e.g., "x to fit".
	x2fit, y2fit = np.array([]), np.array([]) # Gets the while-loop going
	
	while len(y2fit)!=len(y2fit_new):
		x2fit, y2fit = np.array(x2fit_new), np.array(y2fit_new)
		p = np.polyfit(x2fit, y2fit, 1)
		yfit = p[0]*x2fit + p[1]
		chi = abs(yfit-y2fit) # Separation in the y-direction for the fit from the data
		chif = (chi<eps) # Filter for what chi-values are acceptable
		x2fit_new, y2fit_new = x2fit[chif], y2fit[chif]
	
	r_bmp = x2fit[0] # Radius from the baryonic-mass-profile technique, returned as a fraction of the virial radius!
	Nfit = len(x2fit) # Number of points on the profile fitted to in the end

	return r_bmp, Nfit