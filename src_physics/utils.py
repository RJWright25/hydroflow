# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: lower level mathematical functions for the repository.

import numpy as np
from astropy import units
from astropy.cosmology import Planck13 as cosmology

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
 

def get_limits(ivol,nslice,boxsize,buffer=0.2):
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
	galid_key='GalaxyID'
	descid_key='DescendantID'

	galid_idepth=galid
	nmerger_min=0;nmerger_maj=0

	snapnum=

	for idepth in range(depth):
		#find main progenitor
		match_idepth_progen=np.logical_and(subcat[descid_key].values==galid_idepth,subcat['SnapNum'].values==())
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

vel_conversion=1*units.Mpc/units.Gyr
vel_conversion=vel_conversion.to(units.km/units.s)
vel_conversion=vel_conversion.value
