# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_sims/colibre/particle.py: routines to read and convert particle data from COLIBRE snapshot outputs – uses swiftsimio.

import numpy as np
import pandas as pd
import os 
import time
import logging
import unyt

from scipy.spatial import cKDTree
from swiftsimio import load as swiftsimio_loader
from swiftsimio import mask as swiftsimio_mask

from hydroflow.src_physics.utils import get_limits, partition_neutral_gas

def read_subvol(path,ivol,nslice,metadata,logfile=None,verbose=False,gasonly=False):
    """
    read_subvol: Read particle data for a subvolume from an COLIBRE simulation snapshot using swiftsimio.

    Input:
    -----------
    path: str
        Path to the simulation snapshot.
    ivol: int
        Subvolume index (0-nslice**3-1).
    nslice: int
        Number of subvolumes in each dimension.

    Output:
    -----------
    pdata: pd.DataFrame
        DataFrame containing the particle data for the subvolume.
    pdata_kdtree: scipy.spatial.cKDTree
        KDTree containing the particle data for the subvolume.

    """
    # Set up logging
    if logfile is not None:
        os.makedirs(os.path.dirname(logfile),exist_ok=True)
        logging.basicConfig(filename=logfile,level=logging.INFO)
        logging.info(f"Reading subvolume {ivol} from {path}...")

    # Load snapshot with swiftsimio
    pdata_snap=swiftsimio_loader(path)
    boxsize=metadata.boxsize
    zval=metadata.snapshots_z[np.where(metadata.snapshots_flist==path)[0][0]]

    logging.info(f"Boxsize: {boxsize}")
    if verbose:
        print(f"Boxsize: {boxsize}")

    # Spatially mask the subregion if nslice>1
    if nslice>1:
        limits=get_limits(ivol=ivol,nslice=nslice,boxsize=boxsize)
        logging.info(f"Limits given: {limits}")
        mask=swiftsimio_mask(path)
        mask.constrain_spatial([[limits[0]*unyt.Mpc,limits[1]*unyt.Mpc],[limits[2]*unyt.Mpc,limits[3]*unyt.Mpc],[limits[4]*unyt.Mpc,limits[5]*unyt.Mpc]])
        pdata_snap_masked=swiftsimio_loader(path, mask=mask)
    else:
        pdata_snap_masked=pdata_snap

    pdata=[]
    t0=time.time()

    # Select particle types
    if gasonly:
        parttypes=[0]
        partstrs=['gas']
        partbuffer=[pdata_snap_masked.gas]
    else:
        parttypes=[0,1,4]
        partstrs=['gas','dm','stars','bhs']
        partbuffer=[pdata_snap_masked.gas,pdata_snap_masked.dark_matter,pdata_snap_masked.stars,pdata_snap_masked.black_holes]

    # Loop over particle types
    for iptype,ptype,pdata_masked_object  in zip(parttypes,partstrs,partbuffer):
        
        if ptype=='dm':
            subset=15 # only read ~10% of dark matter particles to save memory
        elif ptype=='stars':
            subset=2 # only read 50% of star particles to save memory
        else:
            subset=1 # read every gas/bh particle

        logging.info(f"Reading {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
        pdata_ptype=pd.DataFrame()

        if hasattr(pdata_masked_object,'halo_catalogue_index'):
            logging.info(f"Reading HaloCatalogueIndex for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            pdata_ptype['HaloCatalogueIndex']=pdata_masked_object.halo_catalogue_index.value[::subset]
        
        logging.info(f"Reading masses for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
        masses=pdata_masked_object.masses
        masses.convert_to_units('Msun')
        pdata_ptype['Masses']=masses.value[::subset]*subset
        del masses

        logging.info(f"Reading coordinates for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
        coordinates=pdata_masked_object.coordinates
        coordinates.convert_to_units('Mpc') #comoving
        for ix,x in enumerate('xyz'):
            pdata_ptype[f'Coordinates_{x}']=coordinates.value[::subset,ix]
        del coordinates
        
        logging.info(f"Reading velocities for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
        velocities=pdata_masked_object.velocities
        velocities.convert_to_units('km/s');velocities.convert_to_physical()
        for ix,x in enumerate('xyz'):
                pdata_ptype[f'Velocities_{x}']=velocities.value[::subset,ix]
        del velocities

        logging.info(f"Reading IDs for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
        pdata_ptype['ParticleIDs']=pdata_masked_object.particle_ids.value[::subset]
        pdata_ptype['ParticleType']=iptype*np.ones(pdata_ptype['ParticleIDs'].shape[0]) 
        
 
        # Read additional gas properties
        if ptype=='gas':

            # Temperature
            logging.info(f"Reading temperature... [pdata time: {time.time()-t0:.2f} s]")
            temp=pdata_masked_object.temperatures
            temp.convert_to_units('K')
            pdata_ptype['Temperature']=temp.value
            del temp

            hsml=pdata_masked_object.smoothing_lengths
            hsml.convert_to_units('Mpc') #comoving
            pdata_ptype['SmoothingLength']=hsml.value
            del hsml


            # Density
            logging.info(f"Reading density... [pdata time: {time.time()-t0:.2f} s]")
            dens=pdata_masked_object.densities
            dens.convert_to_units('g/cm**3');dens.convert_to_physical()
            pdata_ptype['Density']=dens.value
            del dens

            # Star formation rate
            logging.info(f"Reading SFR... [pdata time: {time.time()-t0:.2f} s]")
            sfr=pdata_masked_object.star_formation_rates
            sfr.convert_to_units('Msun/yr')
            pdata_ptype['StarFormationRate']=sfr.value
            del sfr

            # # Smoothing length
            # logging.info(f"Reading smoothing length... [pdata time: {time.time()-t0:.2f} s]")
            # hsml=pdata_masked_object.smoothing_lengths
            # hsml.convert_to_units('Mpc');hsml.convert_to_physical()
            # pdata_ptype['SmoothingLength']=hsml.value;del hsml

            # Species fractions
            if 'species_fractions' in pdata_snap.metadata.gas_properties.field_names:
                logging.info(f"Reading species fractions... [pdata time: {time.time()-t0:.2f} s]")
                hydrogen_frac=pdata_snap_masked.gas.element_mass_fractions.hydrogen.value
                logging.info(f"Reading HI... [pdata time: {time.time()-t0:.2f} s]")
                pdata_ptype['mfrac_HI']=pdata_snap_masked.gas.species_fractions.HI.value
                logging.info(f"Reading H2... [pdata time: {time.time()-t0:.2f} s]")
                pdata_ptype['mfrac_H2']=pdata_snap_masked.gas.species_fractions.H2.value*2 # multiply by 2 to mass fraction (instead of number density fraction)
                logging.info(f"Reading Z... [pdata time: {time.time()-t0:.2f} s]")
                pdata_ptype['Metallicity']=pdata_snap_masked.gas.metal_mass_fractions.value
                logging.info(f"Converting H species fractions to mass fractions... [pdata time: {time.time()-t0:.2f} s]")
                for spec in ['HI','H2']:
                    pdata_ptype[f'mfrac_{spec}']=pdata_ptype[f'mfrac_{spec}']*hydrogen_frac

                del hydrogen_frac

        # Read additional star properties
        elif ptype=='stars':
            # Metallicity
            logging.info(f"Reading Z... [pdata time: {time.time()-t0:.2f} s]")
            pdata_ptype['Metallicity']=pdata_masked_object.metal_mass_fractions.value[::subset]


        logging.info(f"Appending {ptype} particles to pdata... [pdata time: {time.time()-t0:.2f} s]")
        pdata.append(pdata_ptype)

    # Concatenate pdata
    pdata=pd.concat(pdata)
    pdata.sort_values('ParticleIDs',inplace=True)
    pdata.reset_index(drop=True,inplace=True)

    # Print fraction of particles that are gas
    gas=pdata['ParticleType'].values==0
    logging.info(f"Fraction of particles that are gas: {np.sum(gas)/pdata.shape[0]*100:.2f}%")

    # Add post-processed partitions into HI, H2, HII from Rahmati (2013) and Blitz & Rosolowsky (2006)
    # logging.info(f"Adding hydrogen partitioning...")
    # gas=pdata['ParticleType'].values==0
    # fHI,fH2,fHII=partition_neutral_gas(pdata,redshift=zval,sfonly=True)
    # logging.info(f"Minima: fHI: {np.nanmin(fHI)}, fHII: {np.nanmin(fHII)}, fH2: {np.nanmin(fH2)}]")
    # logging.info(f"Maxima: fHI: {np.nanmax(fHI)}, fHII: {np.nanmax(fHII)}, fH2: {np.nanmax(fH2)}]")

    # pdata.loc[:,['mfrac_HI_BR06','mfrac_H2_BR06']]=np.nan
    # pdata.loc[gas,'mfrac_HI_BR06']=fHI
    # pdata.loc[gas,'mfrac_H2_BR06']=fH2

    # Generate KDtree
    logging.info(f"Generating KDTree... [pdata time: {time.time()-t0:.2f} s]")
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize+1e-4,perio) #add buffer to avoid weird edge effects


    return pdata,pdata_kdtree
