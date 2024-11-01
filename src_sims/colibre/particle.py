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

from hydroflow.src_physics.utils import get_limits

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
    #set up logging
    if logfile is not None:
        os.makedirs(os.path.dirname(logfile),exist_ok=True)
        logging.basicConfig(filename=logfile,level=logging.INFO)
        logging.info(f"Reading subvolume {ivol} from {path}...")
    if verbose:
            print(f"Reading subvolume {ivol} from {path}...")

    #load snapshot with swiftsimio
    pdata_snap=swiftsimio_loader(path)
    boxsize=metadata.boxsize

    logging.info(f"Boxsize: {boxsize}")
    if verbose:
        print(f"Boxsize: {boxsize}")

    #spatially mask the subregion if nslice>1
    if nslice>1:
        limits=get_limits(ivol=ivol,nslice=nslice,boxsize=boxsize*unyt.Mpc)

        logging.info(f"Limits given: {limits}")
        if verbose:
            print(f"Limits given: {limits}")

        mask=swiftsimio_mask(path)
        mask.constrain_spatial([[limits[0],limits[1]],[limits[2],limits[3]],[limits[4],limits[5]]])
        pdata_snap_masked=swiftsimio_loader(path, mask=mask)

    else:
        pdata_snap_masked=pdata_snap

    pdata=[]
    t0=time.time()

    if gasonly:
        parttypes=[0]
        partstrs=['gas']
        partbuffer=[pdata_snap_masked.gas]
    else:
        parttypes=[0,1,4]
        partstrs=['gas','dm','stars']
        partbuffer=[pdata_snap_masked.gas,pdata_snap_masked.dark_matter,pdata_snap_masked.stars]

    #loop over particle types
    for iptype,ptype,pdata_masked_object  in zip(parttypes,partstrs,partbuffer):
            logging.info(f"Reading {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            if verbose:
                print(f"Reading {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            pdata_ptype=pd.DataFrame()
            
            logging.info(f"Reading masses and coordinates for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            if verbose:
                print(f"Reading masses and coordinates for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")

            masses=pdata_masked_object.masses
            masses.convert_to_units('Msun')
            pdata_ptype['Masses']=pdata_masked_object.masses.value

            for ix,x in enumerate('xyz'):
                 pdata_ptype[f'Coordinates_{x}']=pdata_masked_object.coordinates[:,ix]
            
            logging.info(f"Reading velocities for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            if verbose:
                print(f"Reading velocities for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")

            velocities=pdata_masked_object.velocities
            velocities.convert_to_units('km/s')
            for ix,x in enumerate('xyz'):
                 pdata_ptype[f'Velocities_{x}']=velocities[:,ix].value

            logging.info(f"Reading IDs for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            if verbose:
                print(f"Reading IDs for {ptype} particles... [pdata time: {time.time()-t0:.2f} s]")
            pdata_ptype['ParticleIDs']=pdata_masked_object.particle_ids.value
            pdata_ptype['ParticleType']=np.ones(pdata_ptype.shape[0])*iptype    

            if ptype=='gas':
                #temperature
                logging.info(f"Reading temperature... [pdata time: {time.time()-t0:.2f} s]")
                if verbose:
                    print(f"Reading temperature... [pdata time: {time.time()-t0:.2f} s]")
                temp=pdata_masked_object.temperatures;temp.convert_to_units('K')
                pdata_ptype['Temperature']=temp.value;del temp
                dens=pdata_masked_object.densities;dens.convert_to_units('g/cm**3')
                pdata_ptype['Density']=dens.value;del dens

                #sfr
                logging.info(f"Reading SFR... [pdata time: {time.time()-t0:.2f} s]")
                if verbose:
                    print(f"Reading SFR... [pdata time: {time.time()-t0:.2f} s]")
                sfr=pdata_masked_object.star_formation_rates
                sfr.convert_to_units('Msun/yr')
                pdata_ptype['StarFormationRate']=sfr.value

                #species
                if 'species_fractions' in pdata_snap.metadata.gas_properties.field_names:
                    logging.info(f"Reading species fractions... [pdata time: {time.time()-t0:.2f} s]")
                    if verbose:
                        print(f"Reading species fractions... [pdata time: {time.time()-t0:.2f} s]")
                    hydrogen_frac=pdata_snap_masked.gas.element_mass_fractions.hydrogen.value
                    logging.info(f"Reading HI... [pdata time: {time.time()-t0:.2f} s]")
                    pdata_ptype['mfrac_HI']=pdata_snap_masked.gas.species_fractions.HI.value
                    logging.info(f"Reading HII... [pdata time: {time.time()-t0:.2f} s]")
                    pdata_ptype['mfrac_HII']=pdata_snap_masked.gas.species_fractions.HII.value
                    logging.info(f"Reading HM... [pdata time: {time.time()-t0:.2f} s]")
                    pdata_ptype['mfrac_H2']=pdata_snap_masked.gas.species_fractions.Hm.value*2
                    logging.info(f"Reading Z... [pdata time: {time.time()-t0:.2f} s]")
                    pdata_ptype['Metallicity']=pdata_masked_object.metal_mass_fractions.value

                    logging.info(f"Converting H species fractions to mass fractions... [pdata time: {time.time()-t0:.2f} s]")
                    for spec in ['HI','HII','H2']:
                        pdata_ptype[f'mfrac_{spec}']=pdata_ptype[f'mfrac_{spec}']*hydrogen_frac

            elif ptype=='stars':
                logging.info(f"Reading Z... [pdata time: {time.time()-t0:.2f} s]")
                pdata_ptype['Metallicity']=pdata_masked_object.metal_mass_fractions.value

            logging.info(f"Appending {ptype} particles to pdata... [pdata time: {time.time()-t0:.2f} s]")
            pdata.append(pdata_ptype)

    #concatenate pdata
    pdata=pd.concat(pdata)
    pdata.sort_values('ParticleIDs',inplace=True)
    pdata.reset_index(drop=True,inplace=True)

    #generate KDtree
    logging.info(f"Generating KDTree... [pdata time: {time.time()-t0:.2f} s]")
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values,boxsize=boxsize+1e-4)

    return pdata,pdata_kdtree
