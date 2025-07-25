# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Argument parser
parser=argparse.ArgumentParser()
parser.add_argument('--repo', type=str)
parser.add_argument('--code', type=str)
parser.add_argument('--path', type=str)
parser.add_argument('--nslice', type=int)
parser.add_argument('--ivol', type=int)
parser.add_argument('--snap', type=int)
parser.add_argument('--mcut', type=float)
parser.add_argument('--dump', type=int)
parser.add_argument('--pars', type=str)
args=parser.parse_args()

# Load arguments
repo, code, pathcat = args.repo, args.code, args.path
path = pathcat.split('cat')[0]
nslice, ivol, snap = args.nslice, args.ivol, args.snap
dump = bool(args.dump)
mcut = 10**(args.mcut)

# Set up paths
directory = pathcat.split('cat')[0]
sys.path.append(f"{repo.split('hydroflow')[0]}")

# Import the relevant routines
from hydroflow.run.initialise import load_metadata
from hydroflow.run.tools_hpc import create_dir,import_variables
from hydroflow.run.tools_catalogue import dump_hdf_group, dump_hdf, read_hdf
from hydroflow.src_physics.utils import get_limits, constant_G
from hydroflow.src_physics.galaxy import analyse_galaxy, retrieve_galaxy_candidates

# Load subhalo catalogue
namecat = pathcat.split('/')[-1][:-5]
run = path.split('/')[-1]
sim = run.split('_')[0]

create_dir('./jobs/gasflow')
create_dir('./catalogues/gasflow')

# Logging setup
logging_folder = f'{path}/jobs/gasflow/{namecat}/nvol{str(int(nslice**3)).zfill(3)}/snap{str(snap).zfill(3)}/'
logging_name = f"s{str(snap).zfill(3)}_n{str(int(nslice**3)).zfill(3)}_ivol{str(ivol).zfill(3)}"
create_dir(logging_folder)
os.remove(f'{logging_folder}{logging_name}.log') if os.path.exists(f'{logging_folder}{logging_name}.log') else None
logging.basicConfig(filename=f'{logging_folder}{logging_name}.log', level=logging.INFO)
logging.info(f"************{datetime.now()}************")
logging.info(f'Running hydroflow for {code} simulation with {namecat} catalogue [runtime {time.time():.3f} sec]')

# Initialise variables
r200_shells, rstar_shells, kpc_shells, zslab_radii= None, None, None, None
Tbins, theta_bins, vcuts, drfacs = None, None, None, None
pdata_fields = []

# Load parameters from file if given
pfile=args.pars
if pfile is not None and os.path.exists(pfile):
    params=import_variables(pfile)
    pars=['r200_shells','rstar_shells','kpc_shells','zslab_radii','Tbins','theta_bins','vcuts','drfacs','pdata_fields']
    for par in pars:
        if hasattr(params,par):
            exec(f"{par}=params.{par}")
    #write used parameters to log
    logging.info(f"Parameters loaded from {pfile}:")
    for par in pars:
        if hasattr(params,par):
            logging.info(f"{par} = {eval(par)}")
        else:
            logging.info(f"{par} not found in {pfile}")
else:
    from hydroflow.run.defaults import *

t1 = time.time()
logging.info(f'Loading subhalo catalogue: {pathcat} ...')
subcat = read_hdf(pathcat)

if code == 'colibre':
    from hydroflow.src_sims.colibre.particle import read_subvol
elif code == 'eagle':
    from hydroflow.src_sims.eagle.particle import read_subvol
elif code == 'tng':
    from hydroflow.src_sims.tng.particle import read_subvol
elif code == 'simba':
    from hydroflow.src_sims.simba.particle import read_subvol
elif code == 'swift-bosca':
    from hydroflow.src_sims._development.bosca.particle import read_subvol
else:
    raise ValueError('Invalid simulation code')

#metadata
directory=path.split('cat')[0]
metadata_path=None
if 'metadata' in subcat.attrs:
    metadata_path=subcat.attrs['metadata']
else:
    simflist=os.listdir(directory)
    for metadata_path in simflist:
        if '.pkl' in metadata_path:
            metadata_path=metadata_path
            break
if not metadata_path is None:
    metadata=load_metadata(metadata_path)
    logging.info(f'Metadata file found: {metadata_path} [runtime {time.time()-t1:.3f} sec]')
else:
    raise ValueError('Metadata file not found. Exiting.')


snap_mask = metadata.snapshots_idx == snap
snap_pdata_fname = metadata.snapshots_flist[snap_mask][0]
boxsize = metadata.boxsize
hval = metadata.hval
afac = metadata.snapshots_afac[snap_mask][0]

# Output paths
output_folder = f'{path}/catalogues/gasflow/{namecat}/nvol{str(int(nslice**3)).zfill(3)}/snap{str(snap).zfill(3)}/'
outcat_fname = output_folder + f'ivol_{str(ivol).zfill(3)}.hdf5'
create_dir(outcat_fname)
logging.info(f'Output file: {outcat_fname} [runtime {time.time()-t1:.3f} sec]')

if dump:
    dumpcat_folder = f'{output_folder}/pdata/'
    dumpcat_fname = dumpcat_folder + f'ivol_{str(ivol).zfill(3)}_pdata.hdf5'
    os.remove(dumpcat_fname) if os.path.exists(dumpcat_fname) else None

# Apply subhalo mask
subcat_limits = get_limits(ivol, nslice, boxsize, buffer=0)
snap_key, galid_key, mass_key = 'SnapNum', 'GalaxyID', 'Mass'
logging.info(f'Box limits: x - ({subcat_limits[0]:.1f},{subcat_limits[1]:.1f}); y - ({subcat_limits[2]:.1f},{subcat_limits[3]:.1f}); z - ({subcat_limits[4]:.1f},{subcat_limits[5]:.1f}) [runtime {time.time()-t1:.3f} sec]')

snap_mask = subcat[snap_key].values == snap
mass_mask = subcat[mass_key].values >= mcut
xmask = (subcat['CentreOfPotential_x'].values >= subcat_limits[0]) & (subcat['CentreOfPotential_x'].values < subcat_limits[1])
ymask = (subcat['CentreOfPotential_y'].values >= subcat_limits[2]) & (subcat['CentreOfPotential_y'].values < subcat_limits[3])
zmask = (subcat['CentreOfPotential_z'].values >= subcat_limits[4]) & (subcat['CentreOfPotential_z'].values < subcat_limits[5])

mask = snap_mask & mass_mask & xmask & ymask & zmask
subcat_selection = subcat.loc[mask].copy().sort_values(by='SubGroupNumber').reset_index(drop=True)
numgal = subcat_selection.shape[0]

logging.info(f'Mass limit: {np.log10(mcut):.1f} [runtime {time.time()-t1:.3f} sec]')
logging.info(f'Frac above limit: {np.nanmean(subcat_selection[mass_key].values>=mcut)*100:.1f}% [runtime {time.time()-t1:.3f} sec]')

galaxy_outputs = []

if numgal:
    logging.info(f'Will generate outputs for {numgal} galaxies in this subvolume [runtime {time.time()-t1:.3f} sec]')

    # Load in particle data
    logging.info(f'Loading snap particle data: {snap_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
    pdata_subvol,kdtree_subvol=read_subvol(snap_pdata_fname,ivol,nslice,metadata,logfile=logging_folder+logging_name+'.log')

    # Sanity checks for particle data
    logging.info(f'Coordinate minima: x - {pdata_subvol["Coordinates_x"].min():.2f}, y - {pdata_subvol["Coordinates_y"].min():.2f}, z - {pdata_subvol["Coordinates_z"].min():.2f} [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'Coordinate maxima: x - {pdata_subvol["Coordinates_x"].max():.2f}, y - {pdata_subvol["Coordinates_y"].max():.2f}, z - {pdata_subvol["Coordinates_z"].max():.2f} [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'Temperature min: {np.nanmin(pdata_subvol["Temperature"]):.2e} K')
    logging.info(f'Temperature max: {np.nanmax(pdata_subvol["Temperature"]):.2e} K')
    logging.info(f'Temperature mean: {pdata_subvol["Temperature"].mean():.2e}')

    if dump:
        for field in pdata_subvol.columns:
            if 'mfrac' in field:
                pdata_fields.append(field)

    logging.info(f'')
    logging.info(f'****** Entering main galaxy loop [runtime {time.time()-t1:.3f} sec] ******')
    logging.info(f'')

    for igal, galaxy in subcat_selection.iterrows():
        logging.info(f'')
        logging.info(f"Galaxy {igal+1}/{subcat_selection.shape[0]:.0f}: subhalo mass - {galaxy[mass_key]:.1e}, sgn - {galaxy['SubGroupNumber']} [runtime {time.time()-t1:.3f} sec]")

        # Initialise galaxy output
        central = galaxy['SubGroupNumber'] == 0
        maxrad = 3.5 * galaxy['Group_R_Crit200'] if central else 150e-3 # 3.5*r200 for centrals, 150ckpc for satellites
        
        # Check if the galaxy is on the edge of the box
        com=np.array([galaxy[f'CentreOfPotential_{x}'] for x in 'xyz'])
        galaxy['Edge']=0
        for idim in range(3):
            if com[idim]-maxrad<0 or com[idim]+maxrad>metadata.boxsize:
                galaxy['Edge']=1
                logging.info(f'Galaxy {int(galaxy[galid_key])} is on the edge of the box -- COM {com}. Setting Edge=1 [runtime {time.time()-t1:.3f} sec]')
                break
        
        # Add extra properties
        galaxy['ivol'] = ivol
        galaxy['HydroflowID'] = int(galaxy[galid_key])

        # Get the particle data for this halo
        t1_c=time.time()
        pdata_candidates=retrieve_galaxy_candidates(galaxy,pdata_subvol,kdtree_subvol,maxrad=maxrad,boxsize=boxsize)
        t2_c=time.time()

        if pdata_candidates.shape[0]>0:
            logging.info(f"Candidates: {t2_c-t1_c:.3f} sec (n = {pdata_candidates.shape[0]}, m = {np.nansum(pdata_candidates['Masses'].values):1e})")

            #### MAIN GALAXY ANALYSIS ####
            t1_f=time.time()
            galaxy_output=analyse_galaxy(galaxy,pdata_candidates,
                                                metadata=metadata,
                                                r200_shells=r200_shells,
                                                kpc_shells=kpc_shells,
                                                rstar_shells=rstar_shells,
                                                zslab_radii=zslab_radii,
                                                Tbins=Tbins,
                                                theta_bins=theta_bins,
                                                vcuts=vcuts,
                                                drfacs=drfacs,
                                                logfile=logging_folder+logging_name+'.log')
                                                
            
            t2_f=time.time()
            logging.info(f"Galaxy routine took: {t2_f-t1_f:.3f} sec")
            logging.info(f'Galaxy successfully processed [runtime {time.time()-t1:.3f} sec]')

            # Dump particle data if requested
            if dump:
                logging.info(f'Dumping particle data for galaxy {galaxy[galid_key]} [runtime {time.time()-t1:.3f} sec]')
                group=str(int(galaxy[galid_key]))
                data=pdata_candidates.loc[pdata_candidates['ParticleType'].values>=0,pdata_fields]
                columns=list(subcat_selection.columns)
                for column in list(galaxy_output.keys()):
                    if '0p10r200' in column or '1p00r200' in column or '030pkpc' in column or 'half' in column:
                        columns.append(column)
                metadata_dump={key:galaxy_output[key] for key in columns}
                dump_hdf_group(dumpcat_fname,group,data,metadata=metadata_dump,verbose=False)

        else:
            logging.info(f'No particles found for galaxy {int(galaxy[galid_key])} in subvolume {ivol}.')
            galaxy_output={}



        # Append to the list of galaxy outputs
        logging.info(f'Appending galaxy to output [runtime {time.time()-t1:.3f} sec]')
        galaxy_outputs.append(galaxy_output)
    
    logging.info(f'')
    logging.info(f'Finished with loop - concatenating output results [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')

else:
    logging.info(f'No galaxies found in this subvolume. Exiting.')
    galaxy_outputs = []

# Final output dataframe
if galaxy_outputs:
    galaxy_outputs = pd.DataFrame(galaxy_outputs)
    for key in subcat_selection.columns:
        if key not in galaxy_outputs.columns:
            galaxy_outputs[key] = subcat_selection[key].values
else:
    logging.info(f'No galaxies in this subvolume, empty output [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')
    galaxy_outputs=pd.DataFrame([])


# Save
logging.info(f'Saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')
dump_hdf(outcat_fname, galaxy_outputs)
logging.info(f'Done saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')



logging.info(f"************{datetime.now()}************")