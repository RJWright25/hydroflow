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

warnings.filterwarnings("ignore")

sys.path.append('/Users/rwright/GitHub/')

# Parameters
r200_shells=[0.05,0.1,0.15,0.2,0.25,0.3,0.5,0.75,1,1.5,2,2.5,3]
rstar_shells=[0.5,1,1.5,2,4]
kpc_shells=[1,2,5,10,15,20,25,30,40,50,75,100]
Tcuts_str=['cold','cool','warm','hot']
Tcuts=[0,1e3,1e5,1e7,1e15]
Tbins={Tcuts_str[i]:[Tcuts[i],Tcuts[i+1]] for i in range(len(Tcuts_str))}

# Particle data fields to dump
pdata_fields=['Masses','Relative_r_comoving','Coordinates_x','Coordinates_y','Coordinates_z','Relative_vrad_pec','Relative_vx_pec','Relative_vy_pec','Relative_vz_pec','Relative_theta','Temperature','Density','Metallicity']

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
parser.add_argument('--dr', type=float)
args=parser.parse_args()

# Load arguments
repo, code, pathcat = args.repo, args.code, args.path
path = pathcat.split('cat')[0]
nslice, ivol, snap = args.nslice, args.ivol, args.snap
dump = bool(args.dump)
mcut = 10**(args.mcut)
drfac = args.dr

directory = pathcat.split('cat')[0]
sys.path.append(f"{repo.split('hydroflow')[0]}")

from hydroflow.run.initialise import load_metadata
from hydroflow.run.tools_hpc import create_dir
from hydroflow.run.tools_catalog import dump_hdf_group, dump_hdf, read_hdf
from hydroflow.src_physics.utils import get_limits, constant_G
from hydroflow.src_physics.galaxy import analyse_galaxy, retrieve_galaxy_candidates

# Load subhalo catalogue
namecat = pathcat.split('/')[-1][:-5]
run = path.split('/')[-1]
sim = run.split('_')[0]
dr_str = f"{drfac:.2f}".replace('.', 'p')

create_dir('./jobs/gasflow')
create_dir('./catalogues/gasflow')

# Logging setup
logging_folder = f'{path}/jobs/gasflow/{namecat}/nvol{str(int(nslice**3)).zfill(3)}_dr{dr_str}/snap{str(snap).zfill(3)}/'
logging_name = f"s{str(snap).zfill(3)}_n{str(int(nslice**3)).zfill(3)}_ivol{str(ivol).zfill(3)}"
create_dir(logging_folder)
os.remove(f'{logging_folder}{logging_name}.log') if os.path.exists(f'{logging_folder}{logging_name}.log') else None
logging.basicConfig(filename=f'{logging_folder}{logging_name}.log', level=logging.INFO)

t1 = time.time()
logging.info(f"************{datetime.now()}************")
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

# Load metadata
if 'metadata' in subcat.attrs:
    metadata_path = subcat.attrs['metadata']
else:
    simflist = os.listdir(directory)
    metadata_path = next((f for f in simflist if '.pkl' in f), None)
if metadata_path is None:
    raise ValueError('Metadata file not found')
metadata = load_metadata(metadata_path)

snap_mask = metadata.snapshots_idx == snap
snap_pdata_fname = metadata.snapshots_flist[snap_mask][0]
boxsize = metadata.boxsize
hval = metadata.hval
afac = metadata.snapshots_afac[snap_mask][0]

# Output paths
output_folder = f'{path}/catalogues/gasflow/{namecat}/nvol{str(int(nslice**3)).zfill(3)}_dr{dr_str}/snap{str(snap).zfill(3)}/'
outcat_fname = output_folder + f'ivol_{str(ivol).zfill(3)}.hdf5'
create_dir(outcat_fname)

if dump:
    dumpcat_folder = f'{output_folder}/pdata/'
    dumpcat_fname = dumpcat_folder + f'ivol_{str(ivol).zfill(3)}_pdata.hdf5'
    os.remove(dumpcat_fname) if os.path.exists(dumpcat_fname) else None

# Apply subhalo mask
subcat_limits = get_limits(ivol, nslice, boxsize, buffer=0)
snap_key, galid_key, mass_key = 'SnapNum', 'GalaxyID', 'Mass'

snap_mask = subcat[snap_key].values == snap
mass_mask = subcat[mass_key].values >= mcut
xmask = (subcat['CentreOfPotential_x'].values >= subcat_limits[0]) & (subcat['CentreOfPotential_x'].values < subcat_limits[1])
ymask = (subcat['CentreOfPotential_y'].values >= subcat_limits[2]) & (subcat['CentreOfPotential_y'].values < subcat_limits[3])
zmask = (subcat['CentreOfPotential_z'].values >= subcat_limits[4]) & (subcat['CentreOfPotential_z'].values < subcat_limits[5])

mask = snap_mask & mass_mask & xmask & ymask & zmask
subcat_selection = subcat.loc[mask].copy().sort_values(by='SubGroupNumber').reset_index(drop=True)

numgal = subcat_selection.shape[0]
galaxy_outputs = []

if numgal:
    pdata_subvol, kdtree_subvol = read_subvol(snap_pdata_fname, ivol, nslice, metadata, logfile=logging_folder + logging_name + '.log')

    if dump:
        for field in pdata_subvol.columns:
            if 'mfrac' in field:
                pdata_fields.append(field)

    for igal, galaxy in subcat_selection.iterrows():
        galaxy_output = {}
        central = galaxy['SubGroupNumber'] == 0
        maxrad = 3.5 * galaxy['Group_R_Crit200'] if central else 150e-3 # 3.5*r200 for centrals, 150kpc for satellites

        pdata_candidates = retrieve_galaxy_candidates(galaxy, pdata_subvol, kdtree_subvol, maxrad, boxsize)

        if pdata_candidates.shape[0] > 0:
            result = analyse_galaxy(galaxy, pdata_candidates, metadata, r200_shells, kpc_shells, rstar_shells, Tbins, drfac, logfile=logging_folder + logging_name + '.log')
            if not result.empty:
                result['ivol'] = ivol
                result['HydroflowID'] = int(galaxy[galid_key])
                result['Group_V_Crit200'] = np.sqrt(constant_G * galaxy['Group_M_Crit200'] / (galaxy['Group_R_Crit200'] * afac))
                galaxy_outputs.append(result.to_dict(orient='records')[0])

                if dump:
                    # Dump particle data to a hdf5 group with a subset of metadata
                    group = str(int(galaxy[galid_key]))
                    data = pdata_candidates.loc[pdata_candidates['ParticleType'].values == 0, pdata_fields]
                    columns = list(subcat_selection.columns)
                    for column in result.columns:
                        if '0p10r200' in column or '1p00r200' in column or '030pkpc' in column:
                            columns.append(column)
                    metadata_dump = {key: result[key] for key in columns if key in result.columns}
                    dump_hdf_group(dumpcat_fname, group, data, metadata=metadata_dump, verbose=False)

# Final output dataframe
if galaxy_outputs:
    galaxy_outputs = pd.DataFrame(galaxy_outputs)
    for key in subcat_selection.columns:
        if key not in galaxy_outputs.columns:
            galaxy_outputs[key] = subcat_selection[key].values
else:
    galaxy_outputs = pd.DataFrame([])

# Save
dump_hdf(outcat_fname, galaxy_outputs)
logging.info('Done.')
logging.info(f"************{datetime.now()}************")