# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/execute.py: script to submit job array.

import os
import sys
import time
import logging
import argparse

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

sys.path.append('/Users/rwright/GitHub/')
sys.path.append('/Users/rwright/GitHub/hydroflow_colibre/')

#params
drfac=0.2 #fractional shell width for radial profiles
r200_shells=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3]
ckpc_shells=[10,20,30,40,50,60,70,80,90,100]

#temperature cuts for phase
Tcuts_str=['cold','cool','warm','hot']
Tcuts=[0,1e3,1e5,1e7,1e15]
Tbins={Tcuts_str[i]:[Tcuts[i],Tcuts[i+1]] for i in range(len(Tcuts_str))}

#velocity cuts (pseudo-evolution added in galaxy.py)
vcuts_flow=[0];vcuts_str=[str(int(vcut)).zfill(3)+'kmps' for vcut in vcuts_flow]
vcuts={vcut_str:vcut for vcut_str,vcut in zip(vcuts_str,vcuts_flow)}

#dump fields
pdata_fields=['Masses',
              'Relative_r_comoving',
              'Coordinates_x',
              'Coordinates_y',
              'Coordinates_z',
              'Relative_v_rad_pec',
              'Relative_phi',
              'Temperature',
              'Density',
              'Metallicity']

#arguments
parser=argparse.ArgumentParser()
parser.add_argument('--repo',metavar='-R',type=str,help='where is repo')
parser.add_argument('--code',metavar='-C',type=str,help='which simulation (from hydroflow_colibre.src_sims)')
parser.add_argument('--path',metavar='-P',type=str,help='path to subhalo catalogue')
parser.add_argument('--nslice',metavar='-N',type=int,help='number of slices for simulation sub-boxes')
parser.add_argument('--ivol',metavar='-I',type=int,help='which sub-volume to consider')
parser.add_argument('--snap',metavar='-S',type=int,help='which snapshot to consider')
parser.add_argument('--mcut',metavar='-M',type=float,help='lower mass limit for calc (log mass)')
parser.add_argument('--dump',metavar='-D',type=int,help='whether to dump particle data')

args=parser.parse_args()
repo=args.repo
code=args.code
pathcat=args.path
path=pathcat.split('cat')[0]
nslice=int(args.nslice)
ivol=int(args.ivol)
snap=int(args.snap)
dump=bool(args.dump)
mcut=10**(args.mcut)

sys.path.append(f"{repo.split('hydroflow')[0]}")

from hydroflow.run.initialise import load_metadata,simulation_metadata
from hydroflow.run.tools_hpc import create_dir
from hydroflow.run.tools_catalog import dump_hdf_group,dump_hdf,read_hdf
from hydroflow.src_physics.utils import get_limits,constant_G
from hydroflow.src_physics.galaxy import analyse_galaxy, retrieve_galaxy_candidates

#subhalo catalogue
namecat=pathcat.split('/')[-1][:-5]
run=path.split('/')[-1]
sim=run.split('_')[0]

current_dir=os.getcwd()
create_dir(current_dir+'/jobs/gasflow')
create_dir(current_dir+'/catalogues/gasflow')

#logging file
t1=time.time()
logging_folder=f'{path}/jobs/gasflow/{namecat}/nvol_{str(int(nslice**3)).zfill(3)}/snap{str(snap).zfill(3)}/'
logging_name=f"s{str(snap).zfill(3)}_n{str(int(nslice**3)).zfill(3)}_ivol{str(ivol).zfill(3)}"
create_dir(logging_folder)
if os.path.exists(f'{logging_folder}{logging_name}.log'):
    os.remove(f'{logging_folder}{logging_name}.log')
logging.basicConfig(filename=f'{logging_folder}{logging_name}.log', level=logging.INFO)

logging.info(f'')
logging.info(f'*******************************************************************************************')
logging.info(f'************************************* ANALYSE GASFLOW *************************************')
logging.info(f'*******************************************************************************************')
logging.info(f'')
logging.info(f'************{datetime.now()}************')
logging.info(f'')
logging.info(f'Loading subhalo catalogue: {pathcat} ... [runtime {time.time()-t1:.3f} sec]')

subcat=read_hdf(pathcat)

logging.info(f'Subhalo catalogue loaded [runtime {time.time()-t1:.3f} sec]')
logging.info(f'Masking catalogue and processing metadata [runtime {time.time()-t1:.3f} sec]')

snap_key='SnapNum'
galid_key='GalaxyID'
mass_key='Mass'

if code=='colibre':
    from hydroflow.src_sims.colibre.particle import read_subvol
elif code=='eagle':
    from hydroflow.src_sims.eagle.particle import read_subvol
elif code=='tng':
    from hydroflow.src_sims.tng.particle import read_subvol
elif code=='simba':
    from hydroflow.src_sims.simba.particle import read_subvol
elif code=='swift-bosca':
    from hydroflow.src_sims._development.bosca.particle import read_subvol

else:
    raise ValueError('Particle data type not recognised. Must be one of: colibre, eagle, simba, tng.')

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

snap_mask=metadata.snapshots_idx==snap
snap_pdata_fname=metadata.snapshots_flist[snap_mask][0]
boxsize=metadata.boxsize
hval=metadata.hval
afac=metadata.snapshots_afac[snap_mask][0]

#outputs
output_folder=f'{path}/catalogues/gasflow/{namecat}/nvol_{str(int(nslice**3)).zfill(3)}/snap{str(snap).zfill(3)}/'
outcat_fname=output_folder+f'ivol_{str(ivol).zfill(3)}.hdf5'

logging.info(f'Output file: {outcat_fname} [runtime {time.time()-t1:.3f} sec]')

if dump:
    dumpcat_folder=f'{output_folder}/pdata/'
    dumpcat_fname=dumpcat_folder+f'ivol_{str(ivol).zfill(3)}_pdata.hdf5'
    if os.path.exists(dumpcat_fname):
        os.remove(dumpcat_fname)
        logging.info(f'Will dump subset of particle data to: {dumpcat_fname} [runtime {time.time()-t1:.3f} sec]')

#subhalo catalogue masking
subcat_limits=get_limits(ivol,nslice,boxsize,buffer=0)
logging.info(f'Box limits: x - ({subcat_limits[0]:.1f},{subcat_limits[1]:.1f}); y - ({subcat_limits[2]:.1f},{subcat_limits[3]:.1f}); z - ({subcat_limits[4]:.1f},{subcat_limits[5]:.1f}) [runtime {time.time()-t1:.3f} sec]')

subcat_snapmask=np.logical_and.reduce([subcat[snap_key].values==snap,subcat[mass_key].values>=(mcut)])
subcat_boxmask=np.logical_and.reduce([subcat['CentreOfPotential_x'].values>=subcat_limits[0],subcat['CentreOfPotential_x'].values<subcat_limits[1],
                                      subcat['CentreOfPotential_y'].values>=subcat_limits[2],subcat['CentreOfPotential_y'].values<subcat_limits[3],
                                      subcat['CentreOfPotential_z'].values>=subcat_limits[4],subcat['CentreOfPotential_z'].values<subcat_limits[5]])

subcat_selection=subcat.loc[np.logical_and(subcat_boxmask,subcat_snapmask),:].copy()
subcat_selection.sort_values(by='SubGroupNumber',ascending=True,inplace=True)
subcat_selection.reset_index(drop=True,inplace=True)

logging.info(f'Mass limit: {np.log10(mcut):.1f} [runtime {time.time()-t1:.3f} sec]')
logging.info(f'Frac above limit: {np.nanmean(subcat_selection[mass_key].values>=mcut)*100:.1f}% [runtime {time.time()-t1:.3f} sec]')

del subcat
numgal=subcat_selection.shape[0]
galaxy_outputs=[]

if numgal:
    logging.info(f'Will generate outputs for {numgal} galaxies in this subvolume [runtime {time.time()-t1:.3f} sec]')

    # Load in particle data
    logging.info(f'Loading snap particle data: {snap_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
    pdata_subvol,kdtree_subvol=read_subvol(snap_pdata_fname,ivol,nslice,metadata,logfile=logging_folder+logging_name+'.log')

    logging.info(f'Coordinate minima: x - {pdata_subvol["Coordinates_x"].min():.2f}, y - {pdata_subvol["Coordinates_y"].min():.2f}, z - {pdata_subvol["Coordinates_z"].min():.2f} [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'Coordinate maxima: x - {pdata_subvol["Coordinates_x"].max():.2f}, y - {pdata_subvol["Coordinates_y"].max():.2f}, z - {pdata_subvol["Coordinates_z"].max():.2f} [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'Temperature min: {np.nanmin(pdata_subvol["Temperature"]):.2e} K')
    logging.info(f'Temperature max: {np.nanmax(pdata_subvol["Temperature"]):.2e} K')
    logging.info(f'Temperature mean: {pdata_subvol["Temperature"].mean():.2e}')
    
    # If dumping pdata and mfracs are in pdata, add to pdata_fields
    if dump:
        for field in pdata_subvol.columns:
            if 'mfrac' in field:
                pdata_fields.append(field)

    logging.info(f'')
    logging.info(f'****** Entering main galaxy loop [runtime {time.time()-t1:.3f} sec] ******')
    logging.info(f'')

    # Main halo loop
    for igal,galaxy in subcat_selection.iterrows():
        logging.info(f'')
        logging.info(f"Galaxy {igal+1}/{subcat_selection.shape[0]:.0f}: subhalo mass - {galaxy[mass_key]:.1e}, sgn - {galaxy['SubGroupNumber']} [runtime {time.time()-t1:.3f} sec]")

        # Initialise outputs
        galaxy_output=pd.DataFrame([])
        galaxy_output.loc[0,'HydroflowID']=np.int64(galaxy[galid_key])
        galaxy_output.loc[0,'ivol']=ivol
        
        central=galaxy['SubGroupNumber']==0
        processgal=True # Process all galaxies (including satellites)

        if processgal:
            
            # Set maximum radius for candidate selection (central: 3.5*r200, satellite: 150 kpc)
            if central:
                maxrad=3.5*galaxy['Group_R_Crit200']
            else:
                maxrad=150e-3 #150 kpc

            # Calculate effective v200
            galaxy_output.loc[0,'r200_eff']=galaxy['Group_R_Crit200']
            galaxy_output.loc[0,'m200_eff']=galaxy['Group_M_Crit200']
            galaxy_output.loc[0,'v200_eff']=np.sqrt(constant_G*galaxy['Group_M_Crit200']/(galaxy['Group_R_Crit200']*afac))

            # Get the particle data for this halo
            t1_c=time.time()
            pdata_candidates=retrieve_galaxy_candidates(galaxy,pdata_subvol,kdtree_subvol,maxrad=maxrad)
            t2_c=time.time()

            if pdata_candidates.shape[0]>0:
                logging.info(f"Candidates: {t2_c-t1_c:.3f} sec (n = {pdata_candidates.shape[0]}, m = {np.nansum(pdata_candidates['Masses'].values):1e})")

            # Process the galaxy if candidates were found
            if pdata_candidates.shape[0]>0:

                #### MAIN GALAXY ANALYSIS ####
                t1_f=time.time()
                galaxy_properties=analyse_galaxy(galaxy,
                                                 pdata_candidates,
                                                 metadata=metadata,
                                                 r200_shells=r200_shells,
                                                 ckpc_shells=ckpc_shells,
                                                 Tbins=Tbins,
                                                 drfac=drfac,
                                                 vcuts=vcuts,
                                                 logfile=logging_folder+logging_name+'.log')
                
                t2_f=time.time()
                logging.info(f"Galaxy routine took: {t2_f-t1_f:.3f} sec")
                logging.info(f'Galaxy successfully processed [runtime {time.time()-t1:.3f} sec]')

                # Add existing properties from subhalo catalogue
                if list(galaxy_properties.keys()):
                    for key in list(galaxy_properties.keys()):
                        galaxy_output.loc[0,key]=galaxy_properties[key]

                # Dump a subset of the particle data if requested
                if dump:
                    logging.info(f'Dumping particle data for galaxy {galaxy[galid_key]} [runtime {time.time()-t1:.3f} sec]')
                    group=str(int(galaxy[galid_key]))
                    data=pdata_candidates.loc[pdata_candidates['ParticleType'].values==0,pdata_fields]
                    dump_hdf_group(dumpcat_fname,group,data,metadata=galaxy_output,verbose=False)
                
            else:
                logging.info(f'Could not process galaxy, could not retrieve candidates')

        else:
            logging.info(f'Did not process galaxy')

        galaxy_outputs.append(galaxy_output)

    logging.info(f'')
    logging.info(f'Finished with loop - concatenating output results [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')

if galaxy_outputs:
    galaxy_outputs=pd.concat(galaxy_outputs,ignore_index=True)
    galaxy_outputs.reset_index(drop=True,inplace=True)

else:
    logging.info(f'No galaxies in this subvolume, empty output [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')
    galaxy_outputs=pd.DataFrame([])

#save
logging.info(f'Saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')

create_dir(outcat_fname)
dump_hdf(outcat_fname,galaxy_outputs)

logging.info(f'Done saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')

