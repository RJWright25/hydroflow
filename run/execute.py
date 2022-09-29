# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/execute.py: script to submit job array.

import os
import sys
import time
import logging
import argparse

import h5py
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
#arguments

parser=argparse.ArgumentParser()
parser.add_argument('--repo',metavar='-R',type=str,help='where is repo')
parser.add_argument('--code',metavar='-C',type=str,help='which simulation (from hydroflow.src_sims)')
parser.add_argument('--path',metavar='-P',type=str,help='path to subhalo catalogue')
parser.add_argument('--nslice',metavar='-N',type=int,help='number of slices for simulation sub-boxes')
parser.add_argument('--ivol',metavar='-I',type=int,help='which sub-volume to consider')
parser.add_argument('--snap',metavar='-S',type=int,help='which snapshot to consider')
parser.add_argument('--depth',metavar='-D',type=int,help='time interval')
parser.add_argument('--mcut',metavar='-M',type=float,help='mass limit (log mass)')

args=parser.parse_args()
repo=args.repo
code=args.code
pathcat=args.path
path=pathcat.split('cat')[0]
nslice=int(args.nslice)
ivol=int(args.ivol)
snapf=int(args.snap)
depth=int(args.depth)
snapi=int(snapf-depth)
mcut=10**(args.mcut)

sys.path.append(f"{repo.split('hydroflow')[0]}")

from hydroflow.run.tools_hpc import create_dir
from hydroflow.src_physics.utils import get_limits,get_progidx,constant_G
from hydroflow.src_physics.galaxy import analyse_galaxy,calc_r200
from hydroflow.src_physics.gasflow import candidates_gasflow,analyse_gasflow

#subhalo catalogue
namecat=pathcat.split('/')[-1][:-5]
run=path.split('/')[-1]
sim=run.split('_')[0]

#logging file
t1=time.time()
logging_folder=f'{path}/jobs/gasflow/{namecat}/nvol_{str(int(nslice**3)).zfill(3)}/snap{str(snapf).zfill(3)}_d{str(depth).zfill(2)}/'
logging_name=f"s{str(snapf).zfill(3)}_d{str(depth).zfill(2)}_n{str(int(nslice**3)).zfill(3)}_ivol{ivol}"
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
subcat=pd.read_hdf(pathcat,key='Subhalo')
logging.info(f'Subhalo catalogue loaded [runtime {time.time()-t1:.3f} sec]')
logging.info(f'Masking catalogue and processing metadata [runtime {time.time()-t1:.3f} sec]')


snap_key='SnapNum'
galid_key='GalaxyID'
descid_key='DescendantID'
mass_key='Mass'

#determine sim type
if code=='eaglesnip':#eagle snipshots
    from hydroflow.src_sims.eaglesnip.particle import read_subvol
elif code=='eaglesnap':
    from hydroflow.src_sims.eaglesnap.particle import read_subvol
elif code=='camels.simba':
    from hydroflow.src_sims.camels.simba.particle import read_subvol
elif code=='illustris':
    from hydroflow.src_sims.illustris.particle import read_subvol


#metadata
metadata=pd.read_pickle(path+'/redshifts.dat')
snapf_mask=metadata[snap_key].values==snapf;snapf_pdata_fname=metadata.loc[snapf_mask,'Path'].values[0]
snapi_mask=metadata[snap_key].values==snapi;snapi_pdata_fname=metadata.loc[snapi_mask,'Path'].values[0]
dt=metadata.loc[snapi_mask,'LookbackTime'].values[0]-metadata.loc[snapf_mask,'LookbackTime'].values[0] #gyr
boxsize=metadata.loc[snapf_mask,'BoxSize'].values[0]

#outputs
output_folder=f'{path}/catalogues/gasflow/{namecat}/nvol_{str(int(nslice**3)).zfill(3)}/snap{str(snapf).zfill(3)}_d{str(depth).zfill(2)}/'
outcat_fname=output_folder+f'ivol_{str(ivol).zfill(3)}.hdf5'
logging.info(f'Output file: {outcat_fname} [runtime {time.time()-t1:.3f} sec]')

#subhalo catalogue masking
subcat_limits=get_limits(ivol,nslice,boxsize,buffer=0)
logging.info(f'Box limits: x - ({subcat_limits[0]:.1f},{subcat_limits[1]:.1f}); y - ({subcat_limits[2]:.1f},{subcat_limits[3]:.1f}); z - ({subcat_limits[4]:.1f},{subcat_limits[5]:.1f}) [runtime {time.time()-t1:.3f} sec]')


subcat_snapmask=np.logical_and.reduce([subcat[snap_key].values>=snapi,subcat[snap_key].values<=snapf,subcat[mass_key].values>=(mcut*0.25)])
subcat_boxmask=np.logical_and.reduce([subcat['CentreOfPotential_x'].values>=subcat_limits[0],subcat['CentreOfPotential_x'].values<subcat_limits[1],
                                      subcat['CentreOfPotential_y'].values>=subcat_limits[2],subcat['CentreOfPotential_y'].values<subcat_limits[3],
                                      subcat['CentreOfPotential_z'].values>=subcat_limits[4],subcat['CentreOfPotential_z'].values<subcat_limits[5]])
subcat_selection=subcat.loc[np.logical_and(subcat_boxmask,subcat_snapmask),:].copy()
subcat_selection.reset_index(drop=True,inplace=True)
subcat_selection_final=subcat_selection.loc[np.logical_and(subcat_selection[snap_key].values==snapf,subcat_selection[mass_key].values>=mcut),:].copy()
subcat_selection_final.reset_index(drop=True,inplace=True)

del subcat
numgal=subcat_selection_final.shape[0]
galaxy_outputs=[]

if numgal:
    #check for user requested outputs
    user_radii=[]
    for key in list(subcat_selection_final.keys()):
        if '*' in key:
            user_radii.append(key)
    logging.info(f'Will generate outputs for {numgal} galaxies at this snapshot')

    #load pdata
    logging.info(f'Loading final snap particle data: {snapf_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
    pdata_snapf,kdtree_snapf=read_subvol(snapf_pdata_fname,ivol,nslice)


    # pdata_snapf.sort_values("ParticleIDs",inplace=True)
    # pdata_snapf.reset_index(inplace=True,drop=True)
    logging.info(f'Loading initial snap particle data: {snapi_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
    pdata_snapi,kdtree_snapi=read_subvol(snapi_pdata_fname,ivol,nslice)

    # pdata_snapf.sort_values("ParticleIDs",inplace=True)
    # pdata_snapf.reset_index(inplace=True,drop=True)
    logging.info(f'')
    logging.info(f'****** Entering main galaxy loop [runtime {time.time()-t1:.3f} sec] ******')

    file=h5py.File(snapf_pdata_fname)
    hval=file['Header'].attrs['HubbleParam']
    afac=(1/1+file['Header'].attrs['Redshift'])
    file.close()

    #main loop
    for igal,galaxy_snapf in subcat_selection_final.iterrows():
        logging.info(f'')
        logging.info(f"Galaxy {igal+1}/{subcat_selection_final.shape[0]:.0f}: subhalo mass - {galaxy_snapf[mass_key]:.1e}, sgn - {galaxy_snapf['SubGroupNumber']} [runtime {time.time()-t1:.3f} sec]")

        nmin,nmaj,progid=get_progidx(subcat_selection,galaxy_snapf[galid_key],depth)

        
        galaxy_output=pd.DataFrame([])
        galaxy_output.loc[0,'HydroflowID']=galaxy_snapf[galid_key]
        galaxy_output.loc[0,'HydroflowProgID']=progid
        galaxy_output.loc[0,'nmerger_minor']=nmin
        galaxy_output.loc[0,'nmerger_major']=nmaj
        galaxy_output.loc[0,'ivol']=ivol

        progmatch=progid==subcat_selection[galid_key].values

        if progid and np.nansum(progmatch):

            galaxy_snapi=subcat_selection.loc[progmatch,:].iloc[0]

            r200_eff_f=calc_r200(galaxy_snapf)
            r200_eff_i=calc_r200(galaxy_snapi)
            r200_eff=(r200_eff_f+r200_eff_i)/2
            m200_eff=(galaxy_snapi['Group_M_Crit200']+galaxy_snapf['Group_M_Crit200'])/2

            galaxy_output.loc[0,'r200_eff']=r200_eff
            galaxy_output.loc[0,'m200_eff']=m200_eff

            t1_c=time.time()
            success,pdata_candidates_snapi,pdata_candidates_snapf=candidates_gasflow(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf)
            t2_c=time.time()
            logging.info(f"Candidates: {t2_c-t1_c:.3f} sec")

            if success:
                t1_f=time.time()
                fitf,galaxy_properties_snapf=analyse_galaxy(galaxy_snapf,pdata_candidates_snapf)
                t2_f=time.time()
                logging.info(f"Galaxy: {t2_f-t1_f:.3f} sec")

                t1_g=time.time()
                if fitf:
                    #add galaxy outputs
                    for key in list(galaxy_properties_snapf.keys()):
                        galaxy_output.loc[0,key]=galaxy_properties_snapf[key]
                else:
                    logging.info(f'Could not determine properties of galaxy')

                veject=0.5*np.sqrt(constant_G*m200_eff/r200_eff)*hval/afac


                ### ism
                gasflow_ism=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=r200_eff*0.15,dt=dt,Tcut=5*10**4,veject=veject)
                for key in list(gasflow_ism.keys()):
                    galaxy_output.loc[0,f'0p15r200_coolgas-'+key]=gasflow_ism[key]


                ### r200 facs
                for fac in [0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]:
                    idm=(fac>=1)
                    gasflow_ir200=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=r200_eff*fac,dt=dt,Tcut=None,idm=idm,veject=veject)
                    for key in list(gasflow_ir200.keys()):
                        
                        if not 'dm' in key:
                            galaxy_output.loc[0,f'{fac:.2f}r200_gas-'.replace('.','p')+key]=gasflow_ir200[key]
                        else:
                            galaxy_output.loc[0,f'{fac:.2f}r200_dm-'.replace('.','p')+key[3:]]=gasflow_ir200[key]


                ### user def
                for user_radius in user_radii:
                    iuser_radius=galaxy_snapf[user_radius]
                    gasflow_iuser=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=iuser_radius,dt=dt,Tcut=None,veject=veject)
                    for key in list(gasflow_iuser.keys()):
                        galaxy_output.loc[0,f'{user_radius}-'+key]=gasflow_iuser[key]
                    
                t2_g=time.time()
                logging.info(f"Gasflow: {t2_g-t1_g:.3f} sec")
                logging.info(f'Galaxy successfully processed')
            else:
                logging.info(f'Could not process galaxy, could not retrieve candidates')

        else:
            logging.info(f'Could not process galaxy, progenitor lost')

        galaxy_outputs.append(galaxy_output)

    logging.info(f'')
    logging.info(f'Finished with loop - concatenating output results [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')

if galaxy_outputs:
    galaxy_outputs=pd.concat(galaxy_outputs,ignore_index=True)
    galaxy_outputs.reset_index(drop=True,inplace=True)
    galaxy_outputs.loc[:,'dt']=dt
else:
    logging.info(f'No galaxies in this subvolume, empty output [runtime {time.time()-t1:.3f} sec]')
    logging.info(f'')
    galaxy_outputs=pd.DataFrame([])

#save
logging.info(f'Saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')

create_dir(outcat_fname)
galaxy_outputs.to_hdf(outcat_fname,key='Gasflow')

logging.info(f'Done saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')

