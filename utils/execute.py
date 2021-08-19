import sys
sys.path.append('/home/rwright/Software')

import os
import argparse
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime

parser=argparse.ArgumentParser()
parser.add_argument('--path',metavar='-P',type=str,help='path to subhalo catalogue')
parser.add_argument('--nslice',metavar='-N',type=int,help='number of slices for simulation sub-boxes')
parser.add_argument('--ivol',metavar='-I',type=int,help='which sub-volume to consider')
parser.add_argument('--snap',metavar='-S',type=int,help='which snapshot to consider')
parser.add_argument('--depth',metavar='-D',type=int,help='time interval')

from hydroflow.utils.hpc import create_dir
from hydroflow.physics.math import get_limits,MpcGyr_kms
from hydroflow.physics.gasflow import retrieve_candidates,analyse_gasflow
from hydroflow.physics.galaxy import find_progidx,analyse_galaxy

#arguments
args=parser.parse_args()
pathcat=args.path
path=pathcat.split('cat')[0]
nslice=int(args.nslice)
ivol=int(args.ivol)
snapf=int(args.snap)
depth=int(args.depth)
snapi=int(snapf-depth)

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

#determine sim type
if 'nodeIndex' in list(subcat.columns):
    snap_key='snipshotidx'
    galid_key='nodeIndex'
    progid_key='mainProgenitorIndex'
    descid_key='descendantIndex'
    mass_key='ApertureMeasurements/Mass/030kpc_4'
    from hydroflow.sims.eaglesnip.particle import read_subvol

#metadata
metadata=pd.read_pickle(path+'/redshifts.dat')
snapf_mask=metadata[snap_key].values==snapf;snapf_pdata_fname=metadata.loc[snapf_mask,'path'].values[0]
snapi_mask=metadata[snap_key].values==snapi;snapi_pdata_fname=metadata.loc[snapi_mask,'path'].values[0]
dt=metadata.loc[snapi_mask,'lookbacktime'].values[0]-metadata.loc[snapf_mask,'lookbacktime'].values[0] #gyr
boxsize=metadata.loc[snapf_mask,'boxsize'].values[0]

#outputs
output_folder=f'{path}/catalogues/gasflow/{namecat}/nvol_{str(int(nslice**3)).zfill(3)}/snap_{str(snapf).zfill(3)}_d{str(depth).zfill(2)}/'
outcat_fname=output_folder+f'ivol_{str(ivol).zfill(3)}.hdf5'
logging.info(f'Output file: {outcat_fname} [runtime {time.time()-t1:.3f} sec]')

#subhalo catalogue masking
subcat_limits=get_limits(ivol,nslice,boxsize,buffer=0)
logging.info(f'Box limits: x - ({subcat_limits[0]:.1f},{subcat_limits[1]:.1f}); y - ({subcat_limits[2]:.1f},{subcat_limits[3]:.1f}); z - ({subcat_limits[4]:.1f},{subcat_limits[5]:.1f}) [runtime {time.time()-t1:.3f} sec]')

subcat_snapmask=np.logical_and.reduce([subcat[snap_key].values>=snapi,subcat[snap_key].values<=snapf,subcat[mass_key].values>=1e8])
subcat_boxmask=np.logical_and.reduce([subcat['CentreOfPotential_x'].values>=subcat_limits[0],subcat['CentreOfPotential_x'].values<subcat_limits[1],
                                      subcat['CentreOfPotential_y'].values>=subcat_limits[2],subcat['CentreOfPotential_y'].values<subcat_limits[3],
                                      subcat['CentreOfPotential_z'].values>=subcat_limits[4],subcat['CentreOfPotential_z'].values<subcat_limits[5]])
subcat_selection=subcat.loc[subcat_boxmask,:].copy();del subcat
subcat_selection.reset_index(drop=True,inplace=True)
subcat_selection_final=subcat_selection.loc[subcat_selection[snap_key].values==snapf,:].copy()
subcat_selection_final.reset_index(drop=True,inplace=True)

#load pdata
logging.info(f'Loading final snap particle data: {snapf_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
pdata_snapf,kdtree_snapf=read_subvol(snapf_pdata_fname,ivol,nslice)
logging.info(f'Loading initial snap particle data: {snapi_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
pdata_snapi,kdtree_snapi=read_subvol(snapi_pdata_fname,ivol,nslice)
logging.info(f'')
logging.info(f'****** Entering main galaxy loop [runtime {time.time()-t1:.3f} sec] ******')

#main loop
galaxy_outputs=[]
for igal,galaxy_snapf in subcat_selection_final.iterrows():
    logging.info(f'')
    logging.info(f"Galaxy {igal+1}/{subcat_selection_final.shape[0]:.0f}: stellar mass - {galaxy_snapf[mass_key]:.1e} [runtime {time.time()-t1:.3f} sec]")

    nmin,nmaj,progid=find_progidx(subcat_selection,galaxy_snapf[galid_key],depth)
    
    galaxy_output=pd.DataFrame([])
    galaxy_output.loc[0,'hydroflowID']=galaxy_snapf[galid_key]
    galaxy_output.loc[0,'hydroflowProgID']=progid
    galaxy_output.loc[0,'nmerger_minor']=nmin
    galaxy_output.loc[0,'nmerger_major']=nmaj

    progmatch=progid==subcat_selection[galid_key].values

    if progid and np.nansum(progmatch):
        galaxy_snapi=subcat_selection.loc[progmatch,:].iloc[0]

        pdata_candidates_snapi,pdata_candidates_snapf=retrieve_candidates(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf)
        fitf,galaxy_properties_snapf=analyse_galaxy(galaxy_snapf,pdata_candidates_snapf)
        fiti,galaxy_properties_snapi=analyse_galaxy(galaxy_snapi,pdata_candidates_snapi)

        if fiti and fitf:
            #add outputs
            for key in list(galaxy_properties_snapf.keys()):
                galaxy_output.loc[0,key]=galaxy_properties_snapf[key]
            for key in list(galaxy_properties_snapi.keys()):
                galaxy_output.loc[0,key+'-progen']=galaxy_properties_snapi[key]

            ### r200 facs
            for fac in [0.15,1]:
                gasflow_ir200=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=galaxy_properties_snapf['r200_eff']*fac,dt=dt,Tcut=None)
                for key in list(gasflow_ir200.keys()):
                    galaxy_output.loc[0,f'{fac:.2f}r200-'+key]=gasflow_ir200[key]

            ### barymp
            gasflow_bmp=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=galaxy_properties_snapf['bmp_radius'],dt=dt,Tcut=None)
            for key in list(gasflow_bmp.keys()):
                galaxy_output.loc[0,f'bmp-'+key]=gasflow_bmp[key]

            ### ism
            gasflow_ism=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=galaxy_properties_snapf['bmp_radius'],dt=dt,Tcut=5*10**4)
            for key in list(gasflow_ism.keys()):
                galaxy_output.loc[0,f'ism-'+key]=gasflow_ism[key]

            ### user def
            if 'r_user' in list(galaxy_snapf.keys()):
                for fac in [1,2,3]:
                    gasflow_iuser=analyse_gasflow(pdata_snapi,pdata_snapf,radius=galaxy_properties_snapf['r_user']*fac,dt=dt,Tcut=None)
                    for key in list(gasflow_iuser.keys()):
                        galaxy_output.loc[0,f'{fac:.2f}ruser-'+key]=gasflow_iuser[key]

            logging.info(f'Galaxy successfully processed')

        else:
            logging.info(f'Could not process galaxy, unable to fit bmp')
    else:
        logging.info(f'Could not process galaxy, progenitor lost')


    galaxy_outputs.append(galaxy_output)

logging.info(f'')
logging.info(f'Finished with loop, concatenating output results [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')
galaxy_outputs=pd.concat(galaxy_outputs,ignore_index=True)
galaxy_outputs.reset_index(drop=True,inplace=True)
galaxy_outputs.loc[:,'dt']=dt

#save
logging.info(f'Saving output file [runtime {time.time()-t1:.3f} sec]')
logging.info(f'')
create_dir(outcat_fname)
galaxy_outputs.to_hdf(outcat_fname,key='Gasflow')