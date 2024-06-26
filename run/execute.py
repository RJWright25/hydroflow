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
parser.add_argument('--depth',metavar='-D',type=int,help='snapshot interval for lagrangian calculation (if 0, only eulerian)')
parser.add_argument('--mcut',metavar='-M',type=float,help='mass limit (log mass)')
parser.add_argument('--Tcut',metavar='-T',type=float,help='temperature cut for cool gas')

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
Tcut=10**(args.Tcut)
euleronly=bool(depth==0)

#shells for accretion calculations
drfac=0.25
r200_shells=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,0.25,0.75]
kpc_shells=[10,20,30,40,50,75,100]
vcuts=[0,50,150,250]
vcuts_extra=['0p250vc','0p500vc','1p000vc','1p000vr']

if 'eaglesnip' in code:
    for r200_shell in r200_shells:
        if r200_shell>1:
            r200_shells.remove(r200_shell)

sys.path.append(f"{repo.split('hydroflow')[0]}")

from hydroflow.run.tools_hpc import create_dir
from hydroflow.src_physics.utils import get_limits,get_progidx,constant_G
from hydroflow.src_physics.galaxy import analyse_galaxy
from hydroflow.src_physics.gasflow import analyse_gasflow_lagrangian, analyse_gasflow_eulerian, candidates_gasflow_lagrangian, candidates_gasflow_euleronly

#subhalo catalogue
namecat=pathcat.split('/')[-1][:-5]
run=path.split('/')[-1]
sim=run.split('_')[0]

current_dir=os.getcwd()
create_dir(current_dir+'/jobs/gasflow')
create_dir(current_dir+'/catalogues/gasflow')

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
tracers=False

#determine sim type
if code=='eaglesnip':#eagle snipshots
    from hydroflow.src_sims.eaglesnip.particle import read_subvol
elif code=='eaglesnap':
    from hydroflow.src_sims.eaglesnap.particle import read_subvol
elif code=='camels.simba':
    from hydroflow.src_sims.camels.simba.particle import read_subvol
elif code=='simba':
    from hydroflow.src_sims.simba.particle import read_subvol
elif code=='illustris':
    from hydroflow.src_sims.illustris.particle import read_subvol
    if not euleronly:
        tracers=True

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

logging.info(f'Mass limit: {np.log10(mcut):.1f} [runtime {time.time()-t1:.3f} sec]')
logging.info(f'Frac above limit: {np.nanmean(subcat_selection[mass_key].values>=mcut)*100:.1f}% [runtime {time.time()-t1:.3f} sec]')

del subcat
numgal=subcat_selection_final.shape[0]
galaxy_outputs=[]

if numgal:
    logging.info(f'Will generate outputs for {numgal} galaxies at this snapshot')

    #Check for user requested outputs
    user_radii=[]
    for key in list(subcat_selection_final.keys()):
        if '*' in key:
            user_radii.append(key)

    #Load in final particle data
    logging.info(f'Loading final snap particle data: {snapf_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
    if tracers:
        pdata_snapf,kdtree_snapf,pdata_cells_snapf,kdtree_cells_snapf=read_subvol(snapf_pdata_fname,ivol,nslice)
    else:
        pdata_snapf,kdtree_snapf=read_subvol(snapf_pdata_fname,ivol,nslice)

    #Load in initial particle data
    if not euleronly:
        logging.info(f'Loading initial snap particle data: {snapi_pdata_fname} [runtime {time.time()-t1:.3f} sec]')
        if tracers:
            pdata_snapi,kdtree_snapi,pdata_cells_snapi,kdtree_cells_snapi=read_subvol(snapi_pdata_fname,ivol,nslice)
        else:
            pdata_snapi,kdtree_snapi=read_subvol(snapi_pdata_fname,ivol,nslice)

    logging.info(f'')
    logging.info(f'****** Entering main galaxy loop [runtime {time.time()-t1:.3f} sec] ******')

    file=h5py.File(snapf_pdata_fname)
    hval=file['Header'].attrs['HubbleParam']
    afac=1/(1+file['Header'].attrs['Redshift'])
    file.close()

    subcat_selection_final.loc[:,'hval']=hval
    subcat_selection_final.loc[:,'afac']=afac

    #Main halo loop
    for igal,galaxy_snapf in subcat_selection_final.iterrows():
        logging.info(f'')
        logging.info(f"Galaxy {igal+1}/{subcat_selection_final.shape[0]:.0f}: subhalo mass - {galaxy_snapf[mass_key]:.1e}, sgn - {galaxy_snapf['SubGroupNumber']} [runtime {time.time()-t1:.3f} sec]")

        if not euleronly:
            nmin,nmaj,progid=get_progidx(subcat_selection,galaxy_snapf[galid_key],depth)
        else:
            nmin,nmaj,progid=np.nan,np.nan,np.nan

        #INITIALISE OUTPUTS
        galaxy_output=pd.DataFrame([])
        galaxy_output.loc[0,'HydroflowID']=np.int64(galaxy_snapf[galid_key])
        galaxy_output.loc[0,'HydroflowProgID']=progid
        galaxy_output.loc[0,'nmerger_minor']=nmin
        galaxy_output.loc[0,'nmerger_major']=nmaj
        galaxy_output.loc[0,'ivol']=ivol
        galaxy_output.loc[0,'afac']=afac
        galaxy_output.loc[0,'hval']=hval
        
        progmatch=progid==subcat_selection[galid_key].values
        central=galaxy_snapf['SubGroupNumber']==0

        if euleronly:
            processgal=True
        else:
            processgal=np.nansum(progmatch)

        #CONTINUE IF PROGENITOR FOUND
        if processgal:
            
            if not euleronly:
                galaxy_snapi=subcat_selection.loc[progmatch,:].iloc[0]
            else:
                galaxy_snapi=galaxy_snapf

            #RECORD AVERAGE GALAXY PROPERTIES OVER TIME-STEP
            r200_eff_f=galaxy_snapf['Group_R_Crit200']
            r200_eff_i=galaxy_snapi['Group_R_Crit200']
            r200_eff=(r200_eff_f+r200_eff_i)/2

            m200_eff=(galaxy_snapi['Group_M_Crit200']+galaxy_snapf['Group_M_Crit200'])/2
            v200_eff=np.sqrt(constant_G*m200_eff/(r200_eff*afac))

            galaxy_output.loc[0,'r200_eff']=r200_eff
            galaxy_output.loc[0,'m200_eff']=m200_eff
            galaxy_output.loc[0,'v200_eff']=v200_eff
            galaxy_snapf['v200_eff']=v200_eff

            maxrad=3.5*r200_eff

            #PROCESS FOR CANDIDATES
            t1_c=time.time()

            #RETRIEVE RELEVANT PARTICLES
            if not euleronly:
                success,pdata_candidates_snapi,pdata_candidates_snapf=candidates_gasflow_lagrangian(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf,dt=dt,maxrad=maxrad)
            else:
                success,pdata_candidates_snapi,pdata_candidates_snapf=candidates_gasflow_euleronly(galaxy_snapf,pdata_snapf,kdtree_snapf,maxrad=maxrad)

            
            #RETRIEVE RELEVANT CELLS
            if tracers and not euleronly:
                success_cells,pdata_candidates_cells_snapi,pdata_candidates_cells_snapf=candidates_gasflow_lagrangian(galaxy_snapi,galaxy_snapf,pdata_cells_snapi,kdtree_cells_snapi,pdata_cells_snapf,kdtree_cells_snapf,dt=dt,maxrad=maxrad);success=(success and success_cells)
            t2_c=time.time()
            logging.info(f"Candidates: {t2_c-t1_c:.3f} sec")


            #CONTINUE IF CANDIDATES RETRIEVED
            if success:
                #### CHARACTERISE GALAXY
                t1_f=time.time()
                if tracers:#if have tracers, use cells for galaxy analysis
                    fitf,galaxy_properties_snapf=analyse_galaxy(galaxy_snapf,pdata_candidates_cells_snapf,Tcut,r200_shells=r200_shells[:-2])
                else:
                    fitf,galaxy_properties_snapf=analyse_galaxy(galaxy_snapf,pdata_candidates_snapf,Tcut,r200_shells=r200_shells[:-2])
                
                if fitf:
                    #add galaxy outputs
                    for key in list(galaxy_properties_snapf.keys()):
                        galaxy_output.loc[0,key]=galaxy_properties_snapf[key]

                t2_f=time.time()
                logging.info(f"Galaxy: {t2_f-t1_f:.3f} sec")

                #### CHARACTERISE GAS FLOW
                t1_g=time.time()
                
                #select particle data for eulerian calculation (i.e. switch to cells if appropriate)
                pdata_euler_snapi=pdata_candidates_snapi
                pdata_euler_snapf=pdata_candidates_snapf
                if tracers:
                    pdata_euler_snapi=pdata_candidates_cells_snapi
                    pdata_euler_snapf=pdata_candidates_cells_snapf
                    
                ### Lagrangian ISM calculation
                if not euleronly:
                    gasflow_ism=analyse_gasflow_lagrangian(galaxy_snapf,pdata_candidates_snapi,pdata_candidates_snapf,radius=r200_eff*0.2,dt=dt,vcuts=vcuts,vcuts_extra=vcuts_extra,Tcut=Tcut)
                    for key in list(gasflow_ism.keys()):
                        galaxy_output.loc[0,f'0p20r200_coolgas-'+key]=gasflow_ism[key]
                
                gasflow_ism_euler=analyse_gasflow_eulerian(galaxy_snapf,pdata_candidates_snapf,radius=r200_eff*0.2,vcuts=vcuts,vcuts_extra=vcuts_extra,drfac=drfac,Tcut=Tcut)
                for key in list(gasflow_ism_euler.keys()):
                    galaxy_output.loc[0,f'0p20r200_coolgas-'+key]=gasflow_ism_euler[key]

                for fac in r200_shells:
                    #lagrange
                    if not euleronly:
                        gasflow_ir200_lagrange=analyse_gasflow_lagrangian(galaxy_snapf,pdata_candidates_snapi,pdata_candidates_snapf,dt=dt,radius=r200_eff*fac,vcuts=vcuts,vcuts_extra=vcuts_extra[:2])
                        for key in list(gasflow_ir200_lagrange.keys()):
                            galaxy_output.loc[0,f'{fac:.2f}r200_gas-'.replace('.','p')+key]=gasflow_ir200_lagrange[key]
                    
                    #euler
                    gasflow_ir200_euler=analyse_gasflow_eulerian(galaxy_snapf,pdata_euler_snapf,radius=r200_eff*fac,vcuts=vcuts,vcuts_extra=vcuts_extra,drfac=drfac)
                    for key in list(gasflow_ir200_euler.keys()):
                        galaxy_output.loc[0,f'{fac:.2f}r200_gas-'.replace('.','p')+key]=gasflow_ir200_euler[key]

                ### comoving & physical units
                for rad in kpc_shells:
                    #lagrange
                    if not euleronly:
                        gasflow_irad_lagrange=analyse_gasflow_lagrangian(galaxy_snapf,pdata_candidates_snapi,pdata_candidates_snapf,radius=(rad*1e-3)*hval,dt=dt,vcuts=vcuts,vcuts_extra=vcuts_extra[:2])
                        for key in list(gasflow_irad_lagrange.keys()):
                            galaxy_output.loc[0,f'{str(int(rad)).zfill(3)}ckpc_gas-'+key]=gasflow_irad_lagrange[key]

                    #euler
                    gasflow_irad_euler=analyse_gasflow_eulerian(galaxy_snapf,pdata_euler_snapf,radius=(rad*1e-3)*hval,vcuts=vcuts,vcuts_extra=vcuts_extra,drfac=drfac)
                    for key in list(gasflow_irad_euler.keys()):
                        galaxy_output.loc[0,f'{str(int(rad)).zfill(3)}ckpc_gas-'+key]=gasflow_irad_euler[key]

                    #lagrange
                    if not euleronly:
                        gasflow_irad_lagrange=analyse_gasflow_lagrangian(galaxy_snapf,pdata_candidates_snapi,pdata_candidates_snapf,radius=(rad*1e-3)*hval/afac,dt=dt,vcuts=vcuts,vcuts_extra=vcuts_extra[:2])
                        for key in list(gasflow_irad_lagrange.keys()):
                            galaxy_output.loc[0,f'{str(int(rad)).zfill(3)}pkpc_gas-'+key]=gasflow_irad_lagrange[key]

                    #euler
                    gasflow_irad_euler=analyse_gasflow_eulerian(galaxy_snapf,pdata_euler_snapf,radius=(rad*1e-3)*hval/afac,vcuts=vcuts,vcuts_extra=vcuts_extra,drfac=drfac)
                    for key in list(gasflow_irad_euler.keys()):
                        galaxy_output.loc[0,f'{str(int(rad)).zfill(3)}pkpc_gas-'+key]=gasflow_irad_euler[key]

                # ### user def
                # for user_radius in user_radii:
                #     iuser_radius=galaxy_snapf[user_radius]
                #     gasflow_iuser=analyse_gasflow(pdata_candidates_snapi,pdata_candidates_snapf,radius=iuser_radius,dt=dt,Tcut=None,afac=afac)
                #     for key in list(gasflow_iuser.keys()):
                #         galaxy_output.loc[0,f'{user_radius}-'+key]=gasflow_iuser[key]
                    
                t2_g=time.time()
                logging.info(f"Gasflow: {t2_g-t1_g:.3f} sec")
                logging.info(f'Galaxy successfully processed')
                
            else:
                logging.info(f'Could not process galaxy, could not retrieve candidates')

        else:
            logging.info(f'Did not process galaxy, progenitor lost or is satellite')

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

