# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: routines to analyse reservoir between snapshots, find inflow/outflow particles, and characterise them.

import numpy as np

from hydroflow.src_physics.utils import calc_r200, vel_conversion

def analyse_gasflow(pdata_snapi,pdata_snapf,radius,dt,Tcut=None,idm=False):
    gasflow_output={}

    mass_snap1=pdata_snapi['Mass'].values
    mass_snap2=pdata_snapf['Mass'].values

    Z_snap1=pdata_snapi['Metallicity'].values
    Z_snap2=pdata_snapf['Metallicity'].values

    rcut_snap1=pdata_snapi['R_rel'].values<=radius
    rcut_snap2=pdata_snapf['R_rel'].values<=radius

    gas_snap2=pdata_snapf['ParticleType'].values==0
    gas_snap1=pdata_snapi['ParticleType'].values==0

    star_snap2=pdata_snapf['ParticleType'].values==4
    star_snap1=pdata_snapi['ParticleType'].values==4

    T_snap1=pdata_snapi['Temperature'].values
    T_snap2=pdata_snapf['Temperature'].values

    if Tcut: 
        cool_snap1=T_snap1<=Tcut
        cool_snap2=T_snap2<=Tcut

    else:
        cool_snap1=np.ones(pdata_snapi.shape[0])>0
        cool_snap2=np.ones(pdata_snapf.shape[0])>0

    selection_snap1=np.logical_and.reduce([rcut_snap1,np.logical_or(cool_snap1,star_snap1)])
    selection_snap2=np.logical_and.reduce([rcut_snap2,np.logical_or(cool_snap2,star_snap2)])

    #do DM calcs here
    if idm:
        inflow_mask_dm=np.logical_and.reduce([rcut_snap2,np.logical_not(rcut_snap1),pdata_snapf['ParticleType'].values==1])
        inflow_mass_dm=mass_snap2[inflow_mask_dm]
        gasflow_output['dm-inflow-n']=np.nansum(inflow_mask_dm)
        gasflow_output['dm-inflow-m']=np.nansum(inflow_mass_dm)

        outflow_mask_dm=np.logical_and.reduce([rcut_snap1,np.logical_not(rcut_snap2),pdata_snapf['ParticleType'].values==1])
        outflow_mass_dm=mass_snap2[outflow_mask_dm]
        gasflow_output['dm-outflow-n']=np.nansum(outflow_mask_dm)
        gasflow_output['dm-outflow-m']=np.nansum(outflow_mass_dm)

    #do gas calcs here
    inflow_mask=np.logical_and.reduce([selection_snap2,np.logical_not(selection_snap1),np.logical_or(gas_snap2,gas_snap1)])
    outflow_mask=np.logical_and.reduce([selection_snap1,np.logical_not(selection_snap2),np.logical_or(gas_snap2,gas_snap1)])
    sfr_mask=np.logical_and.reduce([star_snap2,selection_snap2,np.logical_or(np.logical_not(selection_snap1),gas_snap1)])

    #### inflow
    inflow_mass=mass_snap2[inflow_mask]
    gasflow_output['inflow-n']=np.nansum(inflow_mask)
    gasflow_output['inflow-m']=np.nansum(inflow_mass)
    if gasflow_output['inflow-n']>0.:
        gasflow_output['inflow-Z_mean']=np.average(Z_snap2[inflow_mask],weights=inflow_mass)
        gasflow_output['inflow-Z_median']=np.nanmedian(Z_snap2[inflow_mask])
        gasflow_output['inflow-T_mean']=np.average(T_snap2[inflow_mask],weights=inflow_mass)
        gasflow_output['inflow-T_median']=np.nanmedian(T_snap2[inflow_mask])

        #infall vel
        arvel=(pdata_snapf['R_rel'].values[inflow_mask]-pdata_snapi['R_rel'].values[inflow_mask])/dt*vel_conversion
        arvel_mask=np.where(np.logical_and(np.isfinite(arvel),inflow_mass>=0))
        if np.nansum(arvel_mask):
            gasflow_output['inflow-arvel_mean']=np.average(arvel[arvel_mask],weights=inflow_mass[arvel_mask])
            gasflow_output['inflow-arvel_median']=np.nanmedian(arvel[arvel_mask])
    else:
        remove=['inflow-Z_mean','inflow-Z_median','inflow-T_mean','inflow-T_median','inflow-arvel_mean','inflow-arvel_median']
        for field in remove:
            gasflow_output[field]=np.nan

    #### outflow
    outflow_mass=mass_snap1[outflow_mask]
    gasflow_output['outflow-n']=np.nansum(outflow_mask)
    gasflow_output['outflow-m']=np.nansum(outflow_mass)
    if gasflow_output['outflow-n']>0.:
        gasflow_output['outflow-Z_mean']=np.average(Z_snap1[outflow_mask],weights=outflow_mass)
        gasflow_output['outflow-Z_median']=np.nanmedian(Z_snap1[outflow_mask])
        gasflow_output['outflow-T_mean']=np.average(T_snap1[outflow_mask],weights=outflow_mass)
        gasflow_output['outflow-T_median']=np.nanmedian(T_snap1[outflow_mask])
        
        #ejection vel
        arvel=(pdata_snapf['R_rel'].values[outflow_mask]-pdata_snapi['R_rel'].values[outflow_mask])/dt*vel_conversion
        arvel_mask=np.where(np.logical_and(np.isfinite(arvel),outflow_mass>=0))
        if np.nansum(arvel_mask):
            gasflow_output['outflow-arvel_mean']=np.average(arvel[arvel_mask],weights=outflow_mass[arvel_mask])
            gasflow_output['outflow-arvel_median']=np.nanmedian(arvel[arvel_mask])

    else:
        remove=['outflow-Z_mean','outflow-Z_median','outflow-T_mean','outflow-T_median','outflow-arvel_mean','outflow-arvel_median']
        for field in remove:
            gasflow_output[field]=np.nan

    #star formation (mass of new stars)
    sfr_mass=mass_snap2[sfr_mask]
    gasflow_output['ave_SFR-n']=np.nansum(sfr_mask)
    gasflow_output['ave_SFR-m']=np.nansum(sfr_mass)

    return gasflow_output

def candidates_gasflow(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf):

    r200=calc_r200(galaxy_snapf)

    galaxy_com_snapi=np.array([galaxy_snapi[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    galaxy_com_snapf=np.array([galaxy_snapf[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    
    #get gasflow candidates
    rcut=3*r200#choose particles within rcut, which is chosen as 3*r200

    pidx_candidates_snapi=kdtree_snapi.query_ball_point(galaxy_com_snapi[0],rcut)
    pidx_candidates_snapf=kdtree_snapf.query_ball_point(galaxy_com_snapf[0],rcut)

    pids_candidates_snapi=pdata_snapi.loc[pidx_candidates_snapi,'ParticleIDs'].values
    pids_candidates_snapf=pdata_snapf.loc[pidx_candidates_snapf,'ParticleIDs'].values
    
    pid_allcandidates=np.unique(np.concatenate([pids_candidates_snapi,pids_candidates_snapf]))
    bad=False
    try:
        pdata_candidates_idx_snapi=np.searchsorted(pdata_snapi['ParticleIDs'].values,pid_allcandidates)
        pdata_candidates_snapi=pdata_snapi.iloc[pdata_candidates_idx_snapi,:]

    except:
        print('Couldnt get all initial particle candidates')
        bad=True
    try:
        pdata_candidates_idx_snapf=np.searchsorted(pdata_snapf['ParticleIDs'].values,pid_allcandidates)
        pdata_candidates_snapf=pdata_snapi.iloc[pdata_candidates_idx_snapf,:]
    except:
        print('Couldnt get all final particle candidates')
        bad=True

    if not bad:
        print('NO PROBLEM')

        print(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapi)
        print(np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapi),axis=1)))

        pdata_candidates_snapi['R_rel']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapi),axis=1))
        pdata_candidates_snapf['R_rel']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapf),axis=1))

        nomatch_snapi=np.logical_not(pid_allcandidates==pdata_candidates_snapi.loc[:,"ParticleIDs"].values)
        nomatch_snapf=np.logical_not(pid_allcandidates==pdata_candidates_snapf.loc[:,"ParticleIDs"].values)

        pdata_candidates_snapi.loc[nomatch_snapi,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs','Metallicity','R_rel']]=np.nan
        pdata_candidates_snapf.loc[nomatch_snapf,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs','Metallicity','R_rel']]=np.nan

        return True,pdata_candidates_snapi,pdata_candidates_snapf

    else:
        return False,None,None
