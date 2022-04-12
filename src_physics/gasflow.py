
import numpy as np

from hydroflow.src_physics.utils import calc_r200, vel_conversion

def analyse_gasflow(pdata_snapi,pdata_snapf,radius,dt,Tcut=None):
    mass_snap1=pdata_snapi['Mass'].values
    mass_snap2=pdata_snapf['Mass'].values

    Z_snap1=pdata_snapi['Metallicity'].values
    Z_snap2=pdata_snapf['Metallicity'].values

    T_snap1=pdata_snapi['Temperature'].values
    T_snap2=pdata_snapf['Temperature'].values

    rcut_snap1=pdata_snapi['R_rel'].values<=radius
    rcut_snap2=pdata_snapf['R_rel'].values<=radius

    ocut_snap1=pdata_snapi['R_rel'].values>radius
    ocut_snap2=pdata_snapf['R_rel'].values>radius

    gas_snap2=pdata_snapf['ParticleType'].values==0
    gas_snap1=pdata_snapi['ParticleType'].values==0

    star_snap2=pdata_snapf['ParticleType'].values==4
    star_snap1=pdata_snapi['ParticleType'].values==4
    
    if not Tcut:
        cool_snap1=np.ones(pdata_snapi.shape[0])>0
        cool_snap2=np.ones(pdata_snapf.shape[0])>0
        hot_snap1=np.zeros(pdata_snapi.shape[0])>0
        hot_snap2=np.zeros(pdata_snapf.shape[0])>0
    else:
        cool_snap1=T_snap1<=Tcut
        cool_snap2=T_snap2<=Tcut
        hot_snap1=T_snap1>Tcut
        hot_snap2=T_snap2>Tcut

    selection_snap1=np.logical_and.reduce([rcut_snap1,np.logical_or(cool_snap1,star_snap1)])
    selection_snap2=np.logical_and.reduce([rcut_snap2,np.logical_or(cool_snap2,star_snap2)])

    inflow_mask=np.logical_and.reduce([selection_snap2,np.logical_not(selection_snap1),np.logical_or(gas_snap2,gas_snap1)])
    outflow_mask=np.logical_and.reduce([selection_snap1,np.logical_not(selection_snap2),np.logical_or(gas_snap2,gas_snap1)])
    sfr_mask=np.logical_and.reduce([gas_snap1,star_snap2,selection_snap2])

    gasflow_output={}

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
        mask=np.where(np.logical_and(np.isfinite(arvel),inflow_mass>=0))
        if np.nansum(mask):
            gasflow_output['inflow-arvel_mean']=np.average(arvel[mask],weights=inflow_mass[mask])
            gasflow_output['inflow-arvel_median']=np.nanmedian(arvel)

        #reason for inflow
        cooled=np.logical_and.reduce([cool_snap2[inflow_mask],hot_snap1[inflow_mask]])
        radin=np.logical_and.reduce([rcut_snap2[inflow_mask],ocut_snap1[inflow_mask]])
        gasflow_output['inflow-f_cooled_only']=np.average(np.logical_and(cooled,np.logical_not(radin)),weights=mass_snap2[inflow_mask])
        gasflow_output['inflow-f_infall_only']=np.average(np.logical_and(radin,np.logical_not(cooled)),weights=mass_snap2[inflow_mask])
        gasflow_output['inflow-f_both']=np.average(np.logical_and(radin,cooled),weights=mass_snap2[inflow_mask])
        gasflow_output['inflow-f_neither']=np.average(np.logical_not(np.logical_or(radin,cooled)),weights=mass_snap2[inflow_mask])
    else:
        for field in ['inflow-f_cooled_only','inflow-f_infall_only','inflow-f_both','inflow-f_neither','inflow-Z_mean','inflow-Z_median']:
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
        mask=np.where(np.logical_and(np.isfinite(arvel),outflow_mass>=0))
        if np.nansum(mask):
            gasflow_output['outflow-arvel_mean']=np.average(arvel[mask],weights=outflow_mass[mask])
            gasflow_output['outflow-arvel_median']=np.nanmedian(arvel)

        #reason for outflow
        heated=np.logical_and.reduce([cool_snap1[outflow_mask],hot_snap2[outflow_mask]])
        radout=np.logical_and.reduce([rcut_snap1[outflow_mask],ocut_snap2[outflow_mask]])
        gasflow_output['outflow-f_heated_only']=np.average(np.logical_and(heated,np.logical_not(radout)),weights=outflow_mass)
        gasflow_output['outflow-f_ejected_only']=np.average(np.logical_and(radout,np.logical_not(heated)),weights=outflow_mass)
        gasflow_output['outflow-f_both']=np.average(np.logical_and(radout,heated),weights=outflow_mass)
        gasflow_output['outflow-f_neither']=np.average(np.logical_not(np.logical_or(radout,heated)),weights=outflow_mass)
    else:
        for field in ['outflow-f_heated_only','outflow-f_ejected_only','outflow-f_both','outflow-f_neither','outflow-Z_mean','outflow-Z_median']:
            gasflow_output[field]=np.nan


    #star formation
    sfr_mass=mass_snap2[sfr_mask]
    sfr_inflow_mass=mass_snap2[np.logical_and(inflow_mask,sfr_mask)]
    gasflow_output['ave_SFR-n']=np.nansum(sfr_mask)
    gasflow_output['ave_SFR-m']=np.nansum(sfr_mass)
    gasflow_output['directSFR-frac']=np.nansum(sfr_inflow_mass)/gasflow_output['inflow-m']

    return gasflow_output

def candidates_gasflow(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf):
    r200=calc_r200(galaxy_snapf)
    galaxy_com_snapi=np.array([galaxy_snapi[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    galaxy_com_snapf=np.array([galaxy_snapf[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    
    #get gasflow candidates
    pidx_candidates_snapi=kdtree_snapi.query_ball_point(galaxy_com_snapi[0],r200)
    pidx_candidates_snapf=kdtree_snapf.query_ball_point(galaxy_com_snapf[0],r200)

    pids_candidates_snapi=pdata_snapi.loc[pidx_candidates_snapi,'ParticleIDs'].values
    pids_candidates_snapf=pdata_snapf.loc[pidx_candidates_snapf,'ParticleIDs'].values
    
    pid_allcandidates=np.unique(np.concatenate([pids_candidates_snapi,pids_candidates_snapf]))

    try:
        pdata_candidates_snapi=pdata_snapi.loc[pdata_snapi['ParticleIDs'].searchsorted(pid_allcandidates),:]
        pdata_candidates_snapf=pdata_snapf.loc[pdata_snapf['ParticleIDs'].searchsorted(pid_allcandidates),:]
    except:
        return False,None,None

    pdata_candidates_snapi['R_rel']=np.sqrt(np.sum(np.square(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapi),axis=1))
    pdata_candidates_snapf['R_rel']=np.sqrt(np.sum(np.square(pdata_candidates_snapf.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapf),axis=1))

    nomatch_snapi=np.logical_not(pid_allcandidates==pdata_candidates_snapi.loc[:,"ParticleIDs"].values)
    nomatch_snapf=np.logical_not(pid_allcandidates==pdata_candidates_snapf.loc[:,"ParticleIDs"].values)

    pdata_candidates_snapi.loc[nomatch_snapi,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs','Metallicity','R_rel']]=np.nan
    pdata_candidates_snapf.loc[nomatch_snapf,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs','Metallicity','R_rel']]=np.nan

    return True,pdata_candidates_snapi,pdata_candidates_snapf