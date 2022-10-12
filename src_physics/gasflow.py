# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: routines to analyse reservoir between snapshots, find inflow/outflow particles, and characterise them.

import numpy as np
import time

from hydroflow.src_physics.utils import calc_r200, vel_conversion

def analyse_gasflow(pdata_snapi,pdata_snapf,radius,dt,vc=0,Tcut=None,idm=False):
    gasflow_output={}

    t1=time.time()
    mass_snap1=pdata_snapi['Mass'].values
    mass_snap2=pdata_snapf['Mass'].values

    Z_snap1=pdata_snapi['Metallicity'].values
    Z_snap2=pdata_snapf['Metallicity'].values

    rcut_snap1=pdata_snapi['R_rel'].values<=radius
    rcut_snap2=pdata_snapf['R_rel'].values<=radius

    ################
    gas_snap2=pdata_snapf['ParticleType'].values==0
    gas_snap1=pdata_snapi['ParticleType'].values==0

    if 'StellarFormationTime' in pdata_snapi:
        gas_snap2=np.logical_or(gas_snap2,pdata_snapf['StellarFormationTime'].values<=0)
        gas_snap1=np.logical_or(gas_snap1,pdata_snapi['StellarFormationTime'].values<=0)

    star_snap2=pdata_snapf['ParticleType'].values==4
    star_snap1=pdata_snapi['ParticleType'].values==4

    if 'StellarFormationTime' in pdata_snapi:
        star_snap2=np.logical_and(star_snap2,pdata_snapf['StellarFormationTime'].values>0)
        star_snap1=np.logical_and(star_snap1,pdata_snapi['StellarFormationTime'].values>0)

    T_snap1=pdata_snapi['Temperature'].values
    T_snap2=pdata_snapf['Temperature'].values

    #
    arvel=(pdata_snapf['R_rel'].values-pdata_snapi['R_rel'].values)/dt*vel_conversion
    arvel[np.logical_not(np.isfinite(arvel))]=1e5

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
        gasflow_output['dm-inflow-m']=np.nansum(inflow_mass_dm)/dt

        outflow_mask_dm=np.logical_and.reduce([rcut_snap1,np.logical_not(rcut_snap2),pdata_snapf['ParticleType'].values==1])
        outflow_mass_dm=mass_snap2[outflow_mask_dm]
        gasflow_output['dm-outflow-n']=np.nansum(outflow_mask_dm)
        gasflow_output['dm-outflow-m']=np.nansum(outflow_mass_dm)/dt

    print(f'{time.time()-t1:.3f} seconds for dm calcs')

    #do gas calcs here
    inflow_mask=np.logical_and.reduce([selection_snap2,np.logical_not(selection_snap1),np.logical_or(gas_snap2,gas_snap1)])
    outflow_mask=np.logical_and.reduce([selection_snap1,np.logical_not(selection_snap2),np.logical_or(gas_snap2,gas_snap1)])

    ## pristine
    inflow_pristine_mask=np.logical_and(inflow_mask,np.logical_or(Z_snap2<1e-4,Z_snap1<1e-4))

    #vcuts
    vcuts=['000kmps','50kmps','100kmps','200kmps','0p50vc','1p00vc','2p00vc']
    vcuts_val=[0,50,100,150,250,0.25*vc,0.5*vc,vc,2*vc]
    outflow_masks={vcut:np.logical_and.reduce([outflow_mask,arvel>=vcut_val]) for vcut,vcut_val in zip(vcuts,vcuts_val)}

    #### inflow
    for name,mask in zip(['inflow','inflow_pristine'],[inflow_mask,inflow_pristine_mask]):
        inflow_mass=mass_snap2[mask]
        gasflow_output[f'{name}-n']=np.nansum(mask)
        gasflow_output[f'{name}-m']=np.nansum(inflow_mass)/dt
        if gasflow_output[f'{name}-n']>0.:
            gasflow_output[f'{name}-Z_mean']=np.average(Z_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-Z_median']=np.nanmedian(Z_snap2[mask])
            gasflow_output[f'{name}-T_mean']=np.average(T_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-T_median']=np.nanmedian(T_snap2[mask])

            #infall vel
            arvel_inflow=arvel[mask]
            arvel_mask=np.where(np.logical_and(np.isfinite(arvel_inflow),inflow_mass>=0))
            if np.nansum(arvel_mask):
                gasflow_output[f'{name}-arvel_mean']=np.average(arvel_inflow[arvel_mask],weights=inflow_mass[arvel_mask])
                gasflow_output[f'{name}-arvel_median']=np.nanmedian(arvel_inflow[arvel_mask])
        else:
            remove=[f'{name}-Z_mean',f'{name}-Z_median',f'{name}-T_mean',f'{name}-T_median',f'{name}-arvel_mean',f'{name}-arvel_median']
            for field in remove:
                gasflow_output[field]=np.nan


    #### outflows
    for vcut,vcut_val in zip(vcuts,vcuts_val):
        ejected_mask=outflow_masks[vcut]
        outflow_mass=mass_snap1[ejected_mask]
        gasflow_output[f'{vcut}_outflow-n']=np.nansum(ejected_mask)
        gasflow_output[f'{vcut}_outflow-m']=np.nansum(outflow_mass)/dt
        if gasflow_output[f'{vcut}_outflow-n']>0.:
            gasflow_output[f'{vcut}_outflow-Z_mean']=np.average(Z_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut}_outflow-Z_median']=np.nanmedian(Z_snap1[ejected_mask])
            gasflow_output[f'{vcut}_outflow-T_mean']=np.average(T_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut}_outflow-T_median']=np.nanmedian(T_snap1[ejected_mask])
            
            #ejection vel
            arvel_ejected=arvel[ejected_mask]
            arvel_mask=np.where(np.logical_and(np.isfinite(arvel_ejected),outflow_mass>=0))
            if np.nansum(arvel_mask):
                gasflow_output[f'{vcut}_outflow-arvel_mean']=np.average(arvel_ejected[arvel_mask],weights=outflow_mass[arvel_mask])
                gasflow_output[f'{vcut}_outflow-arvel_median']=np.nanmedian(arvel_ejected[arvel_mask])
                gasflow_output[f'{vcut}_outflow-arvel_05P']=np.nanpercentile(arvel_ejected[arvel_mask],5)
                gasflow_output[f'{vcut}_outflow-arvel_95P']=np.nanpercentile(arvel_ejected[arvel_mask],95)

        else:
            remove=[f'{vcut}_outflow-Z_mean',f'{vcut}_outflow-Z_median',f'{vcut}_outflow-T_mean',f'{vcut}_outflow-T_median',f'{vcut}_outflow-arvel_mean',f'{vcut}_outflow-arvel_median',f'{vcut}_outflow-arvel_05P',f'{vcut}_outflow-arvel_95P']
            for field in remove:
                gasflow_output[field]=np.nan

    return gasflow_output

def candidates_gasflow(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf):

    r200=calc_r200(galaxy_snapf)

    galaxy_com_snapi=np.array([galaxy_snapi[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    galaxy_com_snapf=np.array([galaxy_snapf[f'CentreOfPotential_{x}'] for x in 'xyz'],ndmin=2)
    
    #get gasflow candidates
    rcut=2.5*r200#choose particles within rcut, which is chosen as 3*r200

    pidx_candidates_snapi=kdtree_snapi.query_ball_point(galaxy_com_snapi[0],rcut)
    pidx_candidates_snapf=kdtree_snapf.query_ball_point(galaxy_com_snapf[0],rcut)

    pids_candidates_snapi=pdata_snapi.loc[pidx_candidates_snapi,'ParticleIDs'].values
    pids_candidates_snapf=pdata_snapf.loc[pidx_candidates_snapf,'ParticleIDs'].values
    
    pid_allcandidates=np.unique(np.concatenate([pids_candidates_snapi,pids_candidates_snapf]))

    pids_candidates_snapi_forcheck=np.concatenate([pdata_snapi['ParticleIDs'].values,[-1]]) 
    pids_candidates_snapf_forcheck=np.concatenate([pdata_snapf['ParticleIDs'].values,[-1]])

    pdata_candidates_idx_snapi=np.searchsorted(pdata_snapi['ParticleIDs'].values,pid_allcandidates)
    pdata_candidates_idx_snapf=np.searchsorted(pdata_snapf['ParticleIDs'].values,pid_allcandidates)

    #### if ID not found, will come up  here 
    pdata_candidates_idx_snapi_incorrectlyextracted=pids_candidates_snapi_forcheck[(pdata_candidates_idx_snapi,)]!=pid_allcandidates
    pdata_candidates_idx_snapf_incorrectlyextracted=pids_candidates_snapf_forcheck[(pdata_candidates_idx_snapf,)]!=pid_allcandidates
    
    
    if np.nansum(pdata_candidates_idx_snapi_incorrectlyextracted):
        print(f"{np.nanmean(pdata_candidates_idx_snapi_incorrectlyextracted)*100:.3f}% of candidates not in fof at initial snap")
    if np.nansum(pdata_candidates_idx_snapf_incorrectlyextracted):
        print(f"{np.nanmean(pdata_candidates_idx_snapf_incorrectlyextracted)*100:.3f}% of candidates not in fof at final snap")
        
    bad=False
    
    try:
        pdata_candidates_snapi=pdata_snapi.iloc[pdata_candidates_idx_snapi,:]

    except:
        print(np.nansum(pdata_candidates_idx_snapi==np.nanmax(pdata_candidates_idx_snapi)))
        print(np.nanmean(pdata_candidates_idx_snapi==np.nanmax(pdata_candidates_idx_snapi)))
        print('Couldnt get all initial particle candidates')
        bad=True

    try:
        pdata_candidates_snapf=pdata_snapf.iloc[pdata_candidates_idx_snapf,:]
    except:
        print(np.nansum(pdata_candidates_idx_snapf==np.nanmax(pdata_candidates_idx_snapf)))
        print(np.nanmean(pdata_candidates_idx_snapf==np.nanmax(pdata_candidates_idx_snapf)))
        print('Couldnt get all final particle candidates')
        bad=True

    if not bad:
        pdata_candidates_snapi.loc[pdata_candidates_idx_snapi_incorrectlyextracted,:]=np.nan
        pdata_candidates_snapf.loc[pdata_candidates_idx_snapf_incorrectlyextracted,:]=np.nan

        pdata_candidates_snapi.loc[:,'inpdata']=1
        pdata_candidates_snapf.loc[:,'inpdata']=1
        pdata_candidates_snapi.loc[pdata_candidates_idx_snapi_incorrectlyextracted,'inpdata']=0
        pdata_candidates_snapf.loc[pdata_candidates_idx_snapf_incorrectlyextracted,'inpdata']=0
        
        pdata_candidates_snapi['ParticleIDs']=pid_allcandidates
        pdata_candidates_snapf['ParticleIDs']=pid_allcandidates

        pdata_candidates_snapi['R_rel']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapi),axis=1))
        pdata_candidates_snapf['R_rel']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Coordinates_{x}' for x in 'xyz']]-galaxy_com_snapf),axis=1))

        return True,pdata_candidates_snapi,pdata_candidates_snapf

    else:
        return False,None,None
