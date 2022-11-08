# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: routines to analyse reservoir between snapshots, find inflow/outflow particles, and characterise them.

import numpy as np

from hydroflow.src_physics.utils import  MpcpGyr_to_kmps

def analyse_gasflow_lagrangian(pdata_snapi,pdata_snapf,radius,dt,afac=1,Tcut=None,vcuts=[0,50,100,150,250]):
    gasflow_output={}
    
    mass_snap1=pdata_snapi['Mass'].values
    mass_snap2=pdata_snapf['Mass'].values

    Z_snap1=pdata_snapi['Metallicity'].values
    Z_snap2=pdata_snapf['Metallicity'].values

    rcut_snap1=pdata_snapi['Relative_r'].values<=radius
    rcut_snap2=pdata_snapf['Relative_r'].values<=radius

    nopdata_snap1=np.logical_not(pdata_snapi['inpdata'].values)
    nopdata_snap2=np.logical_not(pdata_snapf['inpdata'].values)

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

    #radial velocity
    vrad=np.maximum(pdata_snapi['Relative_v_rad'].values,pdata_snapf['Relative_v_rad'].values)

    if Tcut: 
        cool_snap1=T_snap1<=Tcut
        cool_snap2=T_snap2<=Tcut

    else:
        cool_snap1=np.ones(pdata_snapi.shape[0])>0
        cool_snap2=np.ones(pdata_snapf.shape[0])>0

    selection_snap1=np.logical_and.reduce([rcut_snap1,np.logical_or(cool_snap1,star_snap1)])
    selection_snap2=np.logical_and.reduce([rcut_snap2,np.logical_or(cool_snap2,star_snap2)])

    #do gas calcs here
    inflow_mask=np.logical_and.reduce([selection_snap2,np.logical_or(np.logical_not(selection_snap1),nopdata_snap1),np.logical_or(gas_snap2,gas_snap1)])
    outflow_mask=np.logical_and.reduce([selection_snap1,np.logical_or(np.logical_not(selection_snap2),nopdata_snap2),np.logical_or(gas_snap2,gas_snap1)])

    ## pristine
    inflow_pristine_mask=np.logical_and(inflow_mask,np.logical_or(Z_snap2<1e-4,Z_snap1<1e-4))

    #vcuts
    vcut_keys=[f'{str(int(vcut)).zfill(3)}pkmps' for vcut in vcuts]
    vcuts=np.array(vcuts)/np.sqrt(afac)
    outflow_masks={vcut_key:np.logical_and.reduce([outflow_mask,vrad>=vcut_val]) for vcut_key,vcut_val in zip(vcut_keys,vcuts)}

    #### inflow
    for name,mask in zip(['inflow','inflow_pristine'],[inflow_mask,inflow_pristine_mask]):
        inflow_mass=mass_snap2[mask]
        gasflow_output[f'{name}-n']=np.nansum(mask)
        gasflow_output[f'{name}-m']=np.nansum(inflow_mass)/dt
        gasflow_output[f'{name}-fapp']=np.nanmean(nopdata_snap1[mask])
        if gasflow_output[f'{name}-n']>0.:
            gasflow_output[f'{name}-Z_mean']=np.average(Z_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-Z_median']=np.nanmedian(Z_snap2[mask])
            gasflow_output[f'{name}-T_mean']=np.average(T_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-T_median']=np.nanmedian(T_snap2[mask])

            #infall vel
            vave_inflow=vrad[mask]
            vave_mask=np.where(np.logical_and(np.isfinite(vave_inflow),inflow_mass>=0))
            if np.nansum(vave_mask):
                gasflow_output[f'{name}-vrad_mean']=np.average(vave_inflow[vave_mask],weights=inflow_mass[vave_mask])
                gasflow_output[f'{name}-vrad_median']=np.nanmedian(vave_inflow[vave_mask])

        else:
            remove=[f'{name}-Z_mean',f'{name}-Z_median',f'{name}-T_mean',f'{name}-T_median',f'{name}-vrad_mean',f'{name}-vrad_median']
            for field in remove:
                gasflow_output[field]=np.nan


    #### outflows
    for vcut_key in vcut_keys:
        ejected_mask=outflow_masks[vcut_key]
        outflow_mass=mass_snap1[ejected_mask]
        gasflow_output[f'{vcut_key}_outflow-n']=np.nansum(ejected_mask)
        gasflow_output[f'{vcut_key}_outflow-m']=np.nansum(outflow_mass)/dt
        gasflow_output[f'{vcut_key}_outflow-flost']=np.nanmean(nopdata_snap2[ejected_mask])
        if gasflow_output[f'{vcut_key}_outflow-n']>0.:
            gasflow_output[f'{vcut_key}_outflow-Z_mean']=np.average(Z_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut_key}_outflow-Z_median']=np.nanmedian(Z_snap1[ejected_mask])
            gasflow_output[f'{vcut_key}_outflow-T_mean']=np.average(T_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut_key}_outflow-T_median']=np.nanmedian(T_snap1[ejected_mask])
            
            #ejection vel
            vave_outflow=vrad[ejected_mask]
            vel_mask=np.where(np.logical_and(np.isfinite(vave_outflow),outflow_mass>=0))
            if np.nansum(vel_mask) and vcut_key=='000kmps':
                gasflow_output[f'{vcut_key}_outflow-vrad_mean']=np.average(vave_outflow[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{vcut_key}_outflow-vrad_median']=np.nanmedian(vave_outflow[vel_mask])
                gasflow_output[f'{vcut_key}_outflow-vrad_05P']=np.nanpercentile(vave_outflow[vel_mask],5)
                gasflow_output[f'{vcut_key}_outflow-vrad_95P']=np.nanpercentile(vave_outflow[vel_mask],95)

        else:
            remove=[f'{vcut_key}_outflow-Z_mean',f'{vcut_key}_outflow-Z_median',f'{vcut_key}_outflow-T_mean',f'{vcut_key}_outflow-T_median',f'{vcut_key}_outflow-vrad_mean',f'{vcut_key}_outflow-vrad_median',f'{vcut_key}_outflow-vrad_05P',f'{vcut_key}_outflow-vrad_95P']
            for field in remove:
                gasflow_output[field]=np.nan

    return gasflow_output

def analyse_gasflow_eulerian(pdata,radius,Tcut=0,afac=1,hval=0.67,vcuts=[0,50,100,150,250]):
    gasflow_output={}

    dr=radius*0.3

    boundary_lo=radius-dr/2
    boundary_hi=radius+dr/2

    dr*=hval

    gas=pdata['ParticleType'].values==0
    if 'StellarFormationTime' in pdata:
        gas=np.logical_or(gas,pdata['StellarFormationTime'].values<=0)

    boundary=np.logical_and(pdata['Relative_r'].values>boundary_lo,pdata['Relative_r'].values<boundary_hi)
    if Tcut:
        boundary=np.logical_and(boundary,pdata['Temperature'].values<=Tcut)

    gasflow_output['boundary-n']=np.nansum(boundary)

    pdata=pdata.loc[np.logical_and(boundary,gas),:].copy()

    mass=pdata['Mass'].values
    temp=pdata['Temperature'].values
    Zmet=pdata['Metallicity'].values
    vrad=pdata['Relative_v_rad'].values
    vabs=pdata['Relative_v_abs'].values
    vtan=pdata['Relative_v_tan'].values
    vave=pdata['Average_v_rad'].values

    inflow_mask=vrad<0
    outflow_mask=vrad>0

    ## pristine
    inflow_pristine_mask=np.logical_and(inflow_mask,Zmet<1e-4)

    # outflow
    vcut_keys=[f'{str(int(vcut)).zfill(3)}pkmps' for vcut in vcuts]
    vcuts=np.array(vcuts)/np.sqrt(afac)
    outflow_masks={vcut_key:np.logical_and.reduce([outflow_mask,vrad>=vcut_val]) for vcut_key,vcut_val in zip(vcut_keys,vcuts)}

    #### inflow
    for name,mask in zip([f'inflowflux',f'inflowflux_pristine'],[inflow_mask,inflow_pristine_mask]):
        inflow_mass=mass[mask]
        gasflow_output[f'{name}-n']=np.nansum(mask)
        gasflow_output[f'{name}-m']=-np.nansum(inflow_mass*(vrad[mask]/MpcpGyr_to_kmps))/dr
        gasflow_output[f'{name}-fcov']=np.nanmean(mask)
        if gasflow_output[f'{name}-n']>0.:
            gasflow_output[f'{name}-Z_mean']=np.average(Zmet[mask],weights=inflow_mass)
            gasflow_output[f'{name}-Z_median']=np.nanmedian(Zmet[mask])
            gasflow_output[f'{name}-T_mean']=np.average(temp[mask],weights=inflow_mass)
            gasflow_output[f'{name}-T_median']=np.nanmedian(temp[mask])

            vrad_infall=vrad[mask]
            vabs_infall=vabs[mask]
            vtan_infall=vtan[mask]
            vave_infall=vave[mask]
            vel_mask=np.where(inflow_mass>=0)

            if np.nansum(vel_mask):
                gasflow_output[f'{name}-vrad_mean']=np.average(vrad_infall[vel_mask],weights=inflow_mass[vel_mask])
                gasflow_output[f'{name}-vrad_median']=np.nanmedian(vrad_infall[vel_mask])
                gasflow_output[f'{name}-vrad_05P']=np.nanpercentile(vrad_infall[vel_mask],5)
                gasflow_output[f'{name}-vrad_95P']=np.nanpercentile(vrad_infall[vel_mask],95)

                gasflow_output[f'{name}-vabs_mean']=np.average(vabs_infall[vel_mask],weights=inflow_mass[vel_mask])
                gasflow_output[f'{name}-vabs_median']=np.nanmedian(vabs_infall[vel_mask])
                gasflow_output[f'{name}-vabs_05P']=np.nanpercentile(vabs_infall[vel_mask],5)
                gasflow_output[f'{name}-vabs_95P']=np.nanpercentile(vabs_infall[vel_mask],95)

                gasflow_output[f'{name}-vtan_mean']=np.average(vtan_infall[vel_mask],weights=inflow_mass[vel_mask])
                gasflow_output[f'{name}-vtan_median']=np.nanmedian(vtan_infall[vel_mask])
                gasflow_output[f'{name}-vtan_05P']=np.nanpercentile(vtan_infall[vel_mask],5)
                gasflow_output[f'{name}-vtan_95P']=np.nanpercentile(vtan_infall[vel_mask],95)

                gasflow_output[f'{name}-vave_mean']=np.average(vave_infall[vel_mask],weights=inflow_mass[vel_mask])
                gasflow_output[f'{name}-vave_median']=np.nanmedian(vave_infall[vel_mask])
                gasflow_output[f'{name}-vave_05P']=np.nanpercentile(vave_infall[vel_mask],5)
                gasflow_output[f'{name}-vave_95P']=np.nanpercentile(vave_infall[vel_mask],95)

        else:
            remove=[f'{name}-Z_mean',f'{name}-Z_median',f'{name}-T_mean',f'{name}-T_median']
            for veltype in ['vrad','vabs','vtan','vave']:
                remove.append(f'{name}-{veltype}_mean')
                remove.append(f'{name}-{veltype}_median')
                remove.append(f'{name}-{veltype}_05P')
                remove.append(f'{name}-{veltype}_95P')

            for field in remove:
                gasflow_output[field]=np.nan

    #### outflows
    for vcut_key in vcut_keys:
        ejected_mask=outflow_masks[vcut_key]
        outflow_mass=mass[ejected_mask]
        gasflow_output[f'{vcut_key}_outflowflux-n']=np.nansum(ejected_mask)
        gasflow_output[f'{vcut_key}_outflowflux-m']=np.nansum(outflow_mass*(vrad[ejected_mask]/MpcpGyr_to_kmps))/(dr)
        gasflow_output[f'{vcut_key}_outflowflux-fcov']=np.nanmean(ejected_mask)
        
        if gasflow_output[f'{vcut_key}_outflowflux-n']>0.:
            gasflow_output[f'{vcut_key}_outflowflux-Z_mean']=np.average(Zmet[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut_key}_outflowflux-Z_median']=np.nanmedian(Zmet[ejected_mask])
            gasflow_output[f'{vcut_key}_outflowflux-T_mean']=np.average(temp[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{vcut_key}_outflowflux-T_median']=np.nanmedian(temp[ejected_mask])
            
            #ejection vel
            vrad_ejected=vrad[ejected_mask]
            vabs_ejected=vabs[ejected_mask]
            vtan_ejected=vtan[ejected_mask]
            vave_ejected=vave[ejected_mask]

            vel_mask=np.where(outflow_mass>=0)
            if np.nansum(vel_mask) and vcut_key=='000kmps':
                gasflow_output[f'{vcut_key}_outflowflux-vave_mean']=np.average(vave_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vave_median']=np.nanmedian(vave_ejected[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vave_05P']=np.nanpercentile(vave_ejected[vel_mask],5)
                gasflow_output[f'{vcut_key}_outflowflux-vave_95P']=np.nanpercentile(vave_ejected[vel_mask],95)

                gasflow_output[f'{vcut_key}_outflowflux-vrad_mean']=np.average(vrad_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vrad_median']=np.nanmedian(vrad_ejected[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vrad_05P']=np.nanpercentile(vrad_ejected[vel_mask],5)
                gasflow_output[f'{vcut_key}_outflowflux-vrad_95P']=np.nanpercentile(vrad_ejected[vel_mask],95)

                gasflow_output[f'{vcut_key}_outflowflux-vabs_mean']=np.average(vabs_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vabs_median']=np.nanmedian(vabs_ejected[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vabs_05P']=np.nanpercentile(vabs_ejected[vel_mask],5)
                gasflow_output[f'{vcut_key}_outflowflux-vabs_95P']=np.nanpercentile(vabs_ejected[vel_mask],95)

                gasflow_output[f'{vcut_key}_outflowflux-vtan_mean']=np.average(vtan_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vtan_median']=np.nanmedian(vtan_ejected[vel_mask])
                gasflow_output[f'{vcut_key}_outflowflux-vtan_05P']=np.nanpercentile(vtan_ejected[vel_mask],5)
                gasflow_output[f'{vcut_key}_outflowflux-vtan_95P']=np.nanpercentile(vtan_ejected[vel_mask],95)

        else:
            remove=[f'{vcut_key}_outflowflux-Z_mean',f'{vcut_key}_outflowflux-Z_median',f'{vcut_key}_outflowflux-T_mean',f'{vcut_key}_outflowflux-T_median']
            if vcut_key=='000kmps':
                for veltype in ['vrad','vabs','vtan','vave']:
                    remove.append(f'{vcut_key}_outflowflux-{veltype}_mean')
                    remove.append(f'{vcut_key}_outflowflux-{veltype}_median')
                    remove.append(f'{vcut_key}_outflowflux-{veltype}_05P')
                    remove.append(f'{vcut_key}_outflowflux-{veltype}_95P')
            for field in remove:
                gasflow_output[field]=np.nan

    return gasflow_output
    
def candidates_gasflow(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf,hval=0.67,maxrad=None,dt=None):
    
    afac_snap1=1/(1+galaxy_snapi['Redshift'])
    afac_snap2=1/(1+galaxy_snapf['Redshift'])

    r200_eff=(galaxy_snapf['Group_R_Crit200']+galaxy_snapi['Group_R_Crit200'])/2

    galaxy_com_snapi=np.array([galaxy_snapi[f'CentreOfPotential_{x}'] for x in 'xyz'])
    galaxy_com_snapf=np.array([galaxy_snapf[f'CentreOfPotential_{x}'] for x in 'xyz'])
    
    #get gasflow candidates
    if maxrad:
        rcut=maxrad#choose particles within rcut
    else:
        rcut=1*r200_eff

    pidx_candidates_snapi=kdtree_snapi.query_ball_point(galaxy_com_snapi,rcut)
    pidx_candidates_snapf=kdtree_snapf.query_ball_point(galaxy_com_snapf,rcut)

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

    pdata_candidates_idx_snapi[pdata_candidates_idx_snapi_incorrectlyextracted]=-1
    pdata_candidates_idx_snapf[pdata_candidates_idx_snapf_incorrectlyextracted]=-1

    failed=False
    try:
        pdata_candidates_snapi=pdata_snapi.iloc[pdata_candidates_idx_snapi,:]
        pdata_candidates_snapf=pdata_snapf.iloc[pdata_candidates_idx_snapf,:]
    except:
        failed=True

    numcdt_snapi=pdata_candidates_snapi.shape[0]
    numcdt_snapf=pdata_candidates_snapf.shape[0]

    if not failed and (numcdt_snapi>0 and numcdt_snapf>0):

        pdata_candidates_snapi.loc[pdata_candidates_idx_snapi_incorrectlyextracted,:]=np.nan
        pdata_candidates_snapf.loc[pdata_candidates_idx_snapf_incorrectlyextracted,:]=np.nan

        try:
            pdata_candidates_snapi.loc[:,'inpdata']=1
            pdata_candidates_snapf.loc[:,'inpdata']=1
            pdata_candidates_snapi.loc[pdata_candidates_idx_snapi_incorrectlyextracted,'inpdata']=0
            pdata_candidates_snapf.loc[pdata_candidates_idx_snapf_incorrectlyextracted,'inpdata']=0
        except:
            return False,None,None
                        
        pdata_candidates_snapi['ParticleIDs']=pid_allcandidates
        pdata_candidates_snapf['ParticleIDs']=pid_allcandidates

        pdata_candidates_snapi.loc[:,[f'Relative_{x}' for x in 'xyz']]=(pdata_candidates_snapi.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-galaxy_com_snapi)
        pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']]=(pdata_candidates_snapf.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-galaxy_com_snapf)
        pdata_candidates_snapi.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=pdata_candidates_snapi.loc[:,[f'Relative_{x}' for x in 'xyz']].values/hval
        pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']].values/hval
        pdata_candidates_snapi.loc[:,[f'Relative_{x}_physical' for x in 'xyz']]=pdata_candidates_snapi.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]*afac_snap1
        pdata_candidates_snapf.loc[:,[f'Relative_{x}_physical' for x in 'xyz']]=pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]*afac_snap2

        pdata_candidates_snapi['Relative_r']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Relative_{x}' for x in 'xyz']].values),axis=1)) #h-1cMpc
        pdata_candidates_snapf['Relative_r']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']].values),axis=1)) #h-1cMpc
        pdata_candidates_snapi['Relative_r_comoving']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values),axis=1)) #cMpc
        pdata_candidates_snapf['Relative_r_comoving']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values),axis=1)) #cMpc
        pdata_candidates_snapi['Relative_r_physical']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Relative_{x}_physical' for x in 'xyz']].values),axis=1)) #pMpc
        pdata_candidates_snapf['Relative_r_physical']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}_physical' for x in 'xyz']].values),axis=1)) #pMpc

        vhalo_mean_snapi=[np.nanmean(pdata_candidates_snapi[f'Velocity_{x}']) for x in 'xyz']
        vhalo_mean_snapf=[np.nanmean(pdata_candidates_snapf[f'Velocity_{x}']) for x in 'xyz']

        for idim,dim in enumerate('xyz'):
            pdata_candidates_snapi[f'Relative_v_{dim}']=pdata_candidates_snapi[f'Velocity_{dim}'].values-vhalo_mean_snapi[idim]
            pdata_candidates_snapf[f'Relative_v_{dim}']=pdata_candidates_snapf[f'Velocity_{dim}'].values-vhalo_mean_snapf[idim]

        pdata_candidates_snapi[f'Relative_v_rad']=np.nansum(pdata_candidates_snapi.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values*pdata_candidates_snapi.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values,axis=1)/pdata_candidates_snapi['Relative_r_comoving'].values
        pdata_candidates_snapf[f'Relative_v_rad']=np.nansum(pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values*pdata_candidates_snapf.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values,axis=1)/pdata_candidates_snapf['Relative_r_comoving'].values
        pdata_candidates_snapi[f'Relative_v_abs']=np.sqrt(np.nansum(np.square(pdata_candidates_snapi.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values),axis=1))
        pdata_candidates_snapf[f'Relative_v_abs']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values),axis=1))
        pdata_candidates_snapi[f'Relative_v_tan']=np.sqrt(pdata_candidates_snapi[f'Relative_v_abs'].values**2-pdata_candidates_snapi[f'Relative_v_rad'].values**2)
        pdata_candidates_snapf[f'Relative_v_tan']=np.sqrt(pdata_candidates_snapf[f'Relative_v_abs'].values**2-pdata_candidates_snapf[f'Relative_v_rad'].values**2)

        pdata_candidates_snapi[f'Average_v_rad']=(pdata_candidates_snapf['Relative_r_comoving'].values-pdata_candidates_snapi['Relative_r_comoving'].values)/dt*MpcpGyr_to_kmps
        pdata_candidates_snapf[f'Average_v_rad']=(pdata_candidates_snapf['Relative_r_comoving'].values-pdata_candidates_snapi['Relative_r_comoving'].values)/dt*MpcpGyr_to_kmps

        return True,pdata_candidates_snapi,pdata_candidates_snapf

    else:
        return False,None,None
