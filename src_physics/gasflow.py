# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# src_physics/gasflow.py: routines to analyse reservoir between snapshots, find inflow/outflow particles, and characterise them.

import numpy as np
from hydroflow.src_physics.utils import  MpcpGyr_to_kmps

def analyse_gasflow_lagrangian(galaxy,pdata_snapi,pdata_snapf,radius,dt,Tcut=0,vcuts=None,vcuts_extra=None):
    """
    analyse_gasflow_lagrangian: Analyse gas flows between two snapshots, using Lagrangian approach.

    Inputs:
    -----------
    galaxy: dict or pandas.Series
        Dictionary containing galaxy properties.
    pdata_snapi: pandas.DataFrame
        DataFrame containing particle data at snapshot i.
    pdata_snapf: pandas.DataFrame
        DataFrame containing particle data at snapshot f.
    radius: float
        Radius to consider for gas flow analysis.
    dt: float
        Time difference between snapshots.
    Tcut: float, optional
        Temperature cut for gas particles to define a "cool" phase.
    vcuts: list, optional
        List of velocity cuts for outflow particles.
    vcuts_extra: list, optional
        List of extra velocity cuts for outflow particles which are multiples of v200_eff.

    Returns:
    -----------
    gasflow_output: dict
        Dictionary containing gas flow properties. 
        Each key is a string describing the scale of the gas flow, and the value is a float or array of floats containing the properties of the gas flow.

    
    """
    gasflow_output={}

    afac=galaxy['afac']
    
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

    if 'StellarFormationTime' in pdata_snapi:
        gas_wind=np.logical_or(pdata_snapf['StellarFormationTime'].values<0,pdata_snapi['StellarFormationTime'].values<0)
    elif 'MaximumTemperature' in pdata_snapi:
        gas_wind=pdata_snapf['MaximumTemperature'].values>=1e7
    else:
        gas_wind=np.zeros(pdata_snapf.shape[0])+np.nan

    #radial velocity
    vrad=pdata_snapi['Relative_v_rad'].values

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
    inflow_pristine_mask=np.logical_and(inflow_mask,Z_snap1<1e-4)

    # outflow
    #vcuts
    if vcuts:
        vcuts=list(np.array(vcuts))
    else:
        vcuts=[0]
    
    vcut_keys=[f'{str(int(vcut)).zfill(3)}pkmps' for vcut in vcuts]
    vcuts=list(np.array(vcuts)/np.sqrt(afac))
    
    if vcuts_extra:
        for vcut_extra in vcuts_extra:
            fac=np.float32(vcut_extra[:4].replace('p','.'))
            if 'vc' in vcut_extra:
                val=fac*galaxy['v200_eff']
            else:
                val=np.nan
            vcut_keys.append(vcut_extra)
            vcuts.append(val)

    outflow_masks={vcut_key:np.logical_and.reduce([outflow_mask,vrad>=vcut_val]) for vcut_key,vcut_val in zip(vcut_keys,vcuts)}

    #### inflow
    for name,mask in zip(['inflow','inflow_pristine'],[inflow_mask,inflow_pristine_mask]):
        inflow_mass=mass_snap2[mask]
        gasflow_output[f'{name}-n']=np.nansum(mask)
        gasflow_output[f'{name}-m']=np.nansum(inflow_mass)/dt
        gasflow_output[f'{name}-f_app']=np.nanmean(nopdata_snap1[mask])
        if gasflow_output[f'{name}-n']>0.:
            gasflow_output[f'{name}-Z_mean']=np.average(Z_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-Z_median']=np.nanmedian(Z_snap2[mask])
            gasflow_output[f'{name}-T_mean']=np.average(T_snap2[mask],weights=inflow_mass)
            gasflow_output[f'{name}-T_median']=np.nanmedian(T_snap2[mask])
            gasflow_output[f'{name}-f_wind']=np.average(gas_wind[mask],weights=inflow_mass)
            
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

        if vcut_key=='000pkmps':
            output_outflow_str='outflow'
        else:
            output_outflow_str=f'outflow_{vcut_key}'

        gasflow_output[f'{output_outflow_str}-n']=np.nansum(ejected_mask)
        gasflow_output[f'{output_outflow_str}-m']=np.nansum(outflow_mass)/dt
        gasflow_output[f'{output_outflow_str}-f_lost']=np.nanmean(nopdata_snap2[ejected_mask])
        if gasflow_output[f'{output_outflow_str}-n']>0.:
            gasflow_output[f'{output_outflow_str}-Z_mean']=np.average(Z_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{output_outflow_str}-Z_median']=np.nanmedian(Z_snap1[ejected_mask])
            gasflow_output[f'{output_outflow_str}-T_mean']=np.average(T_snap1[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{output_outflow_str}-T_median']=np.nanmedian(T_snap1[ejected_mask])
            gasflow_output[f'{output_outflow_str}-f_wind']=np.average(gas_wind[ejected_mask],weights=outflow_mass)

            #ejection vel
            vrad_outflow=vrad[ejected_mask]
            vel_mask=np.where(np.logical_and(np.isfinite(vrad_outflow),outflow_mass>=0))
            if np.nansum(vel_mask) and vcut_key=='000pkmps':
                gasflow_output[f'{output_outflow_str}-vrad_mean']=np.average(vrad_outflow[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{output_outflow_str}-vrad_median']=np.nanmedian(vrad_outflow[vel_mask])
                gasflow_output[f'{output_outflow_str}-vrad_05P']=np.nanpercentile(vrad_outflow[vel_mask],5)
                gasflow_output[f'{output_outflow_str}-vrad_95P']=np.nanpercentile(vrad_outflow[vel_mask],95)

        else:
            remove=[f'{output_outflow_str}-Z_mean',f'{output_outflow_str}-Z_median',f'{output_outflow_str}-T_mean',f'{output_outflow_str}-T_median']
            for field in remove:
                gasflow_output[field]=np.nan

    return gasflow_output

def analyse_gasflow_eulerian(galaxy,pdata,radius,Tcut=0,drfac=0.25,vcuts=None,vcuts_extra=None):
    """
    analyse_gasflow_eulerian: Analyse gas flows in a spherical region around a galaxy, using Eulerian approach.
    
    Inputs:
    -----------
    galaxy: dict or pandas.Series
        Dictionary containing galaxy properties.
    pdata: pandas.DataFrame
        DataFrame containing particle data.
    radius: float
        Radius to consider for gas flow analysis.
    Tcut: float, optional
        Temperature cut for gas particles to define a "cool" phase.
    drfac: float, optional
        Factor to multiply radius by to define the width of the shell region.
    vcuts: list, optional
        List of velocity cuts for outflow particles.
    vcuts_extra: list, optional
        List of extra velocity cuts for outflow particles which are multiples of v200_eff.
    
    Returns:
    -----------
    gasflow_output: dict
        Dictionary containing gas flow properties.
        Each key is a string describing the scale of the gas flow, and the value is a float or array of floats containing the properties of the gas flow.

    """
    gasflow_output={}

    afac=galaxy['afac']
    hval=galaxy['hval']

    dr=radius*drfac

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
    gasflow_output['boundary-vrot']=np.nanmedian(pdata.loc[boundary,'Relative_v_tan'].values)
    
    pdata=pdata.loc[np.logical_and(boundary,gas),:].copy()

    mass=pdata['Mass'].values
    temp=pdata['Temperature'].values
    Zmet=pdata['Metallicity'].values
    vrad=pdata['Relative_v_rad'].values
    vabs=pdata['Relative_v_abs'].values
    vtan=pdata['Relative_v_tan'].values

    if 'StellarFormationTime' in pdata:
        wind=pdata['StellarFormationTime'].values<0
    elif 'MaximumTemperature' in pdata:
        wind=pdata['MaximumTemperature'].values>=1e7
    else:
        wind=np.zeros(pdata.shape[0])+np.nan

    inflow_mask=vrad<0
    outflow_mask=vrad>0

    ## pristine
    inflow_pristine_mask=np.logical_and(inflow_mask,Zmet<1e-4)

    # outflow
    #vcuts
    if vcuts:
        vcuts=list(np.array(vcuts))
    else:
        vcuts=[0]
    
    vcut_keys=[f'{str(int(vcut)).zfill(3)}pkmps' for vcut in vcuts]
    vcuts=list(np.array(vcuts)/np.sqrt(afac))
    if vcuts_extra:
        for vcut_extra in vcuts_extra:
            fac=np.float32(vcut_extra[:4].replace('p','.'))
            if 'vc' in vcut_extra:
                val=fac*galaxy['v200_eff']
            elif 'vr' in vcut_extra:
                val=fac*gasflow_output['boundary-vrot']
            else:
                val=np.nan

            vcut_keys.append(vcut_extra)
            vcuts.append(val)

    outflow_masks={vcut_key:np.logical_and.reduce([outflow_mask,vrad>=vcut_val]) for vcut_key,vcut_val in zip(vcut_keys,vcuts)}

    #### inflow
    for name,mask in zip([f'inflowflux',f'inflowflux_pristine'],[inflow_mask,inflow_pristine_mask]):
        inflow_mass=mass[mask]
        gasflow_output[f'{name}-n']=np.nansum(mask)
        gasflow_output[f'{name}-m']=-np.nansum(inflow_mass*(vrad[mask]/MpcpGyr_to_kmps))/dr
        gasflow_output[f'{name}-f_cov']=np.nanmean(mask)
        if gasflow_output[f'{name}-n']>0.:
            gasflow_output[f'{name}-Z_mean']=np.average(Zmet[mask],weights=inflow_mass)
            gasflow_output[f'{name}-Z_median']=np.nanmedian(Zmet[mask])
            gasflow_output[f'{name}-T_mean']=np.average(temp[mask],weights=inflow_mass)
            gasflow_output[f'{name}-T_median']=np.nanmedian(temp[mask])
            gasflow_output[f'{name}-f_wind']=np.average(wind[mask],weights=inflow_mass)

            vrad_infall=vrad[mask]
            vabs_infall=vabs[mask]
            vtan_infall=vtan[mask]
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


        else:
            remove=[f'{name}-Z_mean',f'{name}-Z_median',f'{name}-T_mean',f'{name}-T_median']
            for field in remove:
                gasflow_output[field]=np.nan

    #### outflows
    for vcut_key in vcut_keys:
        ejected_mask=outflow_masks[vcut_key]
        outflow_mass=mass[ejected_mask]

        if vcut_key=='000pkmps':
            output_outflow_str='outflowflux'
        else:
            output_outflow_str=f'outflowflux_{vcut_key}'

        gasflow_output[f'{output_outflow_str}-n']=np.nansum(ejected_mask)
        gasflow_output[f'{output_outflow_str}-m']=np.nansum(outflow_mass*(vrad[ejected_mask]/MpcpGyr_to_kmps))/(dr)
        gasflow_output[f'{output_outflow_str}-f_cov']=np.nanmean(ejected_mask)
        
        if gasflow_output[f'{output_outflow_str}-n']>0.:
            gasflow_output[f'{output_outflow_str}-Z_mean']=np.average(Zmet[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{output_outflow_str}-Z_median']=np.nanmedian(Zmet[ejected_mask])
            gasflow_output[f'{output_outflow_str}-T_mean']=np.average(temp[ejected_mask],weights=outflow_mass)
            gasflow_output[f'{output_outflow_str}-T_median']=np.nanmedian(temp[ejected_mask])
            gasflow_output[f'{output_outflow_str}-f_wind']=np.average(wind[ejected_mask],weights=outflow_mass)

            #ejection vel
            vel_mask=np.where(outflow_mass>=0)
            if np.nansum(vel_mask) and vcut_key=='000pkmps':
                vrad_ejected=vrad[ejected_mask]
                vabs_ejected=vabs[ejected_mask]
                vtan_ejected=vtan[ejected_mask]

                gasflow_output[f'{output_outflow_str}-vrad_mean']=np.average(vrad_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{output_outflow_str}-vrad_median']=np.nanmedian(vrad_ejected[vel_mask])
                gasflow_output[f'{output_outflow_str}-vrad_05P']=np.nanpercentile(vrad_ejected[vel_mask],5)
                gasflow_output[f'{output_outflow_str}-vrad_95P']=np.nanpercentile(vrad_ejected[vel_mask],95)

                gasflow_output[f'{output_outflow_str}-vabs_mean']=np.average(vabs_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{output_outflow_str}-vabs_median']=np.nanmedian(vabs_ejected[vel_mask])
                gasflow_output[f'{output_outflow_str}-vabs_05P']=np.nanpercentile(vabs_ejected[vel_mask],5)
                gasflow_output[f'{output_outflow_str}-vabs_95P']=np.nanpercentile(vabs_ejected[vel_mask],95)

                gasflow_output[f'{output_outflow_str}-vtan_mean']=np.average(vtan_ejected[vel_mask],weights=outflow_mass[vel_mask])
                gasflow_output[f'{output_outflow_str}-vtan_median']=np.nanmedian(vtan_ejected[vel_mask])
                gasflow_output[f'{output_outflow_str}-vtan_05P']=np.nanpercentile(vtan_ejected[vel_mask],5)
                gasflow_output[f'{output_outflow_str}-vtan_95P']=np.nanpercentile(vtan_ejected[vel_mask],95)

        else:
            remove=[f'{output_outflow_str}-Z_mean',f'{output_outflow_str}-Z_median',f'{output_outflow_str}-T_mean',f'{output_outflow_str}-T_median',f'{output_outflow_str}-f_wind']
            for field in remove:
                gasflow_output[field]=np.nan

    return gasflow_output
    
def candidates_gasflow_lagrangian(galaxy_snapi,galaxy_snapf,pdata_snapi,kdtree_snapi,pdata_snapf,kdtree_snapf,maxrad=None,dt=None):
    """
    candidates_gasflow: Find gas flow candidate particles/elements between two snapshots -- finds any particles that are within a given radius of the galaxy at either snapshot, and returns them (and their properties) sorted by their ID.

    Inputs:
    -----------
    galaxy_snapi: dict or pandas.Series
        Dictionary containing galaxy properties at snapshot i.
    galaxy_snapf: dict or pandas.Series
        Dictionary containing galaxy properties at snapshot f.
    pdata_snapi: pandas.DataFrame
        DataFrame containing particle data at snapshot i.
    kdtree_snapi: scipy.spatial.cKDTree
        KDTree containing particle data at snapshot i.
    pdata_snapf: pandas.DataFrame
        DataFrame containing particle data at snapshot f.
    kdtree_snapf: scipy.spatial.cKDTree
        KDTree containing particle data at snapshot f.
    maxrad: float, optional
        Maximum radius to consider for gas flow analysis.
    dt: float, optional
        Time difference between snapshots.

    Returns:
    -----------
    success: bool
        Whether the function successfully found gas flow candidates.
    pdata_candidates_snapi: pandas.DataFrame
        DataFrame containing gas flow candidate particles at snapshot i sorted by their ID. Matches the IDs of pdata_candidates_snapf.
    pdata_candidates_snapf: pandas.DataFrame
        DataFrame containing gas flow candidate particles at snapshot f sorted by their ID. Matches the IDs of pdata_candidates_snapi.
    
    """
    
    afac_snap1=1/(1+galaxy_snapi['Redshift'])
    afac_snap2=1/(1+galaxy_snapf['Redshift'])
    hval=galaxy_snapf['hval']

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

def candidates_gasflow_euleronly(galaxy_snapf,pdata_snapf,kdtree_snapf,maxrad=None):
    """
    candidates_gasflow_euleronly: Find gas flow candidate particles/elements at a single snapshot -- finds any particles that are within a given radius of the galaxy at the snapshot, and returns them (and their properties) sorted by their ID.
    
    Inputs:
    -----------
    galaxy_snapf: dict or pandas.Series
        Dictionary containing galaxy properties at snapshot f.
    pdata_snapf: pandas.DataFrame
        DataFrame containing particle data at snapshot f.
    kdtree_snapf: scipy.spatial.cKDTree
        KDTree containing particle data at snapshot f.
    maxrad: float, optional
        Maximum radius to consider for gas flow analysis.
    
    Returns:
    -----------
    success: bool
        Whether the function successfully found gas flow candidates.
    pdata_candidates_snapi: None
        None (not applicable for single snapshot analysis).
    pdata_candidates_snapf: pandas.DataFrame
        DataFrame containing gas flow candidate particles at snapshot f sorted by their ID.
    
    """
    afac_snap2=1/(1+galaxy_snapf['Redshift'])
    hval=galaxy_snapf['hval']

    r200_eff=galaxy_snapf['Group_R_Crit200']
    galaxy_com_snapf=np.array([galaxy_snapf[f'CentreOfPotential_{x}'] for x in 'xyz'])
    
    #get gasflow candidates
    if maxrad:
        rcut=maxrad#choose particles within rcut
    else:
        rcut=1*r200_eff

    pidx_candidates_snapf=kdtree_snapf.query_ball_point(galaxy_com_snapf,rcut)
    pdata_candidates_snapf=pdata_snapf.loc[pidx_candidates_snapf,:]
    pdata_candidates_snapf.reset_index(drop=True,inplace=True)

    numcdt_snapf=pdata_candidates_snapf.shape[0]

    if numcdt_snapf>0:

        pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']]=(pdata_candidates_snapf.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-galaxy_com_snapf)
        pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]=pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']].values/hval
        pdata_candidates_snapf.loc[:,[f'Relative_{x}_physical' for x in 'xyz']]=pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']]*afac_snap2

        pdata_candidates_snapf['Relative_r']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}' for x in 'xyz']].values),axis=1)) #h-1cMpc
        pdata_candidates_snapf['Relative_r_comoving']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values),axis=1)) #cMpc
        pdata_candidates_snapf['Relative_r_physical']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_{x}_physical' for x in 'xyz']].values),axis=1)) #pMpc

        vhalo_mean_snapf=[np.nanmean(pdata_candidates_snapf[f'Velocity_{x}']) for x in 'xyz']

        for idim,dim in enumerate('xyz'):
            pdata_candidates_snapf[f'Relative_v_{dim}']=pdata_candidates_snapf[f'Velocity_{dim}'].values-vhalo_mean_snapf[idim]

        pdata_candidates_snapf[f'Relative_v_rad']=np.nansum(pdata_candidates_snapf.loc[:,[f'Relative_{x}_comoving' for x in 'xyz']].values*pdata_candidates_snapf.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values,axis=1)/pdata_candidates_snapf['Relative_r_comoving'].values
        pdata_candidates_snapf[f'Relative_v_abs']=np.sqrt(np.nansum(np.square(pdata_candidates_snapf.loc[:,[f'Relative_v_{dim}' for dim in 'xyz']].values),axis=1))
        pdata_candidates_snapf[f'Relative_v_tan']=np.sqrt(pdata_candidates_snapf[f'Relative_v_abs'].values**2-pdata_candidates_snapf[f'Relative_v_rad'].values**2)

        return True,None,pdata_candidates_snapf

    else:
        return False,None,None
