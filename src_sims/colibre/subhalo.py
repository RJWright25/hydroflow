import os
import numpy as np
import pandas as pd
import h5py

from hydroflow.run.tools_catalogue import dump_hdf
from hydroflow.run.initialise import load_metadata

from swiftsimio import load as swiftsimio_loader

def extract_subhaloes(path,mcut=1e10,metadata=None,flowrates=False):
    """
    extract_subhaloes: Read the subhalo catalog from a COLIBRE SOAP output file using swiftsimio. This massages the data into the preferred format for the subhalo catalog, and saves it to a HDF5 file.

    Input:
    -----------

    path: str or list of str
        Path(s) to the simulation SOAP file(s).

    Output:
    ----------- 
    subcat: pd.DataFrame
        DataFrame containing the subhalo catalog.

    """
    

    # Check if just one path is given
    if type(path)==str:
        path=[path]
    
    # Grab metadata from the metadata file
    if metadata is not None:
        metadata_path=metadata
        metadata=load_metadata(metadata)
    else:
        simflist=os.listdir(os.getcwd())
        for metadata_path in simflist:
            if '.pkl' in metadata_path:
                metadata_path=metadata_path
                metadata=load_metadata(metadata_path)
                print(f"Metadata file found: {metadata_path}")
                break
    
    # Units for masses
    munit='Msun'
    dunit='Mpc'
    vunit='km/s'

    # Output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'
    if not os.path.exists(os.getcwd()+'/catalogues/'):
        os.makedirs(os.getcwd()+'/catalogues/')

    # "subcat" will be a list of pandas dataframes which will be concatenated at the end
    subcat=[]

    # Ensure that some catalogues exist
    if len(path)==0:
        print("No catalogue paths given. Exiting...")
        return None
    
    # Loop over all paths for each snapshot
    for ipath in path:
        if os.path.exists(ipath):
            print(f"Reading subhalo catalogue from {ipath}...")
            halodata = swiftsimio_loader(ipath)# Load a dataset

            # Test if soap or subfind
            try:
                halodata.input_halos_subfind.sub_group_number
                subfind=True
            except:
                subfind=False

            # Create a pandas dataframe to store the subhalo data
            halodata_out=pd.DataFrame()
            # Collect redshift & snapshot number
            redshift=halodata.metadata.redshift
            snapnum=int(halodata.metadata.filename.split('/')[-1].split('_')[-1].split('.')[0])
            numhaloes=halodata.input_halos.halo_catalogue_index.shape[0]
            halodata_out['Redshift']=np.ones(numhaloes)*redshift
            halodata_out['SnapNum']=np.ones(numhaloes)*snapnum
            
            # IDs
            central=halodata.input_halos.is_central.value
            if not subfind:
                halodata_out['HostHaloID']=halodata.soap.host_halo_index.value
            halodata_out['GroupNumber']=np.arange(numhaloes)
            halodata_out['SubGroupNumber']=np.zeros(numhaloes)
            halodata_out.loc[np.logical_not(central),'SubGroupNumber']=1
            halodata_out['HaloCatalogueIndex']=halodata.input_halos.halo_catalogue_index.value #This can be used to map to particle data

            #Use TrackID from HBT+ as unique galaxy ID
            if subfind:
                halodata_out['GroupNumber']=halodata.input_halos_subfind.group_number
                halodata_out['SubGroupNumber']=halodata.input_halos_subfind.sub_group_number
                
            else:
                halodata_out['GalaxyID']=halodata.input_halos_hbtplus.track_id.value
                halodata_out['GalaxyID_unique']=snapnum*1e12+halodata_out['GalaxyID'].values # Unique galaxy ID
                halodata_out['DescendantID']=halodata.input_halos_hbtplus.descendant_track_id # Descendant galaxy ID
                halodata_out['ParentID']=halodata.input_halos_hbtplus.nested_parent_track_id # Parent galaxy ID
                halodata_out['SubhaloRank']=halodata.soap.subhalo_rank_by_bound_mass.value # Rank of the subhalo within its host halo by bound mass
            
            # print("Central fraction: ",np.sum(central)/len(halodata_out['HostHaloID']))
            # print("Central fraction by rank: ",np.nanmean(halodata_out['SubhaloRank'].values==0))

            # Host halo properties
            mfof=halodata.input_halos_fof.masses;mfof.convert_to_units(munit)
            if not subfind:
                halodata_out['GroupMass']=np.array(mfof.value)

            for overdensity in zip(['Crit200','Crit500'],[halodata.spherical_overdensity_200_crit,halodata.spherical_overdensity_500_crit]):
                od_str=overdensity[0];od_data=overdensity[1]
                mod=od_data.total_mass;mod.convert_to_units(munit)
                halodata_out[f'Group_M_{od_str}']=np.array(mod.value)
                rod=od_data.soradius;rod.convert_to_units(dunit)
                halodata_out[f'Group_R_{od_str}']=np.array(rod.value) #comoving


            vmax=halodata.bound_subhalo.maximum_circular_velocity;vmax.convert_to_units(vunit)
            halodata_out['Subhalo_V_max']=np.array(vmax.value)

            # Centre of mass -- use the central galaxy 30kpc inclusive sphere
            cop_halo=halodata.exclusive_sphere_30kpc.centre_of_mass
            cop_halo.convert_to_units('Mpc')
            halodata_out['CentreOfPotential_x']=np.array(cop_halo[:,0].value)
            halodata_out['CentreOfPotential_y']=np.array(cop_halo[:,1].value)
            halodata_out['CentreOfPotential_z']=np.array(cop_halo[:,2].value)
            
            # Subhalo mass
            subhalomass=halodata.bound_subhalo.total_mass;subhalomass.convert_to_units(munit)
            halodata_out['Mass']=np.array(subhalomass.value)

            # Miscellaneous baryonic properties
            mstar_30kpc=halodata.exclusive_sphere_30kpc.stellar_mass;mstar_30kpc.convert_to_units(munit)
            halodata_out['030pkpc_sphere-star-m_tot-soapexcl']=np.array(mstar_30kpc.value)
            mgas_30kpc=halodata.exclusive_sphere_30kpc.gas_mass;mgas_30kpc.convert_to_units(munit)
            halodata_out['030pkpc_sphere-gas_all-m_tot-soapexcl']=np.array(mgas_30kpc.value)
            mHI_30kpc=halodata.exclusive_sphere_30kpc.atomic_hydrogen_mass;mHI_30kpc.convert_to_units(munit)
            halodata_out['030pkpc_sphere-gas_all-m_HI-soapexcl']=np.array(mHI_30kpc.value)
            mH2_30kpc=halodata.exclusive_sphere_30kpc.molecular_hydrogen_mass;mH2_30kpc.convert_to_units(munit)
            halodata_out['030pkpc_sphere-gas_all-m_H2-soapexcl']=np.array(mH2_30kpc.value)
            sfr_30kpc=halodata.exclusive_sphere_30kpc.star_formation_rate;sfr_30kpc.convert_to_units(f'{munit}/yr')
            halodata_out['030pkpc_sphere-gas_all-SFR-soapexcl']=np.array(sfr_30kpc.value)
            aveSFR_30kpc=halodata.exclusive_sphere_30kpc.averaged_star_formation_rate;aveSFR_30kpc.convert_to_units(f'{munit}/yr')
            halodata_out['030pkpc_sphere-gas_all-ave_SFR_10Myr-soapexcl']=np.array(aveSFR_30kpc.value[:,0]) #averaged over 10 Myr
            halodata_out['030pkpc_sphere-gas_all-ave_SFR_100Myr-soapexcl']=np.array(aveSFR_30kpc.value[:,1]) #averaged over 100 Myr
            rstar=halodata.exclusive_sphere_30kpc.half_mass_radius_stars;rstar.convert_to_units(dunit)
            halodata_out['030pkpc_sphere-star-r_half-soapexcl']=np.array(rstar.value)
            rgas=halodata.exclusive_sphere_30kpc.half_mass_radius_gas;rgas.convert_to_units(dunit)
            halodata_out['030pkpc_sphere-gas_all-r_half-soapexcl']=np.array(rgas.value)
            disk_to_total_star=halodata.exclusive_sphere_30kpc.disc_to_total_stellar_mass_fraction
            halodata_out['030pkpc_sphere-star-disk_to_total-soapexcl']=np.array(disk_to_total_star)
            disk_to_total_gas=halodata.exclusive_sphere_30kpc.disc_to_total_gas_mass_fraction
            halodata_out['030pkpc_sphere-gas_all-disk_to_total-soapexcl']=np.array(disk_to_total_gas)
            kappaco_star=halodata.exclusive_sphere_30kpc.kappa_corot_stars
            halodata_out['030pkpc_sphere-star-kappa_corot-soapexcl']=np.array(kappaco_star)
            kappaco_gas=halodata.exclusive_sphere_30kpc.kappa_corot_gas
            halodata_out['030pkpc_sphere-gas_all-kappa_corot-soapexcl']=np.array(kappaco_gas)
            stellarluminosities=halodata.exclusive_sphere_30kpc.stellar_luminosity
            stellarluminosities.convert_to_units('1')
            for iband,band in enumerate(['u','g','r','i','z','Y','J','H','K']):
                lum_band=stellarluminosities[:,iband]
                halodata_out[f'030pkpc_sphere-star-L_{band}-soapexcl']=np.array(lum_band.value)

            angmom=halodata.inclusive_sphere_30kpc.angular_momentum_baryons;angmom.convert_to_units('Msun*Mpc*km/s');angmom.convert_to_physical()
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_x']=np.array(angmom.value[:,0])
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_y']=np.array(angmom.value[:,1])
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_z']=np.array(angmom.value[:,2])

            # Black hole properties
            nbh=halodata.exclusive_sphere_30kpc.number_of_black_hole_particles
            halodata_out['030pkpc_sphere-BH-n_tot-soapexcl']=np.array(nbh)
            mbh_total=halodata.exclusive_sphere_30kpc.most_massive_black_hole_mass;mbh_total.convert_to_units(munit)
            halodata_out['030pkpc_sphere-BH-m_tot-soapexcl']=np.array(mbh_total.value)
            bh_aveaccretion=halodata.exclusive_sphere_30kpc.most_massive_black_hole_averaged_accretion_rate;bh_aveaccretion.convert_to_units(f'{munit}/yr')
            halodata_out['030pkpc_sphere-BH-ave_accretion_10Myr-soapexcl']=np.array(bh_aveaccretion.value[:,0]) #averaged over 10 Myr
            halodata_out['030pkpc_sphere-BH-ave_accretion_100Myr-soapexcl']=np.array(bh_aveaccretion.value[:,1]) #averaged over 100 Myr
            bh_thermal_energy=halodata.exclusive_sphere_30kpc.most_massive_black_hole_injected_thermal_energy;bh_thermal_energy.convert_to_units('erg')
            halodata_out['030pkpc_sphere-BH-thermal_energy_soapexcl']=np.array(bh_thermal_energy.value)
            bh_accreted_mass=halodata.exclusive_sphere_30kpc.most_massive_black_hole_total_accreted_mass;bh_accreted_mass.convert_to_units(munit)
            halodata_out['030pkpc_sphere-BH-accreted_mass_soapexcl']=np.array(bh_accreted_mass.value)

            #hybrid AGN props
            if hasattr(halodata.exclusive_sphere_30kpc,'most_massive_black_hole_injected_jet_energy_by_mode'):
                bh_jet_energy_modes=halodata.exclusive_sphere_30kpc.most_massive_black_hole_injected_jet_energy_by_mode;bh_jet_energy_modes.convert_to_units('erg')
                bh_jet_energy_modes=bh_jet_energy_modes.value
                for imode,mode in enumerate(['thin','thick','slim']):
                    halodata_out[f'030pkpc_sphere-BH-jet_energy_{mode}_soapexcl']=bh_jet_energy_modes.value[:,imode]
            if hasattr(halodata.exclusive_sphere_30kpc,'most_massive_black_hole_accretion_mode'):
                bh_accretion_mode=halodata.exclusive_sphere_30kpc.most_massive_black_hole_accretion_mode
                halodata_out['030pkpc_sphere-BH-accdisc_mode_soapexcl']=bh_accretion_mode.value # 0=thin, 1=thick, 2=slim
            if hasattr(halodata.exclusive_sphere_30kpc,'most_massive_black_hole_number_of_mergers'):
                mbh_nmergers=halodata.exclusive_sphere_30kpc.most_massive_black_hole_number_of_mergers
                halodata_out['030pkpc_sphere-BH-n_mergers-soapexcl']=np.array(mbh_nmergers)

            # Give each satellite the group mass, r200 and m200 of the central and distance to central
            print('Matching group data to satellite data...')
            satellites=halodata_out['SubGroupNumber'].values>0
            hosthaloidxs=np.searchsorted(halodata_out['GroupNumber'].values,halodata_out['HostHaloID'].values[satellites])
            halodata_out.loc[satellites,'GroupMass']=halodata_out['GroupMass'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_M_Crit200']=halodata_out['Group_M_Crit200'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_R_Crit200']=halodata_out['Group_R_Crit200'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_Rrel']=np.sqrt((halodata_out['CentreOfPotential_x'].values[satellites]-halodata_out['CentreOfPotential_x'].values[hosthaloidxs])**2+(halodata_out['CentreOfPotential_y'].values[satellites]-halodata_out['CentreOfPotential_y'].values[hosthaloidxs])**2+(halodata_out['CentreOfPotential_z'].values[satellites]-halodata_out['CentreOfPotential_z'].values[hosthaloidxs])**2)

            if flowrates:
                try:
                    print('Extracting flow rates...')
                    scales=['0p10r200','0p30r200','1p00r200']
                    scale_idx={'0p10r200':0,'0p30r200':1,'1p00r200':2}

                    for iscale,scale in enumerate(scales):
                        for key,flowrate in zip(['cold','cool','warm','hot'],[halodata.spherical_overdensity_200_crit.cold_gas_mass_flow_rate,halodata.spherical_overdensity_200_crit.cool_gas_mass_flow_rate,halodata.spherical_overdensity_200_crit.warm_gas_mass_flow_rate,halodata.spherical_overdensity_200_crit.hot_gas_mass_flow_rate]):
                            flowrate.convert_to_units(f'{munit}/Gyr')
                            flowrate=flowrate.value
                            for iflow,flowtype in enumerate(['mdot_tot_inflow_vbdef_vc000kmps','mdot_tot_outflow_vbdef_vc000kmps','mdot_tot_outflow_vbdef_vc0p25vmx']):
                                halodata_out[f'{scale}_shellp10_full-gas_{key}-{flowtype}-soap']=flowrate[:,iflow*3+scale_idx[scale]]
                        flowrate=halodata.spherical_overdensity_200_crit.dark_matter_mass_flow_rate
                        for iflow,flowtype in enumerate(['mdot_tot_inflow_vbdef_vc000kmps','mdot_tot_outflow_vbdef_vc000kmps']):
                            flowrate.convert_to_units(f'{munit}/Gyr')
                            halodata_out[f'{scale}_shellp10_full-dm-{flowtype}-soap']=flowrate[:,iflow*2+scale_idx[scale]]
                        
                except:
                    raise
                    print("Flow rate extraction failed. Continuing without flow rates...")


            # Remove subhalos below mass cut
            halodata_out=halodata_out[halodata_out['Group_M_Crit200']>=mcut]
            halodata_out.reset_index(drop=True,inplace=True)

            # Add to previous snapshots
            subcat.append(halodata_out)

        else:
            print(f"Path {ipath} does not exist. Skipping...")

    
    # Concatenate all snapshots
    subcat=pd.concat(subcat)
    subcat.sort_values(['SnapNum','Group_M_Crit200','SubGroupNumber'],ascending=[False,False,True],inplace=True)
    subcat.reset_index(drop=True,inplace=True)
    
    dump_hdf(outpath,subcat)

    #add path to metadata in hdf5
    if metadata is not None:
        with h5py.File(outpath, 'r+') as subcatfile:
            header= subcatfile.create_group("Header")
            header.attrs['metadata'] = str(metadata_path)
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    return subcat
                