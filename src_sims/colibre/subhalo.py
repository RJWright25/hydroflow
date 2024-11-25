import os
import numpy as np
import pandas as pd
import h5py

from hydroflow.run.tools_catalog import dump_hdf
from hydroflow.run.initialise import load_metadata

from swiftsimio import load as swiftsimio_loader

def extract_subhaloes(path,mcut=1e11,metadata=None):
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

    # Output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'

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

            # Create a pandas dataframe to store the subhalo data
            halodata_out=pd.DataFrame()
            # Collect redshift & snapshot number
            redshift=halodata.metadata.redshift
            snapnum=int(halodata.metadata.filename.split('/')[-1].split('_')[-1].split('.')[0])
            halodata_out['Redshift']=np.ones(halodata.soap.host_halo_index.shape)*redshift
            halodata_out['SnapNum']=np.ones(halodata.soap.host_halo_index.shape)*snapnum

            # IDs
            halodata_out['HostHaloID']=halodata.soap.host_halo_index.value
            halodata_out['GroupNumber']=np.arange(len(halodata_out['HostHaloID']))
            halodata_out['SubGroupNumber']=np.zeros(len(halodata_out['HostHaloID']))
            halodata_out.loc[halodata_out['HostHaloID'].values>=0,'SubGroupNumber']=1
            halodata_out['GalaxyID_raw']=halodata.input_halos_hbtplus.track_id.value
            halodata_out['GalaxyID']=snapnum*1e12+halodata_out['GalaxyID_raw'].values # Unique galaxy ID

            # Host halo properties
            mfof=halodata.input_halos_fof.masses;mfof.convert_to_units(munit)
            halodata_out['GroupMass']=np.array(mfof.value)
            
            m200=halodata.spherical_overdensity_200_crit.total_mass;m200.convert_to_units(munit)
            halodata_out['Group_M_Crit200']=np.array(m200.value)

            r200=halodata.spherical_overdensity_200_crit.soradius;r200.convert_to_units(dunit)
            halodata_out['Group_R_Crit200']=np.array(r200.value) #comoving

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
            rstar=halodata.exclusive_sphere_30kpc.half_mass_radius_stars;rstar.convert_to_units(dunit)
            halodata_out['030pkpc_sphere-star-r_half-soapexcl']=np.array(rstar.value)
            disk_to_total_star=halodata.exclusive_sphere_30kpc.disc_to_total_stellar_mass_fraction
            halodata_out['030pkpc_sphere-star-disk_to_total-soapexcl']=np.array(disk_to_total_star)
            disk_to_total_gas=halodata.exclusive_sphere_30kpc.disc_to_total_gas_mass_fraction
            halodata_out['030pkpc_sphere-gas_all-disk_to_total-soapexcl']=np.array(disk_to_total_gas)
            mbh=halodata.exclusive_sphere_30kpc.most_massive_black_hole_mass;mbh.convert_to_units(munit)
            halodata_out['030pkpc_sphere-BH-m_tot-soapexcl']=np.array(mbh.value)
            angmom=halodata.inclusive_sphere_30kpc.angular_momentum_baryons;angmom.convert_to_units('Msun*Mpc*km/s');angmom.convert_to_physical()
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_x']=np.array(angmom.value[:,0])
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_y']=np.array(angmom.value[:,1])
            halodata_out['030pkpc_sphere-baryon-L_tot-soapincl_z']=np.array(angmom.value[:,2])

            # Give each satellite the group mass, r200 and m200 of the central and distance to central
            print('Matching group data to satellite data...')
            satellites=halodata_out['HostHaloID'].values>=0
            hosthaloidxs=np.searchsorted(halodata_out['GroupNumber'].values,halodata_out['HostHaloID'].values[satellites])
            halodata_out.loc[satellites,'GroupMass']=halodata_out['GroupMass'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_M_Crit200']=halodata_out['Group_M_Crit200'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_R_Crit200']=halodata_out['Group_R_Crit200'].values[hosthaloidxs]
            halodata_out.loc[satellites,'Group_Rrel']=np.sqrt((halodata_out['CentreOfPotential_x'].values[satellites]-halodata_out['CentreOfPotential_x'].values[hosthaloidxs])**2+(halodata_out['CentreOfPotential_y'].values[satellites]-halodata_out['CentreOfPotential_y'].values[hosthaloidxs])**2+(halodata_out['CentreOfPotential_z'].values[satellites]-halodata_out['CentreOfPotential_z'].values[hosthaloidxs])**2)

            # Remove subhalos below mass cut
            halodata_out=halodata_out[halodata_out['Mass']>=mcut]
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
            header.attrs['metadata'] = metadata_path
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    return subcat
                