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
    
    #Grab metadata from the metadata file
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
    
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'

    # "subcat" will be a list of pandas dataframes which will be concatenated at the end
    subcat=[]

    if len(path)==0:
        print("No catalogue paths given. Exiting...")
        return None
    
    for ipath in path:
        if os.path.exists(ipath):
            print(f"Reading subhalo catalogue from {ipath}...")
            halodata = swiftsimio_loader(ipath)# Load a dataset

            #need: redshift, snapnum, galaxyID, CentreofPotential_x, CentreofPotential_y, CentreofPotential_z, Group_R_Crit200, Group_M_Crit200, GroupNumber, SubGroupNumber
            redshift=halodata.metadata.redshift
            snapnum=int(halodata.metadata.filename.split('/')[-1].split('_')[-1].split('.')[0])
            hosthalo=halodata.soap.host_halo_index.value
            galaxyID=halodata.input_halos_hbtplus.track_id.value
            subhalomass=halodata.bound_subhalo.total_mass;subhalomass.convert_to_units('Msun');subhalomass=np.array(subhalomass.value) #Msun
            mstar_30kpc_exclusive=halodata.exclusive_sphere_30kpc.stellar_mass;mstar_30kpc_exclusive.convert_to_units('Msun');mstar_30kpc_exclusive=np.array(mstar_30kpc_exclusive.value) #Msun
            mgas_30kpc_exclusive=halodata.exclusive_sphere_30kpc.gas_mass;mgas_30kpc_exclusive.convert_to_units('Msun');mgas_30kpc_exclusive=np.array(mgas_30kpc_exclusive.value) #Msun
            mHI_30kpc_exclusive=halodata.exclusive_sphere_30kpc.atomic_hydrogen_mass;mHI_30kpc_exclusive.convert_to_units('Msun');mHI_30kpc_exclusive=np.array(mHI_30kpc_exclusive.value) #Msun
            mH2_30kpc_exclusive=halodata.exclusive_sphere_30kpc.molecular_hydrogen_mass;mH2_30kpc_exclusive.convert_to_units('Msun');mH2_30kpc_exclusive=np.array(mH2_30kpc_exclusive.value) #Msun
            sfr_30kpc_exclusive=halodata.exclusive_sphere_30kpc.star_formation_rate;sfr_30kpc_exclusive.convert_to_units('Msun/yr');sfr_30kpc_exclusive=np.array(sfr_30kpc_exclusive.value) #Msun/yr
            rstar=halodata.exclusive_sphere_30kpc.half_mass_radius_stars;rstar.convert_to_units('Mpc');rstar=np.array(rstar.value) #kpc
            disk_to_total_star=halodata.exclusive_sphere_30kpc.disc_to_total_stellar_mass_fraction
            disk_to_total_gas=halodata.exclusive_sphere_30kpc.disc_to_total_gas_mass_fraction
            mbh=halodata.exclusive_sphere_30kpc.most_massive_black_hole_mass;mbh.convert_to_units('Msun');mbh=np.array(mbh.value) #Msun
            angmom=halodata.exclusive_sphere_30kpc.angular_momentum_baryons;angmom.convert_to_units('Msun*Mpc*km/s');angmom.convert_to_physical();angmom=np.array(angmom.value)

            #centre of potential
            cop_halo=halodata.exclusive_sphere_30kpc.centre_of_mass
            cop_halo.convert_to_units('Mpc')
            cops=np.array(cop_halo.value)
            cop_x=np.array(cops[:,0])
            cop_y=np.array(cops[:,1])
            cop_z=np.array(cops[:,2])

            cop_subhalo=halodata.bound_subhalo.centre_of_mass
            cop_subhalo.convert_to_units('Mpc')

            #groupnumber = index for centrals (where hosthaloID==-1), and = hosthaloID for satellites
            groupnumber=np.zeros(halodata.soap.host_halo_index.shape)
            groupnumber[np.where(hosthalo==-1)]=np.arange(len(groupnumber))[hosthalo==-1]
            groupnumber[hosthalo!=-1]=hosthalo[hosthalo!=-1]

            #halo/fof properties
            mfof=halodata.input_halos_fof.masses;mfof.convert_to_units('Msun');mfof=np.array(mfof.value) #Msun
            m200=halodata.spherical_overdensity_200_crit.total_mass;m200.convert_to_units('Msun');m200=np.array(m200.value) #Msun
            r200=halodata.spherical_overdensity_200_crit.soradius;r200.convert_to_units('Mpc');r200=np.array(r200.value) #comoving
            rrel=np.zeros(len(groupnumber))+np.nan
            subgroupnumber=np.zeros(len(groupnumber))

            #give each satellite the group mass, r200 and m200 of the central and distance to central
            for i in range(len(groupnumber)):
                if hosthalo[i]!=-1:
                    subgroupnumber[i]=1
                    m200[i]=m200[hosthalo[i]]
                    r200[i]=r200[hosthalo[i]]
                    mfof[i]=mfof[hosthalo[i]]
                    cop_x[i]=cop_subhalo[i,0]
                    cop_y[i]=cop_subhalo[i,1]
                    cop_z[i]=cop_subhalo[i,2]
                    rrel[i]=np.sqrt((cop_x[i]-cop_x[hosthalo[i]])**2+(cop_y[i]-cop_y[hosthalo[i]])**2+(cop_z[i]-cop_z[hosthalo[i]])**2)


            #make pandas dataframe
            halodata_out=pd.DataFrame()
            halodata_out['Redshift']=np.ones(len(groupnumber))*redshift
            halodata_out['SnapNum']=np.ones(len(groupnumber))*snapnum
            halodata_out['GalaxyID']=snapnum*1e12+galaxyID
            halodata_out['GalaxyID_raw']=galaxyID
            halodata_out['Mass']=subhalomass
            halodata_out['CentreOfPotential_x']=cop_x
            halodata_out['CentreOfPotential_y']=cop_y
            halodata_out['CentreOfPotential_z']=cop_z
            halodata_out['Group_R_Crit200']=r200
            halodata_out['Group_M_Crit200']=m200
            halodata_out['GroupMass']=mfof
            halodata_out['GroupNumber']=groupnumber
            halodata_out['SubGroupNumber']=subgroupnumber
            halodata_out['Group_Rrel']=rrel
            halodata_out['Stellar_Mass_30kpc']=mstar_30kpc_exclusive
            halodata_out['Stellar_Rhalf_30kpc']=rstar
            halodata_out['Gas_Mass_30kpc']=mgas_30kpc_exclusive
            halodata_out['SFR_30kpc']=sfr_30kpc_exclusive
            halodata_out['BH_Mass']=mbh
            halodata_out['HI_Mass_30kpc']=mHI_30kpc_exclusive
            halodata_out['H2_Mass_30kpc']=mH2_30kpc_exclusive
            halodata_out['DiskToTotalStar_30kpc']=disk_to_total_star
            halodata_out['DiskToTotalGas_30kpc']=disk_to_total_gas
            halodata_out['Lbaryon_x']=angmom[:,0]
            halodata_out['Lbaryon_y']=angmom[:,1]
            halodata_out['Lbaryon_z']=angmom[:,2]

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
                