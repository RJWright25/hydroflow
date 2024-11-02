import os
import h5py
import numpy as np
import pandas as pd
import eagleSqlTools as sql 

from hydroflow.run.initialise import load_metadata
from hydroflow.run.tools_catalog import dump_hdf

def extract_subhaloes(simname='RefL0100N1504',snapnums=[],uname=None,pw=None,mcut=1e11,metadata=None):
    """
    extract_subhaloes: Read the subhalo catalog from an EAGLE simulation using the eagleSqlTools module. This massages the data into the preferred format for the subhalo catalog, and saves it to a HDF5 file.

    Input:
    -----------

    simname: str
        Name of the EAGLE simulation (in the format e.g. 'RefL0100N1504').
    uname: str
        Username for the database.
    pw: str
        Password for the database.
    mcut: float
        Minimum mass of subhaloes to include [log10(M/Msun)].
    metadata: str
        Path to the metadata file.

    """
        
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

    # This uses the eagleSqlTools module to connect to the database with your username and password. # If the password is not given , the module will prompt for it .
    con = sql.connect(f"{uname}", password=f"{pw}")
    # Construct and execute query for each simulation. 
    snapnum_str=["SnapNum="+str(snapnum) for snapnum in snapnums]
    snapnum_str=" OR ".join(snapnum_str)
    print(snapnum_str)
    myQuery =f"SELECT\
                Subhalo.Redshift as Redshift, \
                Subhalo.SnapNum as SnapNum, \
                Subhalo.GalaxyID as GalaxyID, \
                Subhalo.SubGroupNumber as SubGroupNumber, \
                Subhalo.CentreOfPotential_x as CentreOfPotential_x, \
                Subhalo.CentreOfPotential_y as CentreOfPotential_y, \
                Subhalo.CentreOfPotential_z as CentreOfPotential_z, \
                Subhalo.Mass as Mass, \
                Aperture.Mass_Star as 030pkpc_sphere-star-m_tot-subfexcl, \
                Aperture.Mass_Gas as 030pkpc_sphere-gas_all-m_tot-subfexcl, \
                Aperture.Mass_BH as 030pkpc_sphere-bh_m_tot-subfexcl, \
                Aperture.Mass_DM as 030pkpc_sphere-dm_m_tot-subfexcl, \
                Aperture.SFR as 030pkpc_sphere-gas_all-SFR-subfexcl, \
                Subhalo.GasSpin_x*Subhalo.MassType_Gas as subhalo-gas-L_tot-subfexcl_x, \
                Subhalo.GasSpin_y*Subhalo.MassType_Gas as subhalo-gas-L_tot-subfexcl_y, \
                Subhalo.GasSpin_z*Subhalo.MassType_Gas as subhalo-gas-L_tot-subfexcl_z, \
                Sizes.R_halfmass30 as R_halfmass30 as 030pkpc_sphere-star-r_half-subfexcl, \
                FOF.GroupMass as GroupMass, \
                FOF.Group_M_Crit200 as Group_M_Crit200, \
                FOF.Group_R_Crit200 as Group_R_Crit200, \
                FOF.GroupCentreOfPotential_x, \
                square(Subhalo.CentreOfPotential_x-FOF.GroupCentreOfPotential_x) \
                      + square(Subhalo.CentreOfPotential_y-FOF.GroupCentreOfPotential_y) \
                      + square(Subhalo.CentreOfPotential_z-FOF.GroupCentreOfPotential_z) as Group_Rrel \
              FROM \
                {simname}_Subhalo as Subhalo,\
                {simname}_Aperture as Aperture,\
                {simname}_Sizes as Sizes,\
                {simname}_FOF as FOF \
             WHERE \
                {snapnum_str} and\
                Subhalo.GalaxyID = Aperture.GalaxyID and \
                Subhalo.GalaxyID = Sizes.GalaxyID and \
                Subhalo.GroupID = FOF.GroupID and\
                Subhalo.SnapNum = FOF.SnapNum and\
                Subhalo.Mass>=1e{mcut:.0f} \
              ORDER BY \
                Subhalo.SnapNum desc, \
                Subhalo.Mass desc \
              "
    print(myQuery)

    # Execute the query and convert the data to a pandas DataFrame          
    data=sql.execute_query(con, myQuery)
    columns=list(data.dtype.names)
    data_pd=pd.DataFrame(data,columns=columns)
    data_pd.reset_index(inplace=True,drop=True)

    # Convert Group_R_Crit200 to cMpc
    data_pd['Group_R_Crit200']=data_pd['Group_R_Crit200']/1e3*(1+data_pd['Redshift'])

    # Convert R_halfmass30 to pMpc
    data_pd['R_halfmass30']=data_pd['R_halfmass30']/1e3

    # Output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'
    dump_hdf(outpath,data_pd)

    # Add path to metadata in hdf5
    if metadata is not None:
        with h5py.File(outpath, 'r+') as subcatfile:
            header= subcatfile.create_group("Header")
            header.attrs['metadata'] = metadata
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")