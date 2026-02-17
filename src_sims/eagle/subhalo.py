import os
import h5py
import numpy as np
import pandas as pd
import eagleSqlTools as sql 

from hydroflow.run.initialise import load_metadata
from hydroflow.run.tools_catalogue import dump_hdf

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

    # This uses the eagleSqlTools module to connect to the database with your username and password. # If the password is not given , the module will prompt for it .
    con = sql.connect(f"{uname}", password=f"{pw}")
    # Construct and execute query for each simulation. 
    snapnum_strs=["Subhalo.SnapNum="+str(snapnum) for snapnum in snapnums]
    myQueries =[f"SELECT\
                Subhalo.Redshift as Redshift, \
                Subhalo.SnapNum as SnapNum, \
                Subhalo.GalaxyID as GalaxyID, \
                Subhalo.Mass as Mass,  \
                Subhalo.SubGroupNumber as SubGroupNumber, \
                Subhalo.CentreOfPotential_x as CentreOfMass_x, \
                Subhalo.CentreOfPotential_y as CentreOfMass_y, \
                Subhalo.CentreOfPotential_z as CentreOfMass_z, \
                Subhalo.GasSpin_x as subhalodashgas_alldashL_totdashsubfexcl_x, \
                Subhalo.GasSpin_y as subhalodashgas_alldashL_totdashsubfexcl_y, \
                Subhalo.GasSpin_z as subhalodashgas_alldashL_totdashsubfexcl_z, \
                Subhalo.MassType_Gas as subhalodashgas_alldashm_totdashsubfexcl, \
                Subhalo.MassType_Star as subhalodashstardashm_totdashsubfexcl, \
                Subhalo.StarFormationRate as subhalodashgas_alldashSFRdashsubfexcl, \
                Aperture.Mass_Star as flag030pkpc_spheredashstardashm_totdashsubfexcl, \
                Aperture.Mass_Gas as flag030pkpc_spheredashgas_alldashm_totdashsubfexcl, \
                Aperture.Mass_BH as flag030pkpc_spheredashbhdashm_totdashsubfexcl, \
                Aperture.Mass_DM as flag030pkpc_spheredashdmdashm_totdashsubfexcl, \
                Aperture.SFR as flag030pkpc_spheredashgas_alldashSFRdashsubfexcl, \
                Sizes.R_halfmass30 as flag030pkpc_spheredashstardashr_halfdashsubfexcl, \
                FOF.GroupMass as GroupMass, \
                FOF.Group_M_Crit200 as Group_M_Crit200, \
                FOF.Group_R_Crit200 as Group_R_Crit200, \
                FOF.GroupCentreOfPotential_x as GroupCentreOfMass_x, \
                FOF.GroupCentreOfPotential_y as GroupCentreOfMass_y, \
                FOF.GroupCentreOfPotential_z as GroupCentreOfMass_z \
              FROM \
                {simname}_Subhalo as Subhalo,\
                {simname}_Aperture as Aperture,\
                {simname}_Sizes as Sizes,\
                {simname}_FOF as FOF \
             WHERE \
                {snapnum_str} and\
                Subhalo.GroupID = FOF.GroupID and\
                Subhalo.SnapNum = FOF.SnapNum and\
                Subhalo.GalaxyID = Aperture.GalaxyID and \
                Subhalo.GalaxyID = Sizes.GalaxyID and \
                Aperture.ApertureSize = 30 and\
                Subhalo.Mass >= 1e9 and\
                FOF.Group_M_Crit200>= {mcut:2e}\
              ORDER BY \
                Subhalo.SnapNum desc, \
                Subhalo.Mass desc \
              " for snapnum_str in snapnum_strs]


    subcats=[]
    # Execute the query and convert the data to a pandas DataFrame          
    for isnap,myQuery in enumerate(myQueries):
      print('Executing query: ',snapnum_strs[isnap])
      data=sql.execute_query(con, myQuery)
      data=pd.DataFrame(data,columns=list(data.dtype.names))
      data.reset_index(inplace=True,drop=True)
      subcats.append(data)
    
    print('Concatenating subhalo dataframes...')
    # Concatenate the subhalo dataframes
    data_pd=pd.concat(subcats)
    data_pd.reset_index(inplace=True,drop=True)

    # Remove "flag" from column names
    data_pd.columns=data_pd.columns.str.replace('flag','')

    # Use '-' instead of 'dash' in column names
    data_pd.columns=data_pd.columns.str.replace('dash','-')

    # Convert GasSpin to angular momentum
    data_pd['subhalo-gas_all-L_tot-subfexcl_x']=data_pd['subhalo-gas_all-L_tot-subfexcl_x']*data_pd['subhalo-gas_all-m_tot-subfexcl']
    data_pd['subhalo-gas_all-L_tot-subfexcl_y']=data_pd['subhalo-gas_all-L_tot-subfexcl_y']*data_pd['subhalo-gas_all-m_tot-subfexcl']
    data_pd['subhalo-gas_all-L_tot-subfexcl_z']=data_pd['subhalo-gas_all-L_tot-subfexcl_z']*data_pd['subhalo-gas_all-m_tot-subfexcl']
    
    # Convert Group_R_Crit200 to cMpc
    data_pd['Group_R_Crit200']=data_pd['Group_R_Crit200']/1e3*(1+data_pd['Redshift'])

    # Convert R_halfmass30 to pMpc
    data_pd['030pkpc_sphere-star-r_half-subfexcl']=data_pd['030pkpc_sphere-star-r_half-subfexcl']/1e3

    # Add Rrel to the subhalo catalogue
    data_pd['Group_Rrel']=np.sqrt((data_pd['CentreOfPotential_x']-data_pd['GroupCentreOfPotential_x'])**2+(data_pd['CentreOfPotential_y']-data_pd['GroupCentreOfPotential_y'])**2+(data_pd['CentreOfPotential_z']-data_pd['GroupCentreOfPotential_z'])**2)

    # Output path
    outpath=os.getcwd()+'/catalogues/subhaloes.hdf5'
    dump_hdf(outpath,data_pd)

    # Add path to metadata in hdf5
    if metadata is not None:
        with h5py.File(outpath, 'r+') as subcatfile:
            header= subcatfile.create_group("Header")
            header.attrs['metadata'] = metadata_path
    else:
        print("No metadata file found. Metadata path not added to subhalo catalogue.")

    # Return the subhalo catalogue
    return data_pd