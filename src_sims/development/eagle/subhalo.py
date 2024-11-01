import numpy as np
import pandas as pd
import eagleSqlTools as sql 

# Array of chosen simulations . Entries refer to the simulation name and comoving box length .
sim='RefL0100N1504'

# This uses the eagleSqlTools module to connect to the database with your username and password. # If the password is not given , the module will prompt for it .
con = sql.connect("uname", password="pw")
# Construct and execute query for each simulation. This query returns the number of galaxies # for a given 30 pkpc aperture stellar mass bin (centered with 0.2 dex width).
 
myQuery =f"SELECT \
            Subhalo.Redshift as Redshift, \
            Subhalo.SnapNum as SnapNum, \
            Subhalo.GalaxyID as GalaxyID, \
            Subhalo.SubGroupNumber as SubGroupNumber, \
            Subhalo.CentreOfPotential_x as CentreOfPotential_x, \
            Subhalo.CentreOfPotential_y as CentreOfPotential_y, \
            Subhalo.CentreOfPotential_z as CentreOfPotential_z, \
            Subhalo.Mass as Mass, \
            Subhalo.StarFormationRate as StarFormationRate \
          FROM \
            {sim}_Subhalo as Subhalo\
          WHERE \
            Subhalo.SnapNum>=12 and\
            Subhalo.Mass>=1e10 and \
            Subhalo.SubGroupNumber=0 \
          ORDER BY \
            Subhalo.SnapNum desc, \
            Subhalo.Mass desc \
          "
            
data=sql.execute_query(con, myQuery)
columns=list(data.dtype.names)
data_pd=pd.DataFrame(data,columns=columns)
    
data_pd=data_pd.loc[np.logical_and(data_pd.Mass>1e10,data_pd.SubGroupNumber==0),:]
data_pd.reset_index(inplace=True,drop=True)