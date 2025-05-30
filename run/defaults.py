# Default runtime parameters
r200_shells=[0.05,0.1,0.2,0.25,0.3,0.5,0.75,1,1.5,2,2.5,3] # Shells as fraction of r200
rstar_shells=[0.5,1,2,4] # Shells as fraction of stellar half mass radius
kpc_shells=[1,2,5,10,15,20,25,30,50,75,100] # Shells in pkpc 
zslab_radii={'rmx10pkpc':10,'rmxzheight':1}
Tbins={'cold':[0,1e3],'cool':[1e3,1e5],'warm':[1e5,1e7],'hot':[1e7,1e15]} # Temperature bins for inflow/outflow calculations
theta_bins={'minax':[30,90],'full':[0,90]} # Angular bins for inflow/outflow calculations
vcuts={'vc0p25vmx':'0.25Vmax','vc1p00vmx':'1.00Vmax','vc050kmps':50,'vc250kmps':250}
drfacs=[0.1] # Fraction of shell radius to use for shell thickness (r-dr/2 - r+dr/2)

# Particle data fields to dump if requested
pdata_fields=['Masses',
              'Relative_r_comoving',
              'Coordinates_x',
              'Coordinates_y',
              'Coordinates_z',
              'Relative_vrad_pec',
              'Relative_vx_pec',
              'Relative_vy_pec',
              'Relative_vz_pec',
              'Relative_theta_pos',
              'Relative_theta_vel',
              'Temperature',
              'Density',
              'Metallicity']
