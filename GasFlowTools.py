import os
import sys
import time
import logging
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from read_eagle import EagleSnapshot
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
from astropy import units

h=0.6777
nh_conversion_prefac=6.76991e-31/(1.6726219e-24)*0.752*h**2

#prep for gasflow calcs

def submit_function(function,arguments,memory,time):
    filename=sys.argv[0]
    cwd=os.getcwd()
    run=cwd.split('/')[-1]

    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    if function=='analyse_gasflow':
        jobname=function+'_'+run+f"_snapidx_{str(arguments['snapidx']).zfill(3)}_n_{str(arguments['nvol']).zfill(2)}_volume_{str(arguments['ivol']).zfill(3)}_delta{str(arguments['snapidx_delta']).zfill(2)}"
    elif function=='analyse_subhalo':
        jobname=function+'_'+run+f"_snapidx_{str(arguments['snapidx']).zfill(3)}_n_{str(arguments['nvol']).zfill(2)}_volume_{str(arguments['ivol']).zfill(3)}"
    elif function=='combine_catalogues':
        jobname=function+'_'+run+f"_n_{str(arguments['nvol']).zfill(2)}_delta_{str(arguments['snapidx_delta']).zfill(2)}"
    else:
        jobname=function+'_'+run
    
    runscriptfilepath=f'{cwd}/jobs/{jobname}-run.py'
    jobscriptfilepath=f'{cwd}/jobs/{jobname}-submit.slurm'
    if os.path.exists(jobscriptfilepath):
        os.remove(runscriptfilepath)
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    argumentstring=''
    for arg in arguments:
        if type(arguments[arg])==str:
            argumentstring+=f"{arg}='{arguments[arg]}',"
        else:
            argumentstring+=f"{arg}={arguments[arg]},"


    with open(runscriptfilepath,"w") as runfile:
        runfile.writelines(f"import warnings\n")
        runfile.writelines(f"warnings.filterwarnings('ignore')\n")
        runfile.writelines(f"import sys\n")
        runfile.writelines(f"sys.path.append('/home/rwright/Software/gasflow')\n")
        runfile.writelines(f"from GasFlowTools import *\n")
        runfile.writelines(f"{function}({argumentstring})")
    runfile.close()

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node={1}\n")
        jobfile.writelines(f"#SBATCH --mem={memory}GB\n")
        jobfile.writelines(f"#SBATCH --time={time}\n")
        jobfile.writelines(f"#SBATCH --output=jobs/{jobname}.out\n")
        jobfile.writelines(f" \n")
        jobfile.writelines(f"OMPI_MCA_mpi_warn_on_fork=0\n")
        jobfile.writelines(f"export OMPI_MCA_mpi_warn_on_fork\n")
        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {runscriptfilepath} \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")
    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")

def extract_tree(path,mcut,snapidxmin=0):

    outname='catalogues/catalogue_tree.hdf5'
    fields=['snapshotNumber',
            'nodeIndex',
            'fofIndex',
            'hostIndex',
            'descendantIndex',
            'mainProgenitorIndex',
            'enclosingIndex',
            'isFoFCentre',
            'positionInCatalogue']

    mcut=10**mcut/10**10*h

    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    if not os.path.exists('catalogues'):
        os.mkdir('catalogues')
    
    if os.path.exists('logs/extract_tree.log'):
        os.remove('logs/extract_tree.log')

    logging.basicConfig(filename='logs/extract_tree.log', level=logging.INFO)
    logging.info(f'Running tree extraction for haloes with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    # get file names
    tree_fnames=os.listdir(path)
    tree_fnames=[tree_fname for tree_fname in tree_fnames if 'tree' in tree_fname]
    nfiles=len(tree_fnames)

    # iterate through all tree files
    t0=time.time()
    for ifile,tree_fname in enumerate(tree_fnames):
        logging.info(f'Processing file {ifile+1} of {nfiles}')
        treefile=h5py.File(f'{path}/{tree_fname}')

        #mass mask
        masses=treefile['/haloTrees/nodeMass'][:];snipshotidx=treefile['/haloTrees/snapshotNumber'][:]
        mask=np.logical_and(masses>mcut,snipshotidx>=snapidxmin)

        #initialise new data
        logging.info(f'Extracting position for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        newdata=pd.DataFrame(treefile['/haloTrees/position'][mask,:],columns=['position_x','position_y','position_z'])

        #grab all fields
        logging.info(f'Extracting data for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        newdata.loc[:,fields]=np.column_stack([treefile['/haloTrees/'+field][mask] for field in fields])

        #append to data frame
        if ifile==0:
            data=newdata
        else:
            data=data.append(newdata,ignore_index=True)


        #close file, move to next
        treefile.close()

    if os.path.exists(outname):
        os.remove(outname)
        
    data.to_hdf(f'{outname}',key='Tree')

def extract_fof(path,mcut,snapidxmin=0):
    outname='catalogues/catalogue_fof.hdf5'
    fields=['/FOF/GroupMass',
            '/FOF/Group_M_Crit200',
            '/FOF/Group_R_Crit200',
            '/FOF/NumOfSubhalos',
            '/FOF/GroupCentreOfPotential']

    mcut=10**mcut/10**10*h
    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')
    dims='xyz'

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+'/'+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    if os.path.exists('logs/extract_fof.log'):
        os.remove('logs/extract_fof.log')

    logging.basicConfig(filename='logs/extract_fof.log', level=logging.INFO)
    logging.info(f'Running FOF extraction for FOFs with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    t0=time.time()
    ifile=0
    isnap=-1
    for groupdir in groupdirs:
        snap=int(groupdir.split('snip_')[-1][:3])
        try:
            snapidx=redshift_table.loc[snap==redshift_table['snapshot'],'snapshotidx'].values[0]
        except:
            logging.info(f'Skipping snap {snapidx} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snapidx>=snapidxmin:
            isnap+=1
            logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(groupdir)} total) [runtime {time.time()-t0:.2f} sec]')
            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                groupdirifile=h5py.File(groupdirfname,'r')

                ifile_fofmasses=groupdirifile['/FOF/GroupMass'].value
                ifile_mask=ifile_fofmasses>mcut
                ifile_nfof=np.sum(ifile_mask)
    
                logging.info(f'Snap {snapidx} ({isnap+1}/{len(groupdir)} total), file {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} FOFs [runtime {time.time()-t0:.2f} sec]')
                
                if ifile_nfof:
                    newdata=pd.DataFrame(groupdirifile['/FOF/GroupMass'][ifile_mask],columns=['GroupMass'])
                    newdata.loc[:,'snapshotidx']=snapidx

                    for field in fields:
                        dset_shape=groupdirifile[field].shape
                        if len(dset_shape)==2:
                            for icol in range(dset_shape[-1]):
                                if dset_shape[-1]==3:
                                    newdata.loc[:,field.split('FOF/')[-1]+f'_{dims[icol]}']=groupdirifile[field][ifile_mask,icol]
                                else:
                                    if icol in [0,1,4,5]:
                                        newdata.loc[:,field.split('FOF/')[-1]+f'_{icol}']=groupdirifile[field][ifile_mask,icol]
                        else:
                            newdata.loc[:,field.split('FOF/')[-1]]=groupdirifile[field][ifile_mask]

                    if ifile==0:
                        data=newdata
                    else:
                        data=data.append(newdata,ignore_index=True)

                    ifile+=1
                    groupdirifile.close()
        
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')

    data.to_hdf(f'{outname}',key='FOF')

def extract_subhalo(path,mcut,snapidxmin=0,overwrite=True):
    outname='catalogues/catalogue_subhalo.hdf5'
    fields=['/Subhalo/GroupNumber',
            '/Subhalo/SubGroupNumber',
            '/Subhalo/Mass',
            '/Subhalo/MassType',
            '/Subhalo/ApertureMeasurements/Mass/030kpc',
            '/Subhalo/ApertureMeasurements/SFR/030kpc',
            '/Subhalo/Vmax',
            '/Subhalo/CentreOfPotential',
            '/Subhalo/Velocity',
            '/Subhalo/CentreOfMass',
            '/Subhalo/HalfMassRad']

    mcut=10**mcut/10**10*h
    redshift_table=pd.read_hdf('snapshot_redshifts.hdf5',key='snapshots')
    dims='xyz'

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+'/'+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    if os.path.exists('logs/extract_subhalo.log'):
        os.remove('logs/extract_subhalo.log')

    logging.basicConfig(filename='logs/extract_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for subhaloes with mass above {mcut*10**10:.1e} after (and including) snapidx {snapidxmin} ...')

    t0=time.time()
    ifile=0
    isnap=-1
    for groupdir in groupdirs:
        snap=int(groupdir.split('snip_')[-1][:3])
        try:
            snapidx=redshift_table.loc[snap==redshift_table['snapshot'],'snapshotidx'].values[0]
        except:
            logging.info(f'Skipping snap {snapidx} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snapidx>=snapidxmin:
            isnap+=1
            logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(groupdirs)} total) [runtime {time.time()-t0:.2f} sec]')
            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                groupdirifile=h5py.File(groupdirfname,'r')

                ifile_submasses=groupdirifile['/Subhalo/Mass'].value
                ifile_mask=ifile_submasses>mcut
                ifile_nfof=np.sum(ifile_mask)
    
                logging.info(f'Snap {snapidx} ({isnap+1}/{len(groupdirs)} total), file {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} subhaloes [runtime {time.time()-t0:.2f} sec]')
                
                if ifile_nfof:
                    newdata=pd.DataFrame(groupdirifile['/Subhalo/Mass'][ifile_mask],columns=['Mass'])
                    newdata.loc[:,'snapshotidx']=snapidx

                    for field in fields:
                        dset_shape=groupdirifile[field].shape
                        if len(dset_shape)==2:
                            for icol in range(dset_shape[-1]):
                                if dset_shape[-1]==3:
                                    newdata.loc[:,field.split('Subhalo/')[-1]+f'_{dims[icol]}']=groupdirifile[field][ifile_mask,icol]
                                else:
                                    if icol in [0,1,4,5]:
                                        newdata.loc[:,field.split('Subhalo/')[-1]+f'_{icol}']=groupdirifile[field][ifile_mask,icol]
                        else:
                            newdata.loc[:,field.split('Subhalo/')[-1]]=groupdirifile[field][ifile_mask]

                    if ifile==0:
                        data=newdata
                    else:
                        data=data.append(newdata,ignore_index=True)

                    ifile+=1
                    groupdirifile.close()
    try:
        if overwrite:
            
            if os.path.exists(f'{outname}'):
                os.remove(f'{outname}')
            data.to_hdf(f'{outname}',key='Subhalo')

        else:
            logging.info(f'Loading old catalogue ...')
            data_old=pd.read_hdf(f'{outname}',key='Subhalo')
            fields_new=list(data)
            fields_old=list(data_old)
            fields_new_mask=np.isin(fields_new,fields_old,invert=True)
            fields_to_add=fields_new[np.where(fields_new_mask)]
            for field_new in fields_to_add:
                logging.info(f'Adding new field to old catalogue: {field_new}')
                data_old.loc[:,field_new]=data[field_new].values

            data_old.to_hdf(f'{outname}',key='Subhalo')
    except:
        data.to_hdf(f'catalogues/catalogue_subhalo-BACKUP.hdf5',key='Subhalo')

    pass

def match_tree(mcut,snapidxs=[]):

    outname='catalogues/catalogue_subhalo.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo',mode='r')
    catalogue_tree=pd.read_hdf('catalogues/catalogue_tree.hdf5',key='Tree',mode='r')
    fields_tree=['nodeIndex',
                 'fofIndex',
                 'hostIndex',
                 'descendantIndex',
                 'mainProgenitorIndex']


    mcut=10**mcut/10**10

    if os.path.exists('logs/match_tree.log'):
        os.remove('logs/match_tree.log')

    logging.basicConfig(filename='logs/match_tree.log', level=logging.INFO)
    logging.info(f'Running tree matching for subhaloes with mass above {mcut*10**10:.1e} for {len(snapidxs)} snaps ')

    nsub_tot=catalogue_subhalo.shape[0]

    t0=time.time()
    for isnap,snapidx in enumerate(snapidxs):
        logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')
        snap_mass_mask=np.logical_and(catalogue_subhalo['snapshotidx']==snapidx,catalogue_subhalo['Mass']>mcut)
        snap_catalogue_subhalo=catalogue_subhalo.loc[snap_mass_mask,:]
        for field in fields_tree:
            snap_catalogue_subhalo.loc[:,field]=-1
        snap_tree_catalogue=catalogue_tree.loc[catalogue_tree['snapshotNumber']==snapidx,:]
        snap_tree_coms=snap_tree_catalogue.loc[:,[f'position_{x}' for x in 'xyz']].values

        iisub=0;nsub_snap=snap_catalogue_subhalo.shape[0]
        t0halo=time.time()
        for isub,sub in snap_catalogue_subhalo.iterrows():
            isub_com=[sub[f'CentreOfPotential_{x}'] for x in 'xyz']
            isub_match=np.sqrt(np.sum(np.square(snap_tree_coms-isub_com),axis=1))==0
            isnap_match=snap_catalogue_subhalo.index==isub
            if np.sum(isub_match):
                isub_treedata=snap_tree_catalogue.loc[isub_match,fields_tree].values
                snap_catalogue_subhalo.loc[isnap_match,fields_tree]=isub_treedata
            else:
                logging.info(f'Warning: could not match subhalo {iisub} at ({isub_com[0]:.2f},{isub_com[1]:.2f},{isub_com[2]:.2f}) cMpc')
                pass

            if not iisub%100:
                logging.info(f'Done matching {(iisub+1)/nsub_snap*100:.1f}% of subhaloes at snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')

            iisub+=1
        
        catalogue_subhalo.loc[snap_mass_mask,:]=snap_catalogue_subhalo
        print(catalogue_subhalo)

    os.remove(outname)
    catalogue_subhalo.to_hdf(outname,key='Subhalo')

def match_fof(mcut,snapidxs=[]):

    outname='catalogues/catalogue_subhalo.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo',mode='r')
    catalogue_fof=pd.read_hdf('catalogues/catalogue_fof.hdf5',key='FOF',mode='r')
    fields_fof=['GroupMass',
                'Group_M_Crit200',
                'Group_R_Crit200',
                'NumOfSubhalos']

    mcut=10**mcut/10**10*h

    if os.path.exists('logs/match_fof.log'):
        os.remove('logs/match_fof.log')

    logging.basicConfig(filename='logs/match_fof.log', level=logging.INFO)
    logging.info(f'Running FOF matching for subhaloes with mass above {mcut*10**10:.1e} for {len(snapidxs)} snaps ...')

    
    t0=time.time()
    for isnap,snapidx in enumerate(snapidxs):
        logging.info(f'Processing snap {snapidx} ({isnap+1}/{len(snapidxs)}) [runtime {time.time()-t0:.2f} sec]')
        snap_mass_mask=np.logical_and(catalogue_subhalo['snapshotidx']==snapidx,catalogue_subhalo['Mass']>mcut)
        central_mask=np.logical_and.reduce([snap_mass_mask,catalogue_subhalo['SubGroupNumber']==0])
        snap_catalogue_subhalo=catalogue_subhalo.loc[snap_mass_mask,:]
        for field in fields_fof:
            snap_catalogue_subhalo.loc[:,field]=-1
        snap_central_catalogue=catalogue_subhalo.loc[central_mask,:]

        logging.info(f'Matching for {np.sum(central_mask)} groups with centrals above {mcut*10**10:.1e}msun at snipshot {snapidx} [runtime {time.time()-t0:.2f} sec]')
        central_coms=snap_central_catalogue.loc[:,[f"CentreOfPotential_{x}" for x in 'xyz']].values
        central_groupnums=snap_central_catalogue.loc[:,f"GroupNumber"].values

        fofcat_snap=catalogue_fof.loc[catalogue_fof['snapshotidx']==snapidx,:]
        fofcat_coms=catalogue_fof.loc[catalogue_fof['snapshotidx']==snapidx,[f"GroupCentreOfPotential_{x}" for x in 'xyz']].values
        for icentral,(central_com,central_groupnum) in enumerate(zip(central_coms,central_groupnums)):
            if icentral%1000==0:
                logging.info(f'Processing group {icentral+1} of {np.sum(central_mask)} at snipshot {snapidx} ({icentral/np.sum(central_mask)*100:.1f}%) [runtime {time.time()-t0:.2f} sec]')
            fofmatch=np.sum(np.square(fofcat_coms-central_com),axis=1)<=(0.001)**2
            ifofmatch_data=fofcat_snap.loc[fofmatch,fields_fof].values
            ifofsubhaloes=snap_catalogue_subhalo['GroupNumber']==int(central_groupnum)
            if np.sum(ifofsubhaloes):
                snap_catalogue_subhalo.loc[ifofsubhaloes,fields_fof]=ifofmatch_data
            else:
                logging.info(f'Warning: no matching group for central {icentral}')
        
        catalogue_subhalo.loc[snap_mass_mask,:]=snap_catalogue_subhalo

    os.remove(outname)
    catalogue_subhalo.to_hdf(outname,key='Subhalo')

def postprocess_subhalo(path):
    catalogue_subhalo=pd.read_hdf(path,key='Subhalo')
    for field in catalogue_subhalo:
        print(field)
        if 'Mass' in field and 'Centre' not in field:
            catalogue_subhalo[field]=catalogue_subhalo[field]*10**10/h
        elif 'Group_M_Crit' in field:
            catalogue_subhalo[field]=catalogue_subhalo[field]*10**10/h
    
    catalogue_subhalo.to_hdf(path,key='Subhalo')

# gasflow calcs

def analyse_subhalo(path,snapidx,nvol,ivol,snip=True):
    snapidx=int(snapidx)
    ivol=int(ivol)
    ivol=str(ivol).zfill(3)
    ix,iy,iz=ivol_idx(ivol,nvol=nvol)

    r200_bins=np.linspace(0,1,101)
    r200_bins_mid=r200_bins[1:]

    t0=time.time()
    logfile=f'logs/subhalo/subhalo_snapidx_{snapidx}_n_{str(nvol).zfill(2)}_volume_{ivol}.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO)
    
    output_fname=f'catalogues/subhalo/subhalo_snapidx_{snapidx}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5'
    if os.path.exists(output_fname):
        os.remove(output_fname)

    redshift_table=pd.read_hdf('../snapshot_redshifts.hdf5',key='snapshots')

    #determine particle data files to use
    if snip:
        fullpd=False
    else:
        fullpd=True

    if not fullpd and snip:
        snapidx_tag=str(redshift_table.loc[redshift_table['snapshotidx']==snapidx,'tag'].values[0])
        snapidx_particledatapath=f'{path}particledata_{snapidx_tag}/eagle_subfind_snip_particles_{snapidx_tag[5:]}.0.hdf5'
        native_sfr=False

    else:
        native_sfr=True
        snapidx_particledatafolder=[snapdir for snapdir in os.listdir(path) if f'snapshot_{str(snapidx).zfill(3)}' in snapdir][0]
        snapidx_tag=snapidx_particledatafolder.split('snapshot_')[-1]
        snapidx_particledatapath=f'{path}snapshot_{snapidx_tag}/snap_{snapidx_tag}.0.hdf5'

    print(f'Using particle data at {snapidx_particledatapath}')
    logging.info(f'Using particle data at {snapidx_particledatapath}')

    #read data
    boxsize=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['BoxSize']
    redshift=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['Redshift']
    scalefactor=1/(1+redshift)
    cosmology=FlatLambdaCDM(H0=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['HubbleParam']*100,
                            Om0=h5py.File(snapidx_particledatapath,'r')['Header'].attrs['Omega0'])

    rhocrit=cosmology.critical_density(redshift)
    rhocrit=rhocrit.to(units.Msun/units.Mpc**3)
    rhocrit=rhocrit.value

    subvol_edgelength=boxsize/nvol
    buffer=2

    xmin=ix*subvol_edgelength;xmax=(ix+1)*subvol_edgelength
    ymin=iy*subvol_edgelength;ymax=(iy+1)*subvol_edgelength
    zmin=iz*subvol_edgelength;zmax=(iz+1)*subvol_edgelength

    logging.info(f'Considering region: (1/{nvol**3} of full box) [runtime = {time.time()-t0:.2f}s]')
    logging.info(f'ix: {ix} - x in [{xmin},{xmax}]')
    logging.info(f'iy: {iy} - y in [{ymin},{ymax}]')
    logging.info(f'iz: {iz} - z in [{zmin},{zmax}]')

    snapidx_eagledata = EagleSnapshot(snapidx_particledatapath)
    snapidx_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)

    logging.info(f'Initialising particle data with IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap=pd.DataFrame(snapidx_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs'])
    particledata_snap.loc[:,"ParticleTypes"]=0

    particledata_snap_star=pd.DataFrame(snapidx_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs'])
    particledata_snap_star.loc[:,"ParticleTypes"]=4;particledata_snap_star.loc[:,"Temperature"]=-1.;particledata_snap_star.loc[:,"Density"]=10**10

    logging.info(f'Reading gas datasets [runtime = {time.time()-t0:.2f}s]')
    gas_fields=['Coordinates','Mass','Density','Temperature','Metallicity']

    if native_sfr:
        gas_fields.append('StarFormationRate')

    for dset in gas_fields:
        dset_snap=snapidx_eagledata.read_dataset(0,dset)
        if dset_snap.shape[-1]==3:
                particledata_snap.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap
        else:
            if dset=='Mass':
                particledata_snap[dset]=dset_snap*10**10/h
            else:
                particledata_snap[dset]=dset_snap

    logging.info(f'Reading star datasets [runtime = {time.time()-t0:.2f}s]')
    star_fields=['Coordinates','Mass']
    for dset in star_fields:
        dset_snap=snapidx_eagledata.read_dataset(4,dset)
        if dset_snap.shape[-1]==3:
            particledata_snap_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap
        else:
            if dset=='Mass':
                particledata_snap_star[dset]=dset_snap*10**10/h
            else:
                particledata_snap_star[dset]=dset_snap

    logging.info(f'Done reading datasets - concatenating gas and star data [runtime = {time.time()-t0:.2f}s]')
    particledata_snap=particledata_snap.append(particledata_snap_star,ignore_index=True)

    logging.info(f'Sorting by IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap.sort_values(by="ParticleIDs",inplace=True);particledata_snap.reset_index(inplace=True,drop=True)
    
    #catalog size
    size1=np.sum(particledata_snap.memory_usage().values)/10**10
    logging.info(f'Particle data snap 1 memory usage: {size1:.2f} GB')

    #particle KD treee
    logging.info(f'Obtaining KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
    kdtree_snap1_fname=f'catalogues/kdtrees/kdtree_snapidx_{str(snapidx).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.dat'
    
    gen1=True
    if os.path.exists(kdtree_snap1_fname):
        try:
            logging.info(f'Getting KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
            kdtree_snap1_periodic=open_pickle(path=kdtree_snap1_fname)
            gen1=False
        except:
            pass
    if gen1:
        logging.info(f'Generating KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap1_periodic=cKDTree(np.column_stack([particledata_snap[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        dump_pickle(data=kdtree_snap1_periodic,path=kdtree_snap1_fname)

    #load catalogues into dataframes
    logging.info(f'Loading base catalogue [runtime = {time.time()-t0:.2f}s]')
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')

    #select relevant subhaloes
    print(catalogue_subhalo[f'snapshotidx'])
    snap_mask=np.abs(catalogue_subhalo[f'snapshotidx']-snapidx)<0.1
    snap_mass_mask=catalogue_subhalo[f'snapshotidx']>=0
    snap_com_mask_1=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']>=ixmin for x,ixmin in zip('xyz',[xmin,ymin,zmin])])
    snap_com_mask_2=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']<=ixmax for x,ixmax in zip('xyz',[xmax,ymax,zmax])])
    snap_com_mask=np.logical_and.reduce([snap_com_mask_1,snap_com_mask_2,snap_mask,snap_mass_mask])
    numgal_subvolume=np.sum(snap_com_mask);numgal_total=np.sum(np.logical_and(snap_mask,snap_mass_mask))
    logging.info(f'Will use  {numgal_subvolume} of {numgal_total} valid galaxies from box [runtime = {time.time()-t0:.2f}s]')

    #initialise output
    if snip:
        initfields=['nodeIndex','GroupNumber','SubGroupNumber']
    else:
        initfields=['GalaxyID','GroupNumber','SubGroupNumber']

    subhalo_subset=catalogue_subhalo.loc[snap_com_mask,:].copy()
    subhalo_subset.reset_index(drop=True,inplace=True)
    output_df=subhalo_subset.loc[:,initfields].copy()
    output_df.reset_index(drop=True,inplace=True)

    output_df.loc[:,'BaryMP-factor']=np.nan
    output_df.loc[:,'BaryMP-radius']=np.nan
    output_df.loc[:,'BaryMP-mgas_ism']=np.nan
    output_df.loc[:,'BaryMP-mgas_sph']=np.nan
    output_df.loc[:,'BaryMP-mstar']=np.nan
    output_df.loc[:,'BaryMP-npart']=np.nan
    output_df.loc[:,'BaryMP-SFR']=np.nan
    output_df.loc[:,'BaryMP-Z']=np.nan
    output_df.loc[:,'BaryMP-nfit']=np.nan
    
    success=[]
    for igalaxy,galaxy in subhalo_subset.iterrows():
        subgroupnumber=galaxy['SubGroupNumber']
        if subgroupnumber==0:
            icen=True
        else:
            icen=False

        com=[galaxy[f"CentreOfPotential_{x}"] for x in 'xyz']

        #select particles in halo-size sphere
        if icen:
            r200_eff=galaxy['Group_R_Crit200']
        else:
            r200_eff=r200(m200=galaxy['Mass'],rhocrit=rhocrit)
        
        logging.info(f'Processing galaxy {igalaxy+1} of {numgal_subvolume} for this subvolume - SGN {subgroupnumber} [runtime = {time.time()-t0:.2f}s]')
        logging.info(f"Using effective radius {r200_eff} cMpc for mass {galaxy['Mass']:.1e} Msun")
        print(f"Processing galaxy {igalaxy+1} of {numgal_subvolume} for this subvolume - SGN {subgroupnumber} [runtime = {time.time()-t0:.2f}s]")
        print(f"Using effective radius {r200_eff} cMpc for mass {galaxy['Mass']:.1e} Msun")

        part_idx_within_radius=kdtree_snap1_periodic.query_ball_point(com,r200_eff)
        part_IDs_within_radius=(particledata_snap.loc[part_idx_within_radius,"ParticleIDs"].values).astype(np.int64)
        part_idx_candidates=particledata_snap['ParticleIDs'].searchsorted(part_IDs_within_radius)
        part_data_candidates=particledata_snap.loc[part_idx_candidates,:]
        part_data_candidates.loc[:,"rrel_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates[f'Coordinates_{x}'].values-com[ix] for ix,x in enumerate('xyz')])),axis=1))/r200_eff #Mpc

        #fit baryon mass profile
        #select cold gas
        npart=len(part_idx_candidates)

        if npart>10:
            cool=part_data_candidates["Temperature"].values<=5*10**4
            part_data_selection=part_data_candidates.loc[cool,:]

            rrel=part_data_selection["rrel_com"].values
            mass=part_data_selection["Mass"].values
            Z=part_data_selection["Metallicity"].values
            types=part_data_selection["ParticleTypes"].values

            if native_sfr:
                sfr=part_data_selection["StarFormationRate"].values
            else:
                nH=part_data_selection["Density"].values*nh_conversion_prefac*scalefactor**(-3)
                sfr=np.zeros(npart)
                oneos=nH>0.1*(Z/0.002)**(-0.64)
                eos=np.where(oneos)
                sfr[eos]=(7e-10)*part_data_selection["Mass"].values[eos]*(nH[eos]**(4/15))

            masks=[rrel<bin_hi for bin_lo, bin_hi in zip(r200_bins[:-1],r200_bins[1:])]
            mass_binned_cumulative=[np.nansum(mass[np.where(mask)]) for mask in masks]
            mass_binned_cumulative=mass_binned_cumulative/mass_binned_cumulative[-1]

            try:
                barymp_fac,nfit=BaryMP(r200_bins_mid,mass_binned_cumulative)
            except:
                barymp_fac,nfit=np.nan,0
        else:
            barymp_fac,nfit=np.nan,0

        barymp_sfr=np.nan;barymp_mstar=np.nan;barymp_Z=np.nan
        barymp_mgas_ism=np.nan;barymp_mgas_sph=np.nan

        if nfit:
            barymp_star_mask=np.logical_and.reduce([rrel<barymp_fac,types==4])
            barymp_gas_mask=np.logical_and.reduce([rrel<barymp_fac,types==0])            
            barymp_mstar=np.nansum(mass[np.where(barymp_star_mask)])
            if np.sum(barymp_gas_mask):
                barymp_Z=np.nansum(Z[np.where(barymp_gas_mask)]*mass[np.where(barymp_gas_mask)])/np.nansum(mass[np.where(barymp_gas_mask)])
                barymp_sfr=np.nansum(sfr[np.where(barymp_gas_mask)])
                barymp_mgas_ism=np.nansum(mass[np.where(barymp_gas_mask)])
                all_barymp_mask=np.logical_and(part_data_candidates.loc[:,"rrel_com"].values<barymp_fac,part_data_candidates.loc[:,"ParticleTypes"].values==0)
                barymp_mgas_sph=np.nansum(part_data_candidates.loc[all_barymp_mask,"Mass"])

        output_df.loc[igalaxy,'BaryMP-npart']=npart
        output_df.loc[igalaxy,'BaryMP-nfit']=nfit
        output_df.loc[igalaxy,'BaryMP-factor']=barymp_fac
        output_df.loc[igalaxy,'BaryMP-radius']=barymp_fac*r200_eff
        output_df.loc[igalaxy,'BaryMP-mstar']=barymp_mstar
        output_df.loc[igalaxy,'BaryMP-mgas_ism']=barymp_mgas_ism
        output_df.loc[igalaxy,'BaryMP-mgas_sph']=barymp_mgas_sph
        output_df.loc[igalaxy,'BaryMP-SFR']=barymp_sfr
        output_df.loc[igalaxy,'BaryMP-Z']=barymp_Z

        
        if icen:
            logging.info(f'Done with galaxy {igalaxy+1} of {numgal_subvolume} for this subvolume - CENTRAL [runtime = {time.time()-t0:.2f}s]')
        else:
            logging.info(f'Done with galaxy {igalaxy+1} of {numgal_subvolume} for this subvolume - SATELLITE [runtime = {time.time()-t0:.2f}s]')

        logging.info(f'')

    logging.info(f'{np.sum(success):.0f} of {len(success):.0f} galaxies were successfully processed ({np.nanmean(success)*100:.1f}%) [runtime = {time.time()-t0:.2f}s]')

    output_df.to_hdf(output_fname,key='Subhalo')
    print(output_df)

def analyse_gasflow(path,snapidx,nvol,ivol,snapidx_delta=1,snip=True):

    ivol=int(ivol)
    ivol=str(ivol).zfill(3)
    ix,iy,iz=ivol_idx(ivol,nvol=nvol)

    t0=time.time()
    logfile=f'logs/gasflow/gasflow_snapidx_{snapidx}_delta_{str(snapidx_delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{ivol}.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO)

    output_fname=f'catalogues/gasflow/gasflow_snapidx_{snapidx}_delta_{str(snapidx_delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5'
    if os.path.exists(output_fname):
        os.remove(output_fname)

    #background data for calc
    redshift_table=pd.read_hdf('../snapshot_redshifts.hdf5',key='snapshots')
    snapidx2=snapidx
    snapidx1=snapidx2-snapidx_delta

    if snip:
        snapidx2_tag=redshift_table.loc[redshift_table['snapshotidx']==snapidx2,'tag'].values[0]
        snapidx1_tag=redshift_table.loc[redshift_table['snapshotidx']==snapidx1,'tag'].values[0]
        snapidx1_particledatapath=f'{path}/particledata_{snapidx1_tag}/eagle_subfind_snip_particles_{snapidx1_tag[5:]}.0.hdf5'
        snapidx2_particledatapath=f'{path}/particledata_{snapidx2_tag}/eagle_subfind_snip_particles_{snapidx2_tag[5:]}.0.hdf5'

    else:
        snapidx2_particledatafolder=[snapdir for snapdir in os.listdir(path) if f'snapshot_{str(snapidx2).zfill(3)}' in snapdir][0]
        snapidx2_tag=snapidx2_particledatafolder.split('snapshot_')[-1]
        snapidx2_particledatapath=f'{path}snapshot_{snapidx2_tag}/snap_{snapidx2_tag}.0.hdf5'
        snapidx1_particledatafolder=[snapdir for snapdir in os.listdir(path) if f'snapshot_{str(snapidx1).zfill(3)}' in snapdir][0]
        snapidx1_tag=snapidx1_particledatafolder.split('snapshot_')[-1]
        snapidx1_particledatapath=f'{path}snapshot_{snapidx1_tag}/snap_{snapidx1_tag}.0.hdf5'

    cosmology=FlatLambdaCDM(H0=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['HubbleParam']*100,
                            Om0=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['Omega0'])
    redshift_snap2=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['Redshift']
    redshift_snap1=h5py.File(snapidx1_particledatapath,'r')['Header'].attrs['Redshift']
    scalefac_snap2=(1/(1+redshift_snap2))
    scalefac_snap1=(1/(1+redshift_snap1))

    rhocrit=cosmology.critical_density(redshift_snap2)
    rhocrit=rhocrit.to(units.Msun/units.Mpc**3)
    rhocrit=rhocrit.value

    snapidx1_z=h5py.File(snapidx1_particledatapath,'r')['Header'].attrs['Redshift'];snapidx1_lt=cosmology.lookback_time(snapidx1_z)
    snapidx2_z=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['Redshift'];snapidx2_lt=cosmology.lookback_time(snapidx2_z)

    boxsize=h5py.File(snapidx2_particledatapath,'r')['Header'].attrs['BoxSize']

    #read data
    snapidx1_eagledata = EagleSnapshot(snapidx1_particledatapath)
    snapidx2_eagledata = EagleSnapshot(snapidx2_particledatapath)

    subvol_edgelength=boxsize/nvol
    buffer=2

    xmin=ix*subvol_edgelength;xmax=(ix+1)*subvol_edgelength
    ymin=iy*subvol_edgelength;ymax=(iy+1)*subvol_edgelength
    zmin=iz*subvol_edgelength;zmax=(iz+1)*subvol_edgelength

    logging.info(f'Considering region: (1/{nvol**3} of full box) [runtime = {time.time()-t0:.2f}s]')
    logging.info(f'ix: {ix} - x in [{xmin},{xmax}]')
    logging.info(f'iy: {iy} - y in [{ymin},{ymax}]')
    logging.info(f'iz: {iz} - z in [{zmin},{zmax}]')

    snapidx1_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)
    snapidx2_eagledata.select_region(xmin-buffer, xmax+buffer, ymin-buffer, ymax+buffer, zmin-buffer, zmax+buffer)

    logging.info(f'Initialising particle data with IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1=pd.DataFrame(snapidx1_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap1.loc[:,"ParticleTypes"]=0
    particledata_snap2=pd.DataFrame(snapidx2_eagledata.read_dataset(0,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap2.loc[:,"ParticleTypes"]=0
    particledata_snap1_star=pd.DataFrame(snapidx1_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap1_star.loc[:,"ParticleTypes"]=4;particledata_snap1_star.loc[:,"Temperature"]=-1.;particledata_snap1_star.loc[:,"Density"]=10**10
    particledata_snap2_star=pd.DataFrame(snapidx2_eagledata.read_dataset(4,'ParticleIDs'),columns=['ParticleIDs']);particledata_snap2_star.loc[:,"ParticleTypes"]=4;particledata_snap2_star.loc[:,"Temperature"]=-1.;particledata_snap2_star.loc[:,"Density"]=10**10

    logging.info(f'Reading gas datasets [runtime = {time.time()-t0:.2f}s]')
    
    gas_fields=['Coordinates','Mass','Density','Temperature']

    for dset in gas_fields:
        dset_snap1=snapidx1_eagledata.read_dataset(0,dset)
        dset_snap2=snapidx2_eagledata.read_dataset(0,dset)
        if dset_snap2.shape[-1]==3:
                particledata_snap1.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap1
                particledata_snap2.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap2
        else:
            if dset=='Mass':
                particledata_snap1[dset]=dset_snap1*10**10/h
                particledata_snap2[dset]=dset_snap2*10**10/h
            else:
                particledata_snap1[dset]=dset_snap1
                particledata_snap2[dset]=dset_snap2

    logging.info(f'Reading star datasets [runtime = {time.time()-t0:.2f}s]')
    star_fields=['Coordinates','Mass']
    for dset in star_fields:
        dset_snap1=snapidx1_eagledata.read_dataset(4,dset)
        dset_snap2=snapidx2_eagledata.read_dataset(4,dset)
        if dset_snap2.shape[-1]==3:
            particledata_snap1_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap1
            particledata_snap2_star.loc[:,[f'{dset}_x',f'{dset}_y',f'{dset}_z']]=dset_snap2
        else:
            if dset=='Mass':
                particledata_snap1_star[dset]=dset_snap1*10**10/h
                particledata_snap2_star[dset]=dset_snap2*10**10/h
            else:
                particledata_snap1_star[dset]=dset_snap1
                particledata_snap2_star[dset]=dset_snap2

    logging.info(f'Done reading datasets - concatenating gas and star data [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1=particledata_snap1.append(particledata_snap1_star,ignore_index=True)
    particledata_snap2=particledata_snap2.append(particledata_snap2_star,ignore_index=True)

    logging.info(f'Sorting by IDs [runtime = {time.time()-t0:.2f}s]')
    particledata_snap1.sort_values(by="ParticleIDs",inplace=True);particledata_snap1.reset_index(inplace=True,drop=True)
    particledata_snap2.sort_values(by="ParticleIDs",inplace=True);particledata_snap2.reset_index(inplace=True,drop=True)
    size1=np.sum(particledata_snap1.memory_usage().values)/10**9;size2=np.sum(particledata_snap2.memory_usage().values)/10**9
    
    logging.info(f'Particle data snap 1 memory usage: {size1:.2f} GB')
    logging.info(f'Particle data snap 2 memory usage: {size2:.2f} GB')

    #particle KD trees
    gen1=True;gen2=True
    kdtree_snap1_fname=f'catalogues/kdtrees/kdtree_snapidx_{str(snapidx1).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.dat'
    kdtree_snap2_fname=f'catalogues/kdtrees/kdtree_snapidx_{str(snapidx2).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.dat'

    #particle KD trees
    logging.info(f'Obtaining KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')    
    if os.path.exists(kdtree_snap1_fname):
        try:
            logging.info(f'Loading KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
            kdtree_snap1_periodic=open_pickle(path=kdtree_snap1_fname)
            gen1=False
        except:
            pass
    if gen1:
        logging.info(f'Generating KDTree for snap 1 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap1_periodic=cKDTree(np.column_stack([particledata_snap1[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        dump_pickle(data=kdtree_snap1_periodic,path=kdtree_snap1_fname)
    
    logging.info(f'Obtaining KDTree for snap 2 [runtime = {time.time()-t0:.2f}s]')    
    if os.path.exists(kdtree_snap2_fname):
        try:
            logging.info(f'Loading KDTree for snap 2 [runtime = {time.time()-t0:.2f}s]')
            kdtree_snap2_periodic=open_pickle(path=kdtree_snap2_fname)
            gen2=False
        except:
            pass
    if gen2:
        logging.info(f'Generating KDTree for snap 2 [runtime = {time.time()-t0:.2f}s]')
        kdtree_snap2_periodic=cKDTree(np.column_stack([particledata_snap2[f'Coordinates_{x}'] for x in 'xyz']),boxsize=boxsize)
        dump_pickle(data=kdtree_snap2_periodic,path=kdtree_snap2_fname)

    #load catalogues into dataframes
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')

    catalogue_subhalo_extended_ivol_fname=f'catalogues/subhalo/subhalo_snapidx_{snapidx2}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5'
    catalogue_subhalo_extended_ivol=pd.read_hdf(catalogue_subhalo_extended_ivol_fname,key='Subhalo')

    detailed_fields=list(catalogue_subhalo_extended_ivol)
    if snip:
        detailed_fields.remove('nodeIndex')
    else:
        detailed_fields.remove('GalaxyID')
    detailed_fields.remove('GroupNumber');detailed_fields.remove('SubGroupNumber')

    #select relevant subhaloes
    snap_mask=catalogue_subhalo[f'snapshotidx']==snapidx2
    snap_mass_mask=catalogue_subhalo[f'snapshotidx']>=0
    snap_com_mask_1=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']>=ixmin for x,ixmin in zip('xyz',[xmin,ymin,zmin])])
    snap_com_mask_2=np.logical_and.reduce([catalogue_subhalo[f'CentreOfPotential_{x}']<=ixmax for x,ixmax in zip('xyz',[xmax,ymax,zmax])])
    snap_com_mask=np.logical_and.reduce([snap_com_mask_1,snap_com_mask_2,snap_mask,snap_mass_mask])
    numgal_subvolume=np.sum(snap_com_mask);numgal_total=np.sum(np.logical_and(snap_mask,snap_mass_mask))
    logging.info(f'Will use  {numgal_subvolume} of {numgal_total} valid galaxies from box [runtime = {time.time()-t0:.2f}s]')

    subhalo_subset=catalogue_subhalo.loc[snap_com_mask,:].copy()
    subhalo_subset.reset_index(drop=True,inplace=True)

    #initialise output
    if snip:
        initfields=['nodeIndex','GroupNumber','SubGroupNumber']
    else:
        initfields=['GalaxyID','GroupNumber','SubGroupNumber']

    gasflow_df=subhalo_subset.loc[:,initfields].copy()
    gasflow_df.reset_index(drop=True,inplace=True)

    gasflow_df.loc[:,'nmerger_minor']=np.nan
    gasflow_df.loc[:,'nmerger_major']=np.nan
    gasflow_df.loc[:,'dt']=snapidx1_lt.value-snapidx2_lt.value

    gasflow_df.loc[:,'inflow-sph_30kpc']=np.nan
    gasflow_df.loc[:,'inflow-ism_30kpc']=np.nan
    gasflow_df.loc[:,'outflow-sph_30kpc']=np.nan
    gasflow_df.loc[:,'outflow-ism_30kpc']=np.nan

    gasflow_df.loc[:,'inflow-sph_barymp']=np.nan
    gasflow_df.loc[:,'inflow-ism_barymp']=np.nan
    gasflow_df.loc[:,'outflow-sph_barymp']=np.nan
    gasflow_df.loc[:,'outflow-ism_barymp']=np.nan
    gasflow_df.loc[:,detailed_fields]=np.nan
    
    gasflow_df.loc[:,'matchrate-halo_in']=np.nan
    gasflow_df.loc[:,'matchrate-halo_out']=np.nan
    gasflow_df.loc[:,'matchrate-ism_barymp_in']=np.nan
    gasflow_df.loc[:,'matchrate-ism_barymp_out']=np.nan

    r200_facs=[0.1,0.15,0.2,0.25,0.5,0.75,1]
    for fac in r200_facs:
        gasflow_df.loc[:,f'inflow-{fac:.3f}r200']=np.nan
        gasflow_df.loc[:,f'outflow-{fac:.3f}r200']=np.nan
    
    #Main halo loop
    for igalaxy_snap2,galaxy_snap2 in subhalo_subset.iterrows():
        if snip:
            idx_field='nodeIndex'
        else:
            idx_field='GalaxyID'
        
        idx=galaxy_snap2[idx_field]
        subgroupnumber=galaxy_snap2['SubGroupNumber']
        nminor,nmajor,progidx=find_progidx(catalogue_subhalo,idx=idx,snap=snapidx2,snapidx_delta=snapidx_delta,snip=snip)
        gasflow_df.loc[igalaxy_snap2,'nmerger_minor']=nminor
        gasflow_df.loc[igalaxy_snap2,'nmerger_major']=nmajor

        if subgroupnumber==0:
            icen=True
            r200_eff=galaxy_snap2['Group_R_Crit200']
        else:
            icen=False
            r200_eff=r200(m200=galaxy_snap2['Mass'],rhocrit=rhocrit)

        logging.info(f'Processing galaxy {igalaxy_snap2+1} of {numgal_subvolume} for this subvolume - SGN {subgroupnumber} [runtime = {time.time()-t0:.2f}s]')
        logging.info(f"Using effective radius {r200_eff} cMpc for subhalo mass {galaxy_snap2['Mass']:.1e} Msun")
        print(f"Processing galaxy {igalaxy_snap2+1} of {numgal_subvolume} for this subvolume - SGN {subgroupnumber} [runtime = {time.time()-t0:.2f}s]")
        print(f"Using effective radius {r200_eff} cMpc for subhalo mass {galaxy_snap2['Mass']:.1e} Msun")

        #ensuring there has been a progenitor found
        if np.sum(progidx==catalogue_subhalo[idx_field]):
            pass           
        else:
            logging.info(f'Skipping galaxy {igalaxy_snap2+1} of {numgal_subvolume} - could not find progenitor')
            continue

        galaxy_snap1=catalogue_subhalo.loc[progidx==catalogue_subhalo[idx_field],:]

        com_snap2=[galaxy_snap2[f"CentreOfPotential_{x}"] for x in 'xyz']
        com_snap1=[galaxy_snap1[f"CentreOfPotential_{x}"].values[0] for x in 'xyz']

        galaxy_snap2_detailed=catalogue_subhalo_extended_ivol.loc[igalaxy_snap2,detailed_fields]

        #select particles in halo-size sphere
        candidate_radius=r200_eff
        part_idx_candidates_snap2=kdtree_snap2_periodic.query_ball_point(com_snap2,candidate_radius)
        part_idx_candidates_snap1=kdtree_snap1_periodic.query_ball_point(com_snap1,candidate_radius)

        logging.info(f"SGN: {subgroupnumber}, mass {galaxy_snap2['Mass']:.1e} - using candidate radius of {candidate_radius} cMpc")

        part_IDs_candidates_all=np.unique(np.concatenate([particledata_snap2.loc[part_idx_candidates_snap2,"ParticleIDs"].values,
                                                          particledata_snap1.loc[part_idx_candidates_snap1,"ParticleIDs"].values])).astype(np.int64)
        part_idx_candidates_snap1=particledata_snap1['ParticleIDs'].searchsorted(part_IDs_candidates_all)
        part_idx_candidates_snap2=particledata_snap2['ParticleIDs'].searchsorted(part_IDs_candidates_all)
        part_data_candidates_snap1=particledata_snap1.loc[part_idx_candidates_snap1,:]
        part_data_candidates_snap2=particledata_snap2.loc[part_idx_candidates_snap2,:]
        
        #needed if using subfind particle data
        matches_snap1=part_IDs_candidates_all==part_data_candidates_snap1.loc[:,"ParticleIDs"].values
        matches_snap2=part_IDs_candidates_all==part_data_candidates_snap2.loc[:,"ParticleIDs"].values
        print()
        nomatch_snap1=np.logical_not(matches_snap1)
        nomatch_snap2=np.logical_not(matches_snap2)
        
        matches=np.logical_and(matches_snap1,matches_snap2)
        print(np.nansum(matches)/len(matches))
        
        if not snip:
            part_data_candidates_snap1.reset_index(drop=True,inplace=True)
            part_data_candidates_snap2.reset_index(drop=True,inplace=True)

        else:
            mass_ave=np.nanmean(part_data_candidates_snap2.loc[:,"Mass"])
            part_data_candidates_snap1.loc[nomatch_snap1,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs']]=np.array([np.nan,np.nan,np.nan,np.nan,np.nan,mass_ave,-1])
            part_data_candidates_snap2.loc[nomatch_snap2,['Coordinates_x','Coordinates_y','Coordinates_z','Temperature','Density','Mass','ParticleIDs']]=np.array([np.nan,np.nan,np.nan,np.nan,np.nan,mass_ave,-1])

        #adding rcom and vrad
        part_data_candidates_snap2.loc[:,"r_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates_snap2[f'Coordinates_{x}']-com_snap2[ix] for ix,x in enumerate('xyz')])),axis=1))#Mpc
        # part_data_candidates_snap2.loc[:,[f"runit_{x}rel" for x in 'xyz']]=np.column_stack([(part_data_candidates_snap2[f'Coordinates_{x}']-com_snap2[ix])/part_data_candidates_snap2[f'r_com'] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap2.loc[:,[f"Velocity_{x}rel" for x in 'xyz']]=np.column_stack([part_data_candidates_snap2[f'Velocity_{x}']-vcom_snap2[ix] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap2["vrad_inst"]=np.sum(np.multiply(np.column_stack([part_data_candidates_snap2[f"Velocity_{x}rel"] for x in 'xyz']),np.column_stack([part_data_candidates_snap2[f"runit_{x}rel"] for x in 'xyz'])),axis=1)
        part_data_candidates_snap1.loc[:,"r_com"]=np.sqrt(np.sum(np.square(np.column_stack([part_data_candidates_snap1[f'Coordinates_{x}']-com_snap1[ix] for ix,x in enumerate('xyz')])),axis=1))#Mpc
        # part_data_candidates_snap1.loc[:,[f"runit_{x}rel" for x in 'xyz']]=np.column_stack([(part_data_candidates_snap1[f'Coordinates_{x}']-com_snap1[ix])/part_data_candidates_snap1[f'r_com'] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap1.loc[:,[f"Velocity_{x}rel" for x in 'xyz']]=np.column_stack([part_data_candidates_snap1[f'Velocity_{x}']-vcom_snap1[ix] for ix,x in enumerate('xyz')])
        # part_data_candidates_snap1["vrad_inst"]=np.sum(np.multiply(np.column_stack([part_data_candidates_snap1[f"Velocity_{x}rel"] for x in 'xyz']),np.column_stack([part_data_candidates_snap1[f"runit_{x}rel"] for x in 'xyz'])),axis=1)

        #masks snap 1
        gas_snap1=part_data_candidates_snap1["ParticleTypes"].values==0
        tempreq_snap1=part_data_candidates_snap1["Temperature"].values<=5*10**4
        inside_30kpc_snap1=part_data_candidates_snap1.loc[:,"r_com"].values<0.03
        insidebarymp_snap1=part_data_candidates_snap1.loc[:,"r_com"].values<galaxy_snap2_detailed['BaryMP-radius']
        

        #masks snap 2
        tempreq_snap2=part_data_candidates_snap2["Temperature"].values<=5*10**4
        inside_30kpc_snap2=part_data_candidates_snap2.loc[:,"r_com"].values<0.03
        insidebarymp_snap2=part_data_candidates_snap2.loc[:,"r_com"].values<galaxy_snap2_detailed['BaryMP-radius']

        ism_30kpc_snap1=np.logical_and.reduce([tempreq_snap1,inside_30kpc_snap1])
        ism_30kpc_snap2=np.logical_and.reduce([tempreq_snap2,inside_30kpc_snap2])
        sph_30kpc_snap1=np.logical_and.reduce([inside_30kpc_snap1])
        sph_30kpc_snap2=np.logical_and.reduce([inside_30kpc_snap2])

        ism_barymp_snap1=np.logical_and.reduce([tempreq_snap1,insidebarymp_snap1])
        ism_barymp_snap2=np.logical_and.reduce([tempreq_snap2,insidebarymp_snap2])
        sph_barymp_snap1=np.logical_and.reduce([insidebarymp_snap1])
        sph_barymp_snap2=np.logical_and.reduce([insidebarymp_snap2])

        # particles in/out
        ism_partidx_out_30kpc=np.logical_and.reduce([ism_30kpc_snap1,np.logical_not(ism_30kpc_snap2),gas_snap1])
        ism_partidx_in_30kpc=np.logical_and.reduce([np.logical_not(ism_30kpc_snap1),ism_30kpc_snap2,gas_snap1])
        
        print('Part of ISM at snap 1, not at snap 2')
        print(np.nansum(ism_30kpc_snap1),np.nansum(np.logical_not(ism_30kpc_snap2)))
        
        sph_partidx_out_30kpc=np.logical_and.reduce([sph_30kpc_snap1,np.logical_not(sph_30kpc_snap2),gas_snap1])
        sph_partidx_in_30kpc=np.logical_and.reduce([np.logical_not(sph_30kpc_snap1),sph_30kpc_snap2,gas_snap1])
        
        gasflow_df.loc[igalaxy_snap2,'inflow-ism_30kpc']=np.nansum(part_data_candidates_snap2.loc[ism_partidx_in_30kpc,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'inflow-sph_30kpc']=np.nansum(part_data_candidates_snap2.loc[sph_partidx_in_30kpc,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'outflow-ism_30kpc']=np.nansum(part_data_candidates_snap2.loc[ism_partidx_out_30kpc,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'outflow-sph_30kpc']=np.nansum(part_data_candidates_snap2.loc[sph_partidx_out_30kpc,'Mass'])

        ism_partidx_out_barymp=np.logical_and.reduce([ism_barymp_snap1,np.logical_not(ism_barymp_snap2),gas_snap1])
        ism_partidx_in_barymp=np.logical_and.reduce([np.logical_not(ism_barymp_snap1),ism_barymp_snap2,gas_snap1])
        sph_partidx_out_barymp=np.logical_and.reduce([sph_barymp_snap1,np.logical_not(sph_barymp_snap2),gas_snap1])
        sph_partidx_in_barymp=np.logical_and.reduce([np.logical_not(sph_barymp_snap1),sph_barymp_snap2,gas_snap1])

        gasflow_df.loc[igalaxy_snap2,'inflow-ism_barymp']=np.nansum(part_data_candidates_snap2.loc[ism_partidx_in_barymp,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'inflow-sph_barymp']=np.nansum(part_data_candidates_snap2.loc[sph_partidx_in_barymp,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'outflow-ism_barymp']=np.nansum(part_data_candidates_snap2.loc[ism_partidx_out_barymp,'Mass'])
        gasflow_df.loc[igalaxy_snap2,'outflow-sph_barymp']=np.nansum(part_data_candidates_snap2.loc[sph_partidx_out_barymp,'Mass'])
        
        npart_in_barymp=np.sum(ism_partidx_in_barymp)
        npart_out_barymp=np.sum(ism_partidx_out_barymp)
        if npart_in_barymp:
            gasflow_df.loc[igalaxy_snap2,'matchrate-ism_barymp_in']=1-np.nanmean(part_data_candidates_snap1.loc[ism_partidx_in_barymp,'ParticleIDs']==-1)
        else:
            gasflow_df.loc[igalaxy_snap2,'matchrate-ism_barymp_in']=2

        if npart_out_barymp:
            gasflow_df.loc[igalaxy_snap2,'matchrate-ism_barymp_out']=1-np.nanmean(part_data_candidates_snap2.loc[ism_partidx_out_barymp,'ParticleIDs']==-1)
        else:
            gasflow_df.loc[igalaxy_snap2,'matchrate-ism_barymp_out']=2

        gasflow_df.loc[igalaxy_snap2,detailed_fields]=np.array([galaxy_snap2_detailed[detailed_field] for detailed_field in detailed_fields])

        #halo def
        for fac in r200_facs:
            halo_snap1=np.logical_and.reduce([part_data_candidates_snap1["r_com"].values<fac*candidate_radius])
            halo_snap2=np.logical_and.reduce([part_data_candidates_snap2["r_com"].values<fac*candidate_radius])
            
            #new halo particles
            halo_partidx_in=np.logical_and.reduce([np.logical_not(halo_snap1),halo_snap2,gas_snap1])
            #removed halo particles
            halo_partidx_out=np.logical_and.reduce([halo_snap1,np.logical_not(halo_snap2),gas_snap1]) 
            
            #sum masses
            gasflow_df.loc[igalaxy_snap2,f'inflow-{fac:.3f}r200']=np.sum(part_data_candidates_snap2.loc[halo_partidx_in,'Mass'])
            gasflow_df.loc[igalaxy_snap2,f'outflow-{fac:.3f}r200']=np.sum(part_data_candidates_snap2.loc[halo_partidx_out,'Mass'])

            if fac==1:
                if np.sum(halo_partidx_in):
                    infrac=1-np.nanmean(part_data_candidates_snap1.loc[halo_partidx_in,'ParticleIDs']==-1)
                else:
                    infrac=1
                if np.sum(halo_partidx_out):
                    outfrac=1-np.nanmean(part_data_candidates_snap2.loc[halo_partidx_out,'ParticleIDs']==-1)
                else:
                    outfrac=1

                gasflow_df.loc[igalaxy_snap2,'matchrate-halo_in']=infrac
                gasflow_df.loc[igalaxy_snap2,'matchrate-halo_out']=outfrac
 
        if icen:
            logging.info(f'Done with galaxy {igalaxy_snap2+1} of {numgal_subvolume} for this subvolume - CENTRAL [runtime = {time.time()-t0:.2f}s]')
        else:
            logging.info(f'Done with galaxy {igalaxy_snap2+1} of {numgal_subvolume} for this subvolume - SATELLITE [runtime = {time.time()-t0:.2f}s]')

        logging.info(f'')

    gasflow_df['netflow-ism_30kpc']=gasflow_df['inflow-ism_30kpc']-gasflow_df['outflow-ism_30kpc']
    gasflow_df['netflow-sph_30kpc']=gasflow_df['inflow-sph_30kpc']-gasflow_df['outflow-sph_30kpc']
    gasflow_df['netflow-ism_barymp']=gasflow_df['inflow-ism_barymp']-gasflow_df['outflow-ism_barymp']
    gasflow_df['netflow-sph_barymp']=gasflow_df['inflow-sph_barymp']-gasflow_df['outflow-sph_barymp']
    for fac in r200_facs:
        gasflow_df[f'netflow-{fac:.3f}r200']=gasflow_df[f'inflow-{fac:.3f}r200']-gasflow_df[f'outflow-{fac:.3f}r200']

    logging.info(f'{gasflow_df.shape[0]:.0f} galaxies were successfully processed [runtime = {time.time()-t0:.2f}s]')
    print(gasflow_df.loc[:,['BaryMP-mstar','SubGroupNumber','BaryMP-mgas_ism','inflow-ism_barymp','outflow-ism_barymp','matchrate-ism_barymp_in','matchrate-ism_barymp_out']])
    gasflow_df.to_hdf(output_fname,key='Flux')

def combine_catalogues(mcut,snapidxs,nvol,snapidx_delta=1,snip=True):
    
    outname=f'catalogues/catalogue_gasflow_nvol_{str(nvol).zfill(2)}_mcut_{str(mcut).zfill(2)}_delta_{str(snapidx_delta).zfill(2)}.hdf5'
    catalogue_subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5',key='Subhalo')
    catalogue_subhalo=catalogue_subhalo.loc[np.logical_and.reduce([np.logical_or.reduce([catalogue_subhalo['snapshotidx']==snapidx for snapidx in snapidxs])]),:]

    accfile_data_vols=[]

    ifile=0
    isnap=0
    logfile=f'logs/combine_cats_delta{str(snapidx_delta).zfill(2)}.log'
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO)

    for snapidx in snapidxs:
        ivol=0
        for ivol in range(nvol**3):
            logging.info(f'Loading volume {ivol+1}/{nvol**3} for snap {snapidx} (delta {snapidx_delta})')
            
            print(f'Loading volume {ivol+1}/{nvol**3} for snap {snapidx} (delta {snapidx_delta})')
            try:
                accfile_data_file=pd.read_hdf(f'catalogues/gasflow/gasflow_snapidx_{snapidx}_delta_{str(snapidx_delta).zfill(3)}_n_{str(nvol).zfill(2)}_volume_{str(ivol).zfill(3)}.hdf5',key='Flux')
                if ifile==0:
                    accfields=list(accfile_data_file)
                # print(accfile_data_file.shape[0],' subhaloes in subvolume')
            except:
                print(f'Could not load volume {ivol}')
                continue
            accfile_data_isnap_ivol=accfile_data_file.loc[:,:]
            accfile_data_vols.append(accfile_data_isnap_ivol)
            
            ifile+=1
            ivol+=1
        isnap+=1

    if not snip:
        ID_fields=['GalaxyID']

    else:
        ID_fields=['nodeIndex']


    accfile_data=pd.concat(accfile_data_vols,ignore_index=True)
    catalogue_subhalo=catalogue_subhalo.sort_values(by=ID_fields[0])
    accfile_data=accfile_data.sort_values(by=ID_fields[0])
    catalogue_subhalo.reset_index();accfile_data.reset_index()
    accfile_data.loc[:,ID_fields[0]+'-acc']=accfile_data[ID_fields[0]].values
    accfile_fields=list(accfile_data)
    print(accfile_fields)
    for ID_field in ID_fields:
        print(ID_field)
        accfile_fields.remove(ID_field)

    accretion_nodeidx=accfile_data[ID_fields[0]].values
    subcat_nodeidx=catalogue_subhalo[ID_fields[0]].values

    accretion_idxinsubcat=np.searchsorted(subcat_nodeidx,accretion_nodeidx)
    mask=np.zeros(len(catalogue_subhalo[ID_field]))
    mask[accretion_idxinsubcat]=True;mask=mask.astype(bool)
    catalogue_subhalo.loc[:,accfile_fields]=np.nan
    for field in accfile_fields:
        catalogue_subhalo.loc[mask,field]=accfile_data[field].values

    valid=catalogue_subhalo[ID_field+'-acc']>0
    catalogue_subhalo=catalogue_subhalo.loc[valid,:]

    if os.path.exists(outname):
        os.remove(outname)
    
    catalogue_subhalo.to_hdf(outname,key='Subhalo')
    print(np.column_stack([catalogue_subhalo[::1000][ID_field+'-acc'],catalogue_subhalo[::1000][ID_field]]))

#lower level

def ivol_gen(ix,iy,iz,nvol):
    ivol=ix*nvol**2+iy*nvol+iz
    ivol_str=str(ivol).zfill(3)
    return ivol_str

def ivol_idx(ivol,nvol):
    if type(ivol)==str:
        ivol=int(ivol)
    ix=int(np.floor(ivol/nvol**2))
    iz=int(ivol%nvol)
    iy=int((ivol-ix*nvol**2-iz)/nvol)
    return (ix,iy,iz)

def tfloor_eagle(nh,norm=17235.4775202):
    T=np.zeros(np.shape(nh))+8000
    dense=np.where(nh>=10**-1)
    T[dense]=norm*nh[dense]**(1/3)

    return T

def r200(m200,rhocrit):
    r200_cubed=3*m200/(800*np.pi*rhocrit)
    return r200_cubed**(1/3)
    
def find_progidx(catalogue_subhalo,idx,snap,snapidx_delta,snip=True,return_all=False):
    if snip:
        nodeidx_depth=idx
        nodeidx_depths=[idx]
        nminor=0;nmajor=0
        for idepth in range(snapidx_delta):
            matchingnode=nodeidx_depth==catalogue_subhalo['nodeIndex'].values
            if np.sum(matchingnode)==1:
                #find where nodeidx_depth is the main descendent
                progenitor_match=catalogue_subhalo.loc[:,'descendantIndex']==nodeidx_depth
                if np.sum(progenitor_match):
                    progenitors=catalogue_subhalo.loc[progenitor_match,:]
                    progenitors_mass=progenitors['Mass']
                    maxmass=np.nanmax(progenitors_mass)
                    progenitors_massratio=np.abs(np.log10(progenitors_mass/maxmass))
                    nminor_idepth=np.sum(progenitors_massratio>1)
                    nmajor_idepth=np.sum(progenitors_massratio<1)-1
                    maxmass_idx=np.where(progenitors['Mass'].values==maxmass)
                    nodeidx_depth=progenitors['nodeIndex'].values[maxmass_idx][0]
                    nminor+=nminor_idepth
                    nmajor+=nmajor_idepth
                else:
                    nodeidx_depth=None
                    break
            else:
                nodeidx_depth=None
                break

            nodeidx_depths.append(nodeidx_depth)
        
        if len(nodeidx_depths)==snapidx_delta+1:
            if return_all:
                return nminor,nmajor,nodeidx_depths
            else:
                return nminor,nmajor,nodeidx_depths[-1]
        else:
            return np.nan,np.nan,np.nan


    else:
        nodeidx_depth=idx
        nodeidx_depths=[idx]
        nminor=0;nmajor=0
        for idepth in range(snapidx_delta):
            matchingnode=nodeidx_depth==catalogue_subhalo['GalaxyID'].values
            if np.sum(matchingnode)==1:
                #find where nodeidx_depth is the main descendent
                progenitor_match=np.logical_and(catalogue_subhalo.loc[:,'DescendantID']==nodeidx_depth,
                                                np.logical_not(catalogue_subhalo.loc[:,'snapshotidx']==snap))
                if np.sum(progenitor_match):
                    progenitors=catalogue_subhalo.loc[progenitor_match,:]
                    print(progenitors)
                    progenitors_mass=progenitors['Mass']
                    maxmass=np.nanmax(progenitors_mass)
                    progenitors_massratio=np.abs(np.log10(progenitors_mass/maxmass))
                    nminor_idepth=np.sum(progenitors_massratio>1)
                    nmajor_idepth=np.sum(progenitors_massratio<1)-1
                    maxmass_idx=np.where(progenitors['Mass'].values==maxmass)
                    nodeidx_depth=progenitors['GalaxyID'].values[maxmass_idx][0]
                    nminor+=nminor_idepth
                    nmajor+=nmajor_idepth
                else:
                    nodeidx_depth=None
                    break
            else:
                nodeidx_depth=None
                break
            nodeidx_depths.append(nodeidx_depth)

        if len(nodeidx_depths)==snapidx_delta+1:
            return nminor,nmajor,nodeidx_depths[-1]
        else:
            return np.nan,np.nan,np.nan


def BaryMP(x,y,eps=0.01,grad=1):
	"""
	Find the radius for a galaxy from the BaryMP method
	x = r/r_200
	y = cumulative baryonic mass profile
	eps = epsilon, if data 
	"""
	dydx = np.diff(y)/np.diff(x)
	
	maxarg = np.argwhere(dydx==np.max(dydx))[0][0] # Find where the gradient peaks
	xind = np.argwhere(dydx[maxarg:]<=grad)[0][0] + maxarg # The index where the gradient reaches 1
	
	x2fit_new, y2fit_new = x[xind:], y[xind:] # Should read as, e.g., "x to fit".
	x2fit, y2fit = np.array([]), np.array([]) # Gets the while-loop going
	
	while len(y2fit)!=len(y2fit_new):
		x2fit, y2fit = np.array(x2fit_new), np.array(y2fit_new)
		p = np.polyfit(x2fit, y2fit, 1)
		yfit = p[0]*x2fit + p[1]
		chi = abs(yfit-y2fit) # Separation in the y-direction for the fit from the data
		chif = (chi<eps) # Filter for what chi-values are acceptable
		x2fit_new, y2fit_new = x2fit[chif], y2fit[chif]
	
	r_bmp = x2fit[0] # Radius from the baryonic-mass-profile technique, returned as a fraction of the virial radius!
	Nfit = len(x2fit) # Number of points on the profile fitted to in the end

	return r_bmp, Nfit

def open_pickle(path):
    """

    open_pickle : function
	----------------------

    Open a (binary) pickle file at the specified path, close file, return result.

	Parameters
	----------
    path : str
        Path to the desired pickle file. 


    Returns
	----------
    output : data structure of desired pickled object

    """

    with open(path,'rb') as picklefile:
        pickledata=pickle.load(picklefile)
        picklefile.close()

    return pickledata

def dump_pickle(data,path):
    """

    dump_pickle : function
	----------------------

    Dump data to a (binary) pickle file at the specified path, close file.

	Parameters
	----------
    data : any type
        The object to pickle. 

    path : str
        Path to the desired pickle file. 


    Returns
	----------
    None

    Creates a file containing the pickled object at path. 

    """

    with open(path,'wb') as picklefile:
        pickle.dump(data,picklefile,protocol=4)
        picklefile.close()
    return data



# orbweaver
def ReadOrbitData(filenamelist,iFileno=False,apsispoints=True,crossingpoints=True,endpoints=True,desiredfields=[]):

	"""
	Function to read in the data from the .orbweaver.orbitdata.hdf files
	Parameters
	----------
	filenamelist : str
		The file containing the basenames for the orbweaver catalogue (can use the file generated from the creation of the (preprocessed) orbit catalogue)
	iFileno : bool, optional
		If True, the file number where the each of the entries came from is to be outputted.
	apsispoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the apsispoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	crossingpoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the crossingpoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	endpoints : bool, optional
		If False, a boolean selection is done on the data to be loaded so the endpoints are not loaded in. It is done so it also reduces the memory used. But this means the reading takes significatly longer.
	desiredfields : list, optional
		A list containing the desired fields to put returned, please see the FieldsDescriptions.md for the list of fields availible. If not supplied then all fields are returned
	Returns
	-------
	orbitdata : dict
		Dictionary of the fields to be outputted, where each field is a ndarray .
	"""
	start =  time.time()

	#See if any of the desired datasets are false
	createselection=False
	if((apsispoints==False) | (crossingpoints==False) | (endpoints==False)):
		createselection  = True

	#First see if the file exits
	if(os.path.isfile(filenamelist)==False):
		raise IOError("The filelist",filenamelist,"does not exist")

	filelist = open(filenamelist,"r")

	#Try and read the first line as int
	try:
		numfiles = int(filelist.readline())
	except ValueError:
		raise IOError("The first line of the filelist (which says the number of files), cannot be interpreted as a integer")

	if(iFileno): fileno = np.zeros(numfiles,dtype=np.int32)
	numentries = np.zeros(numfiles,dtype=np.uint64)
	maxorbitIDs = np.zeros(numfiles,dtype=np.uint64)
	prevmaxorbitID = np.uint64(0)
	filenames = [""]*numfiles
	orbitdatatypes = {}
	orbitdatakeys = None
	if(createselection): selArray = [[] for i in range(numfiles)]

	#Loop through the filelist and read the header of each file
	for i in range(numfiles):

		#Extract the filename
		filename = filelist.readline().strip("\n")
		filename+=".orbweaver.orbitdata.hdf"

		if(os.path.isfile(filename)==False):
			raise IOError("The file",filename,"does not exits")

		#Open up the file
		hdffile = h5py.File(filename,"r")

		#Read the header information
		if(iFileno): fileno[i] = np.int32(hdffile.attrs["Fileno"][...])
		numentries[i] =  np.uint64(hdffile.attrs["Number_of_entries"][...])
		maxorbitIDs[i] = prevmaxorbitID
		prevmaxorbitID += np.uint64(hdffile.attrs["Max_orbitID"][...])

		#Use the entrytype dataset to find points to be extracted
		if(createselection):

			#Load in the entrytype dataset to create the selection
			ientrytype = np.asarray(hdffile["entrytype"],dtype=np.float64)

			#Create an array the size of the number of entries to load in the data
			sel = np.zeros(numentries[i],dtype=bool)

			#If want apsis points
			if(apsispoints):
				sel = (np.round(np.abs(ientrytype),1) == 99.0)

			#If crossing points are also desired
			if(crossingpoints):
				sel = ((np.round(np.abs(ientrytype),1) != 99.0) & (np.round(np.abs(ientrytype),1) != 0.0)) | sel

			#The final endpoint for the orbiting halo
			if(endpoints):
				sel = (np.round(np.abs(ientrytype),1) == 0.0) | sel

			selArray[i] = sel

			#Update the number of entries based on the selection
			numentries[i] = np.sum(sel,dtype = np.uint64)

		#If the first file then file then find the dataset names and their datatypes
		if(i==0):
			if(len(desiredfields)>0):
				orbitdatakeys = desiredfields
			else:
				orbitdatakeys = list(hdffile.keys())

			for key in orbitdatakeys:
				orbitdatatypes[key] = hdffile[key].dtype

		hdffile.close()

		#Add this filename to the filename list
		filenames[i] = filename

	#Now can initilize the array to contain the data
	totnumentries = np.sum(numentries)
	orbitdata = {key:np.zeros(totnumentries,dtype = orbitdatatypes[key]) for key in orbitdatakeys}

	if(iFileno):
		#Add a field to contain the file number
		orbitdata["Fileno"] = np.zeros(totnumentries,dtype=np.int32)

	ioffset = np.uint64(0)

	#Now read in the data
	for i in range(numfiles):

		filename=filenames[i]

		print("Reading orbitdata from",filename)

		#Open up the file
		hdffile = h5py.File(filename,"r")

		#Get the start and end index
		startindex = ioffset
		endindex = np.uint64(ioffset+numentries[i])

		#Read the datasets
		for key in orbitdatakeys:
			if(createselection):
				orbitdata[key][startindex:endindex] = np.asarray(hdffile[key][selArray[i]],dtype=orbitdatatypes[key])
			else:
				orbitdata[key][startindex:endindex] = np.asarray(hdffile[key],dtype=orbitdatatypes[key])

		if("OrbitID" in orbitdatakeys):
			#Lets offset the orbitID to make it unique across all the data
			orbitdata["OrbitID"][startindex:endindex]+= np.uint64(maxorbitIDs[i] + i)

		if(iFileno):
			#Set the fileno that this came from
			orbitdata["Fileno"][startindex:endindex]=fileno[i]

		hdffile.close()

		ioffset+=numentries[i]

	print("Done reading in",time.time()-start)

	return orbitdata
