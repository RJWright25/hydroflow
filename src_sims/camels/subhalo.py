
import os
import h5py
import numpy as np
import pandas as pd

#read gadget output
def read_subcat(path):
    group_fields=['GroupMass',
                  'Group_M_Crit200',
                  'Group_R_Crit200',
                  'GroupPos']
    

    subcat_fnames=sorted([path+fname for fname in os.listdir(path) if 'hdf5' in fname])
    subhalo_dfs=[]

    for subcat_snapnum,subcat_fname in enumerate(subcat_fnames):
        print(f'Reading file {subcat_snapnum}')
        subcat_file=h5py.File(subcat_fname,'r')

        hfac=subcat_file['Header'].attrs['HubbleParam']
        afac=subcat_file['Header'].attrs['Time']
        zval=subcat_file['Header'].attrs['Redshift']
        
        group_df=pd.DataFrame()
        subhalo_df=pd.DataFrame()

        for field in group_fields: 
            if ('Potential' in field) or ('Pos' in field):
                group_df.loc[:,[f'GroupCentreOfPotential_{x}' for x in 'xyz']]=subcat_file['Group/'+field][:]*1e-3
            else:
                group_df[field]=subcat_file['Group'][field][:]

            if ('Mass' in field) or ('_M_' in field):
                group_df[field]=group_df[field]*10**10/hfac

        subhalo_df['GroupNumber']=subcat_file['Subhalo/SubhaloGrNr'][:]
        subhalo_df['Vmax']=subcat_file['Subhalo/SubhaloVmax'][:]
        subhalo_df['VmaxRadius']=subcat_file['Subhalo/SubhaloVmaxRad'][:]*1e-3
        subhalo_df['HalfMassRad']=subcat_file['Subhalo/SubhaloHalfmassRad'][:]*1e-3
        subhalo_df['IDMostBound']=subcat_file['Subhalo/SubhaloIDMostbound'][:]
        
        subhalo_df['Mass']=np.nansum(subcat_file['Subhalo/SubhaloMassType'][:],axis=1)*10**10/hfac
        subhalo_df.loc[:,[f'MassType_{itype}' for itype in [0,1,2,3,4,5]]]=subcat_file['Subhalo/SubhaloMassType'][:]*10**10/hfac
        subhalo_df.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloPos'][:]*1e-3
        subhalo_df.loc[:,[f'CentreOfMass_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloCM'][:]*1e-3
        subhalo_df.loc[:,[f'Velocity_{x}' for x in 'xyz']]=subcat_file['Subhalo/SubhaloVel'][:]
        subhalo_df.loc[:,[f'Spin_{x}' for x in 'xyz']]=s=subcat_file['Subhalo/SubhaloSpin'][:]

        subhalo_df.loc[:,'SnapNum']=subcat_snapnum
        subhalo_df.loc[:,'Redshift']=zval

        subhalo_df=subhalo_df.loc[subhalo_df['Mass'].values>=5e9,:].copy()
        subhalo_df.reset_index(drop=True,inplace=True)

        for groupnum in list(range(group_df.shape[0])):
            groupmatch=subhalo_df['GroupNumber']==groupnum
            subhalo_df.loc[groupmatch,list(group_df.keys())[1:]]=group_df.iloc[groupnum].to_numpy()[1:]
            subhalo_df.loc[groupmatch,'SubGroupNumber']=np.argsort(np.argsort(-subhalo_df.loc[groupmatch,'Mass'].values)).astype(int)

        subhalo_dfs.append(subhalo_df)

    subcat=pd.concat(subhalo_dfs)
    subcat.sort_values(by=['Mass','SnapNum'],ascending=[False,False])
    subcat.reset_index(inplace=True,drop=True)

    return subcat


def submit_serial_job(func,memory,time,arguments=None,partition=None,repo=None):

    cwd=os.getcwd()
    run=cwd.split('/')[-1]

    funcname=

    jobname=f'{run}_{}'

    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('logs'):
        os.mkdir('logs')




    
    runscriptfilepath=f'{cwd}/jobs/{jobname}-run.py'
    jobscriptfilepath=f'{cwd}/jobs/{jobname}-submit.slurm'
    if os.path.exists(runscriptfilepath):
        os.remove(runscriptfilepath)
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    argumentstring=''
    for arg in arguments:
        if type(arguments[arg])==str:
            argumentstring+=f"{arg}='{arguments[arg]}',"
        else:
            argumentstring+=f"{arg}={arguments[arg]},


    with open(runscriptfilepath,"w") as runfile:
        runfile.writelines(f"import warnings\n")
        runfile.writelines(f"warnings.filterwarnings('ignore')\n")
        runfile.writelines(f"import sys\n")
        runfile.writelines(f"sys.path.append({repo})\n")
        runfile.writelines(f"from {func.split('.')} import {func.split('.')}\n")
        runfile.writelines(f"read_subcat({argumentstring})")

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

    if partition:
        os.system(f"sbatch {jobscriptfilepath} --p {partition} ")
    else:
        os.system(f"sbatch {jobscriptfilepath}")






    