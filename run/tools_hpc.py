# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_hpc.py: handy functions to submit gas flow execute script via batch system.

import os
from tkinter import NONE

def create_dir(path):
    running_dir=''
    for idir in path.split('/')[:-1]:
        running_dir=running_dir+'/'+idir
        if not os.path.exists(running_dir):
            os.mkdir(running_dir)

def submit_gasflow_jobarray(repo,arguments,memory,time,partition=None,array=None,dependency=None):
    
    code=arguments['code']
    pathcat=arguments['path']
    nslice=int(arguments['nslice'])
    nvol=nslice**3
    snapf=int(arguments['snap'])
    depth=int(arguments['depth'])
    mcut=arguments['mcut']
    namecat=pathcat.split('/')[-1][:-5]
    cwd=pathcat.split('catalogues')[0]

    jobfolder=f'{cwd}/jobs/gasflow/{namecat}/nvol_{str(nvol).zfill(3)}/snap{str(snapf).zfill(3)}_d{str(depth).zfill(2)}/'
    jobname=f"s{str(snapf).zfill(3)}_d{str(depth).zfill(2)}_n{str(int(nslice**3)).zfill(3)}"
    create_dir(jobfolder)

    runscriptfilepath=repo+'/run/execute.py'
    num=len(array.split(','))

    jobscriptfilepath=f'{jobfolder}submit-{jobname}_{num}.slurm'
    
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --partition={partition}\n")
        jobfile.writelines(f"#SBATCH --ntasks={1}\n")
        jobfile.writelines(f"#SBATCH --mem={memory}GB\n")
        jobfile.writelines(f"#SBATCH --time={time}\n")
        if dependency:
            jobfile.writelines(f"#SBATCH --dependency={dependency}\n")

        jobfile.writelines(f"#SBATCH --output={jobfolder}{jobname}_ivol%a.out\n")

        jobfile.writelines(f" \n")

        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {runscriptfilepath} --repo {repo} --code {code} --path {pathcat} --nslice {nslice} --snap {snapf} --depth {depth} --mcut {mcut} --ivol $SLURM_ARRAY_TASK_ID \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")

    jobfile.close()
    if len(array):
        os.system(f"sbatch --array={array} {jobscriptfilepath}")
        print(f"sbatch --array={array} {jobscriptfilepath}")
    else:
        os.system(f"sbatch --array=0-{nvol-1} {jobscriptfilepath}")
        print(f"sbatch --array=0-{nvol-1} {jobscriptfilepath}")

def submit_gasflow_disBatch(repo,arguments,memory,time,partition=None,nodetype=None,volumes=None):
    
    code=arguments['code']
    pathcat=arguments['path']
    nslice=int(arguments['nslice'])
    nvol=nslice**3
    snapf=int(arguments['snap'])
    depth=int(arguments['depth'])
    mcut=arguments['mcut']
    namecat=pathcat.split('/')[-1][:-5]
    cwd=pathcat.split('catalogues')[0]

    jobfolder=f'{cwd}/jobs/gasflow/{namecat}/nvol_{str(nvol).zfill(3)}/snap{str(snapf).zfill(3)}_d{str(depth).zfill(2)}/'
    jobname=f"s{str(snapf).zfill(3)}_d{str(depth).zfill(2)}_n{str(int(nslice**3)).zfill(3)}"
    create_dir(jobfolder)

    runscriptfilepath=repo+'/run/execute.py'
    num=len(volumes)

    disbatch_dir=f'{jobfolder}/{jobname}_{str(num).zfill(3)}/'
    if not os.path.exists(disbatch_dir):
        os.mkdir(disbatch_dir)

    jobscriptfilepath=f'{disbatch_dir}{jobname}_{str(num).zfill(3)}.tasks'
    submitscriptfilepath=f'{disbatch_dir}{jobname}_{str(num).zfill(3)}.sh'

    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    with open(jobscriptfilepath,"w") as taskfile:
        for ivol in volumes:
            taskfile.writelines(f"python {runscriptfilepath} --repo {repo} --code {code} --path {pathcat} --nslice {nslice} --snap {snapf} --depth {depth} --mcut {mcut} --ivol {ivol}&>{jobfolder}{jobname}_ivol{ivol}.out\n")
    taskfile.close()

    with open(submitscriptfilepath,"w") as submitfile:
        submitfile.writelines(f"cd {disbatch_dir}\n")        
        submitfile.writelines(f"sbatch --time {time} -n {num} --partition {partition} -C {nodetype} --mem {memory}GB --output {jobfolder}{jobname}.out --job-name {jobname} disBatch {jobscriptfilepath}\n")
        submitfile.writelines(f"cd {cwd}\n")

    submitfile.close()

    print(f'sh {submitscriptfilepath}')

def submit_gasflow_function(repo,function,arguments,memory,time):
    cwd=os.getcwd()
    run=cwd.split('/')[-1]

    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    jobname=function+'_'+run
    
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
            argumentstring+=f"{arg}={arguments[arg]},"


    with open(runscriptfilepath,"w") as runfile:
        runfile.writelines(f"import warnings\n")
        runfile.writelines(f"warnings.filterwarnings('ignore')\n")
        runfile.writelines(f"import sys\n")
        runfile.writelines(f"sys.path.append({repo})\n")
        runfile.writelines(f"{function}({argumentstring})")
    runfile.close()

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --ntasks={1}\n")
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

def submit_serial_job(func,memory,time,arguments={},partition=None,repo=None):

    cwd=os.getcwd()
    run=cwd.split('/')[-1]

    funcname=func.split('.')[-1]
    funcloc=func.split('.')[:-1]
    funclocstr=''
    for ifuncloc in funcloc:
        funclocstr+=ifuncloc+'.'
    funclocstr=funclocstr[:-1]
    print(funclocstr)


    if not os.path.exists('jobs'):
        os.mkdir('jobs')
    if not os.path.exists('logs'):
        os.mkdir('logs')

    argumentstring=''
    for arg in arguments:
        if type(arguments[arg])==str:
            argumentstring+=f"{arg}='{arguments[arg]}',"
        else:
            argumentstring+=f"{arg}={arguments[arg]},"
    
    if func=='hydroflow.src_sims.illustris.subhalo.read_subcat':
        argumentstring_fname=arguments['snapnums'][-1]
    else:
        argumentstring_fname='x'
    
    jobname=f'{funcname}_{argumentstring_fname}'
    runscriptfilepath=f'{cwd}/jobs/{jobname}-run.py'
    jobscriptfilepath=f'{cwd}/jobs/{jobname}-submit.slurm'
    if os.path.exists(runscriptfilepath):
        os.remove(runscriptfilepath)
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)


    with open(runscriptfilepath,"w") as runfile:
        runfile.writelines(f"import warnings\n")
        runfile.writelines(f"warnings.filterwarnings('ignore')\n")
        runfile.writelines(f"import sys\n")
        runfile.writelines(f"sys.path.append('{repo}')\n")
        runfile.writelines(f"from {funclocstr} import {funcname}\n")
        runfile.writelines(f"{funcname}({argumentstring})")

    runfile.close()

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}-{run}\n")
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node=1\n")
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
        os.system(f"sbatch -p {partition} {jobscriptfilepath}  ")
    else:
        os.system(f"sbatch {jobscriptfilepath}")
