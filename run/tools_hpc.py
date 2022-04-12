# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_hpc.py: handy functions to submit gas flow execute script via batch system.

import os

def submit_gasflow_jobarray(repo,arguments,memory,time,array=None):
    
    cwd=os.getcwd()
    pathcat=arguments['path']
    nslice=int(arguments['nslice'])
    nvol=nslice**3
    snapf=int(arguments['snap'])
    depth=int(arguments['depth'])
    mcut=arguments['mcut']
    namecat=pathcat.split('/')[-1][:-5]

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
        jobfile.writelines(f"#SBATCH --nodes=1\n")
        jobfile.writelines(f"#SBATCH --ntasks-per-node={1}\n")
        jobfile.writelines(f"#SBATCH --mem={memory}GB\n")
        jobfile.writelines(f"#SBATCH --time={time}\n")
        jobfile.writelines(f"#SBATCH --output={jobfolder}{jobname}_ivol%a.out\n")

        jobfile.writelines(f" \n")

        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {runscriptfilepath} --path {pathcat} --nslice {nslice} --snap {snapf} --depth {depth} --mcut {mcut} --ivol $SLURM_ARRAY_TASK_ID \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")

    jobfile.close()
    if len(array):
        os.system(f"sbatch --array={array} {jobscriptfilepath}")
        print(f"sbatch --array={array} {jobscriptfilepath}")
    else:
        os.system(f"sbatch --array=0-{nvol-1} {jobscriptfilepath}")
        print(f"sbatch --array=0-{nvol-1} {jobscriptfilepath}")

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

def create_dir(path):
    running_dir=''
    for idir in path.split('/')[:-1]:
        running_dir=running_dir+'/'+idir
        if not os.path.exists(running_dir):
            os.mkdir(running_dir)
