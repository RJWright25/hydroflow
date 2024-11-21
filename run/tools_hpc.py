# HYDROFLOW – GAS FLOWS IN COSMOLOGICAL SIMULATIONS
# Ruby Wright (2021)

# run/tools_hpc.py: handy functions to submit gas flow execute script via batch system.

import os

def create_dir(path):
    """
    create_dir: Create a directory if it does not exist.
    
    Input:
    -----------
    path: str
        Path to the directory.

    Output:
    -----------
    None
    (Creates a directory at the specified path.)

    """
    running_dir=''
    for idir in path.split('/')[:-1]:
        running_dir=running_dir+'/'+idir
        if not os.path.exists(running_dir):
            os.mkdir(running_dir)

def submit_gasflow_jobarray(repo,arguments,memory,time,partition=None,account=None,array=None,dependency=None):

    """
    submit_gasflow_jobarray: Submit a job array to the batch system for gas flow calculations. 
    
        NOTE: This function is should only be used if the HPC system does not reserve full nodes for e.g. single core tasks. If full nodes are reserved, use a node-level submission script.

    Input:
    -----------
    repo: str
        Path to the repository.
    arguments: dict
        Dictionary containing the arguments for the gas flow calculation.
    memory: int
        Memory per CPU in GB.
    time: str
        Time limit for the job.
    array: str
        Array of subvolume indices to process.
    partition: str
        Partition to submit the job to.
    account: str
        Account to charge the job to.
    dependency: str
        Job dependencies.

    
    Output:
    -----------
    None
    (Submits a job array to the batch system.)

    
    """
    
    code=arguments['code']
    pathcat=arguments['path']
    nslice=int(arguments['nslice'])
    nvol=nslice**3
    snap=int(arguments['snap'])
    mcut=arguments['mcut']
    dump=arguments['dump']
    drfac=arguments['dr']
    namecat=pathcat.split('/')[-1][:-5]
    cwd=pathcat.split('catalogues')[0]    
    dr_str=f"{drfac:.2f}".replace('.','p')

    jobfolder=f'{cwd}/jobs/gasflow/{namecat}/nvol{str(int(nslice**3)).zfill(3)}_dr{dr_str}/snap{str(snap).zfill(3)}/'
    jobname=f"s{str(snap).zfill(3)}_n{str(int(nslice**3)).zfill(3)}"
    create_dir(jobfolder)

    runscriptfilepath=repo+'/run/execute.py'
    num=len(array.split(','))

    jobscriptfilepath=f'{jobfolder}submit-{jobname}_a{str(num).zfill(3)}.slurm'
    
    if os.path.exists(jobscriptfilepath):
        os.remove(jobscriptfilepath)

    with open(jobscriptfilepath,"w") as jobfile:
        jobfile.writelines(f"#!/bin/sh\n")
        jobfile.writelines(f"#SBATCH --job-name={jobname}\n")
        jobfile.writelines(f"#SBATCH --time={time}\n")
        jobfile.writelines(f"#SBATCH --ntasks=1\n")
        jobfile.writelines(f"#SBATCH --mem-per-cpu={memory}GB\n")
        jobfile.writelines(f"#SBATCH --array={array}\n")
        if dependency:
            jobfile.writelines(f"#SBATCH --dependency={dependency}\n")
        if partition:
            jobfile.writelines(f"#SBATCH --partition={partition}\n")
        if account:
            jobfile.writelines(f"#SBATCH --account={account}\n")

        jobfile.writelines(f"#SBATCH --output={jobfolder}{jobname}_ivol%03a.out\n")

        jobfile.writelines(f" \n")

        jobfile.writelines(f"echo JOB START TIME\n")
        jobfile.writelines(f"date\n")
        jobfile.writelines(f"echo CPU DETAILS\n")
        jobfile.writelines(f"lscpu\n")
        jobfile.writelines(f"python {runscriptfilepath}  --repo {repo} --code {code} --path {pathcat} --nslice {nslice} --snap {snap} --mcut {mcut} --dump {dump} --dr {drfac} --ivol $SLURM_ARRAY_TASK_ID \n")
        jobfile.writelines(f"echo JOB END TIME\n")
        jobfile.writelines(f"date\n")

    jobfile.close()
    os.system(f"sbatch {jobscriptfilepath}")
    print(f"sbatch  {jobscriptfilepath}")

    return None