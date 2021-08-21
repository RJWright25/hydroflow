
import os
import time
import logging
import h5py
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from astropy.cosmology import Planck13 as cosmology

# Retrieve haloes from tree files
def extract_tree(path,mcut=10,snipidxmin=0):

    #outputs
    outname='catalogues/catalogue_tree.hdf5'
    fields=['snapshotNumber',
            'redshift',
            'nodeIndex',
            'nodeMass',
            'descendantIndex',
            'mainProgenitorIndex',
            'mostBoundID',
            'isFoFCentre',
            'isMainProgenitor']

    if os.path.exists('jobs/extract_tree.log'):
        os.remove('jobs/extract_tree.log')

    logging.basicConfig(filename='jobs/extract_tree.log', level=logging.INFO)

    mcut=10**mcut
    logging.info(f'Running tree extraction for haloes with mass above {mcut:.1e} Msun after (and including) snapidx {snipidxmin} ...')

    # get file names
    tree_fnames=os.listdir(path)
    tree_fnames=[tree_fname for tree_fname in tree_fnames if 'tree' in tree_fname]
    nfiles=len(tree_fnames)

    # iterate through all tree files
    t0=time.time()
    treedata={}
    for ifile,tree_fname in enumerate(tree_fnames):
        
        print(f'Processing file {ifile+1} of {nfiles}')

        logging.info(f'Processing file {ifile+1} of {nfiles}')
        treefile=h5py.File(f'{path}/{tree_fname}')

        #mass mask
        masses=treefile['/haloTrees/nodeMass'].value*10**10/(cosmology.H0.value/100)

        snipshotidx=treefile['/haloTrees/snapshotNumber'].value
        mask=np.logical_and(masses>mcut,snipshotidx>=snipidxmin)

        #initialise new data
        logging.info(f'Extracting position for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        treedata[ifile]=pd.DataFrame(treefile['/haloTrees/position'].value[mask,:],columns=['position_x','position_y','position_z'])

        #grab all fields
        logging.info(f'Extracting data for {np.sum(mask):.0f} nodes [runtime {time.time()-t0:.2f} sec]')
        treedata[ifile].loc[:,fields]=np.column_stack([treefile['/haloTrees/'+field][mask] for field in fields])

        #close file, move to next
        treefile.close()

    treedata=pd.concat([treedata[ifile] for ifile in range(len(tree_fnames))],ignore_index=True)
    treedata.reset_index(drop=True,inplace=True)

    #convert masses
    treedata['nodeMass']=treedata['nodeMass'].values*10**10/(cosmology.H0.value/100)

    if os.path.exists(outname):
        os.remove(outname)
    
    treedata.to_hdf(f'{outname}',key='Tree')

# Retrieve groups from subfind files
def extract_fof(path,mcut=10,snipidxmin=0):
    
    #basic data
    redshifts=pd.read_pickle('redshifts.dat')

    #outputs
    outname='catalogues/catalogue_fof.hdf5'
    fields=['/FOF/GroupMass',
            '/FOF/Group_M_Crit200',
            '/FOF/Group_R_Crit200',
            '/FOF/NumOfSubhalos',
            '/FOF/GroupCentreOfPotential']
            
    if os.path.exists('jobs/extract_fof.log'):
        os.remove('jobs/extract_fof.log')
    logging.basicConfig(filename='jobs/extract_fof.log', level=logging.INFO)
    logging.info(f'Running FOF extraction for FOFs with mass above {mcut*10**10:.1e} after (and including) snipidx {snipidxmin} ...')
    
    mcut=10**mcut

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    fofdata=[];dims='xyz'
    ifile=0;isnip=0
    t0=time.time()
    for groupdir in groupdirs:
        snip=int(groupdir.split('snip_')[-1][:3])
        try:
            snipidx=redshifts.loc[snip==redshifts['snipshot'].values,'snipshotidx'].values[0]
        except:
            logging.info(f'Skipping snip {snip} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snipidx>=snipidxmin:
            logging.info(f'')
            logging.info(f'***********************************************************************')
            logging.info(f'Processing snipidx {snipidx} ({isnip+1}/{len(groupdirs)} total) [runtime {time.time()-t0:.2f} sec]')
            logging.info(f'***********************************************************************')

            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                with h5py.File(groupdirfname,'r') as groupdirifile:
                    ifile_fofmasses=groupdirifile['/FOF/GroupMass'][::100]*10**10/(cosmology.H0.value/100)

                    ifile_nfof=np.sum(ifile_fofmasses>mcut)
        
                    if ifile_nfof:
                        ifile_fofmasses=groupdirifile['/FOF/GroupMass'].value*10**10/(cosmology.H0.value/100)
                        ifile_mask=ifile_fofmasses>mcut
                        ifile_nfof=np.sum(ifile_mask)

                        logging.info(f'File {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nfof:.0f} FOFs [runtime {time.time()-t0:.2f} sec]')
                        newdata=pd.DataFrame(np.int16(np.ones(ifile_nfof)*snipidx),columns=['snipshotidx'])
                        for ifield,field in enumerate(fields):
                            if ifield<=1:
                                fac=10**10/(cosmology.H0.value/100)
                            else:
                                fac=1.

                            dset_shape=groupdirifile[field].shape
                            if len(dset_shape)==2:
                                for icol in range(dset_shape[-1]):
                                    if dset_shape[-1]==3:
                                        newdata[field.split('FOF/')[-1]+f'_{dims[icol]}']=groupdirifile[field].value[ifile_mask,icol]*fac
                                    else:
                                        if icol in [0,1,4,5]:
                                            newdata[field.split('FOF/')[-1]+f'_{icol}']=groupdirifile[field].value[ifile_mask,icol]*fac
                            else:
                                newdata[field.split('FOF/')[-1]]=groupdirifile[field].value[ifile_mask]*fac
                        
                        fofdata.append(newdata)
        
                        ifile+=1

                    groupdirifile.close()

            isnip+=1


    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final FOF data structure...')
    logging.info(f'*********************************************')

    fofdata=pd.concat(fofdata,ignore_index=True)
        
    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final FOF data structure...')
    logging.info(f'*********************************************')

    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    fofdata.to_hdf(f'{outname}',key='FOF')

# Retrieve subhaloes from subfind files
def extract_subhalo(path,mcut,snipidxmin=0):
    #basic data
    redshifts=pd.read_pickle('redshifts.dat')

    #outputs
    outname='catalogues/catalogue_subhalo.hdf5'
    fields=['/Subhalo/GroupNumber',
            '/Subhalo/SubGroupNumber',
            '/Subhalo/Mass',
            '/Subhalo/MassType',
            '/Subhalo/ApertureMeasurements/Mass/030kpc',
            '/Subhalo/ApertureMeasurements/VelDisp/030kpc',
            '/Subhalo/Vmax',
            '/Subhalo/VmaxRadius',
            '/Subhalo/CentreOfPotential',
            '/Subhalo/Velocity',
            '/Subhalo/CentreOfMass',
            '/Subhalo/IDMostBound',
            '/Subhalo/HalfMassRad',
            '/Subhalo/BlackHoleMassAccretionRate',
            '/Subhalo/Stars/Spin',
            '/Subhalo/GasSpin']

    mcut=10**mcut

    groupdirs=os.listdir(path)
    groupdirs=sorted([path+groupdir for groupdir in groupdirs if ('groups_snip' in groupdir and 'tar' not in groupdir)])

    if os.path.exists('jobs/extract_subhalo.log'):
        os.remove('jobs/extract_subhalo.log')

    logging.basicConfig(filename='jobs/extract_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo extraction for subhaloes with mass above {mcut:.1e} after (and including) snapidx {snipidxmin} ...')

    subhalo_data=[];dims='xyz'
    ifile=0;isnip=0
    t0=time.time()

    for groupdir in groupdirs:
        snip=int(groupdir.split('snip_')[-1][:3])
        try:
            snipidx=redshifts.loc[snip==redshifts['snipshot'].values,'snipshotidx'].values[0]
        except:
            logging.info(f'Skipping snip {snip} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue

        if snipidx>=snipidxmin:
            logging.info(f'')
            logging.info(f'***********************************************************************')
            logging.info(f'Processing snipidx {snipidx} ({isnip+1}/{len(groupdirs)} total) [runtime {time.time()-t0:.2f} sec]')
            logging.info(f'***********************************************************************')

            groupdirfnames=os.listdir(groupdir)
            groupdirfnames=sorted([groupdir+'/'+groupdirfname for groupdirfname in groupdirfnames if groupdirfname.startswith('eagle_subfind')])
            groupdirfnames_n=len(groupdirfnames)

            for ifile_snap, groupdirfname in enumerate(groupdirfnames):
                with h5py.File(groupdirfname,'r') as groupdirifile:
                    ifile_submasses=groupdirifile['/Subhalo/Mass'][::100]*10**10/(cosmology.H0.value/100)
                    ifile_nsub=np.sum(ifile_submasses>mcut)
        
                    if ifile_nsub:
                        ifile_submasses=groupdirifile['/Subhalo/Mass'].value*10**10/(cosmology.H0.value/100)
                        ifile_mask=ifile_submasses>mcut
                        ifile_nsub=np.sum(ifile_mask)

                        logging.info(f'File {ifile_snap+1}/{groupdirfnames_n}: extracting data for {ifile_nsub:.0f} subhaloes [runtime {time.time()-t0:.2f} sec]')
                        newdata=pd.DataFrame(groupdirifile['/Subhalo/Mass'][ifile_mask],columns=['Mass'])
                        newdata.loc[:,'snipshotidx']=snipidx

                        for field in fields:
                            if 'Mass' in field and 'Centre' not in field:
                                fac=10**10/(cosmology.H0.value/100)
                            else:
                                fac=1.

                            dset_shape=groupdirifile[field].shape
                            if len(dset_shape)==2:
                                for icol in range(dset_shape[-1]):
                                    if dset_shape[-1]==3:
                                        newdata.loc[:,field.split('Subhalo/')[-1]+f'_{dims[icol]}']=groupdirifile[field].value[ifile_mask,icol]*fac
                                    else:
                                        if icol in [0,1,4,5]:
                                            newdata.loc[:,field.split('Subhalo/')[-1]+f'_{icol}']=groupdirifile[field].value[ifile_mask,icol]*fac
                            else:
                                newdata.loc[:,field.split('Subhalo/')[-1]]=groupdirifile[field].value[ifile_mask]*fac

                        subhalo_data.append(newdata)

                        ifile+=1

                groupdirifile.close()

            isnip+=1

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Concatenating final subhalo data structure...')
    logging.info(f'*********************************************')
    subhalo_data=pd.concat(subhalo_data,ignore_index=True)
    subhalo_data.reset_index(drop=True,inplace=True)

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subhalo_data.to_hdf(f'{outname}',key='Subhalo')

# Match subhaloes to groups & tree
def match_subhalo(fof_mcut=5e10):
    outname='catalogues/catalogue_subhalo_matched.hdf5'
    redshifts=pd.read_pickle('redshifts.dat')

    subhalo=pd.read_hdf('catalogues/catalogue_subhalo.hdf5')
    tree=pd.read_hdf('catalogues/catalogue_tree.hdf5');tree_fields=list(tree.columns)
    fof=pd.read_hdf('catalogues/catalogue_fof.hdf5');fof_fields=list(fof.columns)

    if os.path.exists('jobs/match_subhalo.log'):
        os.remove('jobs/match_subhalo.log')

    logging.basicConfig(filename='jobs/match_subhalo.log', level=logging.INFO)
    logging.info(f'Running subhalo matching (to trees and fofs above mass cut)')

    boxsize=cosmology.H0.value
    snipshotidxs=subhalo['snipshotidx'].unique();subhalo_out=[]
    # snipshotidxs=[199,200];subhalo_out=[]
    t0=time.time()

    for isnip,snipidx in enumerate(snipshotidxs):
        logging.info(f'')
        logging.info(f'***********************************************************************')
        logging.info(f'Processing snipidx {snipidx} ({isnip+1}/{len(snipshotidxs)} total) [runtime {time.time()-t0:.2f} sec]')
        logging.info(f'***********************************************************************')

        try:
            lookbacktime=redshifts.loc[snipidx==redshifts['snipshotidx'].values,'lookbacktime'].values[0]
        except:
            logging.info(f'Skipping snipidx {snipidx} (not in trees) [runtime {time.time()-t0:.2f} sec]')
            continue


        mask_sub=subhalo['snipshotidx'].values==snipidx        
        mask_fof=np.logical_and(fof['snipshotidx'].values==snipidx,fof['GroupMass'].values>=fof_mcut)
        mask_tree=tree['snapshotNumber'].values==snipidx

        subhalo_snap=subhalo.loc[mask_sub,:].copy();subhalo_snap.reset_index(drop=True,inplace=True)
        fof_snap=fof.loc[mask_fof,:].copy();fof_snap.reset_index(drop=True,inplace=True)
        tree_snap=tree.loc[mask_tree,:].copy();tree_snap.reset_index(drop=True,inplace=True)
        
        for field in np.concatenate([tree_fields,fof_fields]):
            subhalo_snap.loc[:,field]=np.nan
        subhalo_snap.loc[:,'nodeIndex_host']=-1

        subhalo_com=subhalo_snap.loc[:,[f'CentreOfPotential_{x}' for x in 'xyz']].values
        fof_com=fof_snap.loc[:,[f'GroupCentreOfPotential_{x}' for x in 'xyz']].values
        tree_com=tree_snap.loc[:,[f'position_{x}' for x in 'xyz']].values

        subhalo_tree=cKDTree(subhalo_com,boxsize=boxsize)
        fof_tree=cKDTree(fof_com,boxsize=boxsize)
        tree_tree=cKDTree(tree_com,boxsize=boxsize)

        ##### TREE MATCHING #####
        tree_pairs=tree_tree.query_ball_tree(subhalo_tree,r=0.001)
        tree_idxs=[];subhalo_idxs=[]
        for itree,imatch in enumerate(tree_pairs):
            if imatch:
                tree_idxs.append(itree)
                subhalo_idxs.append(imatch[0])
        subhalo_snap.loc[subhalo_idxs,tree_fields]=tree_snap.loc[tree_idxs,:].values

        ##### FOF MATCHING #####
        #returns list for each fof of the matching subhalo(s)
        central_pairs=fof_tree.query_ball_tree(subhalo_tree,r=0.001)
        fof_idxs=[];central_idxs=[]

        for ifof,imatch in enumerate(central_pairs):
            if imatch:
                fof_idxs.append(ifof)
                central_idxs.append(imatch[0])

        #matching centrals
        subhalo_snap.loc[central_idxs,fof_fields]=fof_snap.loc[fof_idxs,:].values
        central_pair_groupnums=subhalo_snap.loc[central_idxs,'GroupNumber'].values
        central_pair_fofcoms=fof_snap.loc[fof_idxs,[f'GroupCentreOfPotential_{x}' for x in 'xyz']].values
        central_pair_nodeidxs=subhalo_snap.loc[central_idxs,'nodeIndex'].values

        for iifof,(ifof,groupnum,com,nodeidx) in enumerate(zip(fof_idxs,central_pair_groupnums,central_pair_fofcoms,central_pair_nodeidxs)):
            if not iifof%10000:
                print(f'{iifof}/{len(central_pair_nodeidxs)}')
            #check for sats
            groupnum_match=np.logical_and(subhalo_snap['GroupNumber'].values==groupnum,subhalo_snap['SubGroupNumber'].values>0.1)
            
            if np.sum(groupnum_match):
                subhalo_snap.loc[groupnum_match,fof_fields]=fof_snap.loc[ifof,:].values
                subhalo_com=subhalo_snap.loc[groupnum_match,[f'CentreOfPotential_{x}' for x in 'xyz']].values
                subhalo_snap.loc[groupnum_match,'R_halocentric']=np.sqrt(np.sum(np.square(subhalo_com-com),axis=1))
                subhalo_snap.loc[groupnum_match,'nodeIndex_host']=nodeidx
        subhalo_snap.loc[:,'lookbacktime']=lookbacktime
        subhalo_out.append(subhalo_snap)

    subhalo_out=pd.concat(subhalo_out,ignore_index=True)

    logging.info(f'')
    logging.info(f'*********************************************')
    logging.info(f'Saving final subhalo data structure...')
    logging.info(f'*********************************************')
    if os.path.exists(f'{outname}'):
        os.remove(f'{outname}')
    subhalo_out.to_hdf(f'{outname}',key='Subhalo')

# orbweaver
def read_orbdata(filenamelist,iFileno=False,apsispoints=True,crossingpoints=True,endpoints=True,desiredfields=[]):

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
