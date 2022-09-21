# src_sims/illustris/particle.py: routines to read and convert particle data from TNG snapshot outputs.

from turtle import tracer
import numpy as np
import pandas as pd
import h5py 
import os
import time

from scipy.spatial import cKDTree
from hydroflow.src_physics.utils import get_limits


def read_subvol(path,ivol,nslice):
    pdata_file=h5py.File(path,'r')
    boxsize=pdata_file['Header'].attrs['BoxSize']
    hval=pdata_file['Header'].attrs['HubbleParam']
    masstable=pdata_file['Header'].attrs['MassTable']
    nparttable=pdata_file['Header'].attrs['NumPart_Total']
    pdata_file.close()
    basepath=path.split('snapdir')[0]
    snapnum=int(path.split('snapdir_')[-1][:3])
    
    flist=sorted([path.split('snap_')[0]+fname for fname in os.listdir(path.split('snap_')[0]) if '.hdf5' in fname])
    numfiles=len(flist)
    print(f'Loading from {numfiles} files')

    lims=get_limits(ivol,nslice,boxsize,buffer=0.1)

    #in addition to mass and ID
    ptype_fields={0:['InternalEnergy','ElectronAbundance','GFM_Metallicity','StarFormationRate'],
                  1:[],
                  4:['GFM_Metallicity'],
                  5:[]}
    
    pdata={ptype:{} for ptype in ptype_fields}

    for ptype in pdata:
        print(f'Loading ptype {ptype}')
        npart=nparttable[ptype]
    
        #masking
        coordinates=loadSubset(basepath,snapnum,ptype,fields=['Coordinates'],float32=True)
        subvol_mask=np.ones(coordinates.shape[0])
        
        for idim,dim in enumerate('xyz'):
            lims_idim=lims[2*idim:(2*idim+2)]
            if lims_idim[0]<0 and nslice>1:#check for periodic
                otherside=coordinates[:,idim]>=(boxsize+lims_idim[0])
                coordinates[:,idim][otherside]=coordinates[:,idim][otherside]-boxsize
            if lims_idim[1]>boxsize and nslice>1:#check for periodic
                otherside=coordinates[:,idim]<=(lims_idim[1]-boxsize)
                coordinates[:,idim][otherside]=coordinates[:,idim][otherside]+boxsize
            idim_mask=np.logical_and(coordinates[:,idim]>=lims_idim[0],coordinates[:,idim]<=lims_idim[1])

            try:
                print(coordinates.shape[0])
                print(subvol_mask.shape[0])
            except:
                pass

            subvol_mask=np.logical_and(subvol_mask,idim_mask)
        
        subvol_mask=np.where(subvol_mask)
        coordinates=coordinates[subvol_mask]*1e-3

        #coordinates
        pdata[ptype]=pd.DataFrame(coordinates,columns=[f'Coordinates_{x}' for x in 'xyz'])
        del coordinates

        #ID and mass
        if not ptype==1:
            pdata_idmass=loadSubset(basepath,snapnum,ptype,fields=['ParticleIDs','Masses'],subset=subvol_mask,float32=True)
            pdata[ptype]['ParticleIDs']=pdata_idmass['ParticleIDs'];del pdata_idmass['ParticleIDs']
            pdata[ptype].loc[:,'Mass']=pdata_idmass['Masses'];del pdata_idmass['Masses']
        else:
            pdata[ptype]['ParticleIDs']=loadSubset(basepath,snapnum,ptype,fields=['ParticleIDs'],subset=subvol_mask,float32=True)
            pdata[ptype].loc[:,'Mass']=masstable[1]

        pdata[ptype]['Mass']=pdata[ptype]['Mass']*1e10/hval
        pdata[ptype].loc[:,'ParticleType']=ptype
        
        #everything else
        if len(ptype_fields[ptype]):
            pdata_rest=loadSubset(basepath,snapnum,ptype,fields=ptype_fields[ptype],subset=subvol_mask,float32=True)
            if len(ptype_fields[ptype])>1:
                for field in list(pdata_rest.keys())[1:]:
                    pdata[ptype][field]=pdata_rest[field];del pdata_rest[field]
            else:
                field=ptype_fields[ptype][0]
                pdata[ptype][field]=pdata_rest;del pdata_rest
    
    pdata[0]=pd.concat([pdata[ptype] for ptype in [0,4,5]]);del pdata[4],pdata[5] #baryons!
    pdata[0].sort_values(by=['ParticleIDs'],inplace=True)
    pdata[0].reset_index(drop=True,inplace=True)

    ####### tracers boi
    print(f'Loading & matching tracers')

    numbar=np.nansum([nparttable[ptype]for ptype in [0,4,5]])
    numtracers=np.nansum([nparttable[ptype]for ptype in [3]])
    t0=time.time()
    if numbar and numtracers:
        tracerdata=loadSubset(basepath,snapnum,3,subset=None,fields=['ParentID','TracerID'],float32=True)        
        pdata_tracer_parentIDs=tracerdata['ParentID'];del tracerdata['ParentID']
        pdata_tracer_tracerIDs=tracerdata['TracerID'];del tracerdata['TracerID']
        pdata_ifile_baryons_IDs=pdata[0]['ParticleIDs'].values

        # #all tracers in this file
        expected_idx_of_tracer_in_pdata=np.searchsorted(pdata_ifile_baryons_IDs,pdata_tracer_parentIDs)
        tracer_match_1=pdata_tracer_parentIDs==np.concatenate([pdata_ifile_baryons_IDs,[np.nan]])[(expected_idx_of_tracer_in_pdata,)]
        pdata_tracer_tracerIDs=pdata_tracer_tracerIDs[tracer_match_1]
        expected_idx_of_tracer_in_pdata=expected_idx_of_tracer_in_pdata[tracer_match_1]

        pdata[0]=pdata[0].loc[expected_idx_of_tracer_in_pdata,:]; pdata[0].reset_index(inplace=True,drop=True)
        pdata[0]['ParticleIDs']=pdata_tracer_tracerIDs #set particle IDs as the tracer IDs
        pdata[0]['ParentID']=pdata[0]['ParticleIDs'].values

        print(f'Matched tracers in {time.time()-t0:.3f} sec ({np.nanmean(tracer_match_1)*100:.4f}% of the tracers in this file were in the desired ivol {ivol}/{nslice**3})')

    else:
        print('No baryons in ifile for desired volume, will not match tracers')
        del pdata[0]

    #concat all pdata into one df
    pdata=pd.concat(pdata)
    pdata.sort_values(by="ParticleIDs",inplace=True)
    pdata.reset_index(inplace=True,drop=True)

    print(f'Making KDTree')
    pdata_kdtree=cKDTree(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values)
    return pdata, pdata_kdtree





        






















#####################################################################
########## fork of illustris_python


""" Illustris Simulation: Public Data Release.
snapshot.py: File I/O related to the snapshot files. """

import numpy as np
import h5py
import six


def partTypeNum(partType):
    """ Mapping between common names and numeric particle types. """
    if str(partType).isdigit():
        return int(partType)
        
    if str(partType).lower() in ['gas','cells']:
        return 0
    if str(partType).lower() in ['dm','darkmatter']:
        return 1
    if str(partType).lower() in ['dmlowres']:
        return 2 # only zoom simulations, not present in full periodic boxes
    if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
        return 3
    if str(partType).lower() in ['star','stars','stellar']:
        return 4 # only those with GFM_StellarFormationTime>0
    if str(partType).lower() in ['wind']:
        return 4 # only those with GFM_StellarFormationTime<0
    if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
        return 5
    
    raise Exception("Unknown particle type name.")


def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basePath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart



def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        # if subset:
        #     offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

        #     fileNum = np.max(np.where(offsetsThisType >= 0))
        #     fileOff = offsetsThisType[fileNum]
        #     numToRead = subset['lenType'][ptNum]
        # else:

        fileNum = 0
        fileOff = 0
        numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)+np.nan

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead and fileNum<=10:

        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')
        if not fileNum%10:
            print('chunk ', fileNum)
        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    # if origNumToRead != wOffset:
    #     raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if subset:
        for field in fields:
            result[field]=result[field][subset]

    if sq and len(fields) == 1:
        return result[fields[0]]
    
    #do temp conv here
    if 'InternalEnergy' in fields:
        ne     = result['ElectronAbundance']
        energy = result['InternalEnergy']
        yhelium = 0.0789
        Temp = energy*(1.0 + 4.0*yhelium)/(1.0 + yhelium + ne)*1e10*(2.0/3.0)
        Temp *= (1.67262178e-24/ 1.38065e-16  )
        result['Temperature']=Temp
        del result['InternalEnergy']
        del result['ElectronAbundance']

    return result


# def getSnapOffsets(basePath, snapNum, id, type):
#     """ Compute offsets within snapshot for a particular group/subgroup. """
#     r = {}

#     # old or new format
#     if 'fof_subhalo' in gcPath(basePath, snapNum):
#         # use separate 'offsets_nnn.hdf5' files
#         with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
#             groupFileOffsets = f['FileOffsets/'+type][()]
#             r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
#     else:
#         # load groupcat chunk offsets from header of first file
#         with h5py.File(gcPath(basePath, snapNum), 'r') as f:
#             groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
#             r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

#     # calculate target groups file chunk which contains this id
#     groupFileOffsets = int(id) - groupFileOffsets
#     fileNum = np.max(np.where(groupFileOffsets >= 0))
#     groupOffset = groupFileOffsets[fileNum]

#     # load the length (by type) of this group/subgroup from the group catalog
#     with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
#         r['lenType'] = f[type][type+'LenType'][groupOffset, :]

#     # old or new format: load the offset (by type) of this group/subgroup within the snapshot
#     if 'fof_subhalo' in gcPath(basePath, snapNum):
#         with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
#             r['offsetType'] = f[type+'/SnapByType'][id, :]
#     else:
#         with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
#             r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

#     return r


# def loadSubhalo(basePath, snapNum, id, partType, fields=None):
#     """ Load all particles/cells of one type for a specific subhalo
#         (optionally restricted to a subset fields). """
#     # load subhalo length, compute offset, call loadSubset
#     subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
#     return loadSubset(basePath, snapNum, partType, fields, subset=subset)


# def loadHalo(basePath, snapNum, id, partType, fields=None):
#     """ Load all particles/cells of one type for a specific halo
#         (optionally restricted to a subset fields). """
#     # load halo length, compute offset, call loadSubset
#     subset = getSnapOffsets(basePath, snapNum, id, "Group")
#     return loadSubset(basePath, snapNum, partType, fields, subset=subset)
