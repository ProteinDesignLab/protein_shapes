import pickle
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
import matplotlib
import pickle
import numpy as np
import torch
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning) # Ignore chain breaks.
warnings.simplefilter('ignore', RuntimeWarning) # Bypass Scipy warning.
matplotlib.pyplot.switch_backend('agg')


parser = PDBParser(PERMISSIVE=1)
cif_parser = MMCIFParser(QUIET=True)


def loadSelectedChains(filename):
    in_file = open(filename, 'rb')
    sel_chain_dict = pickle.load(in_file)
    in_file.close()
    return sel_chain_dict

def getContactMap(pdb):
    if pdb.endswith("cif"):
        structure = cif_parser.get_structure(pdb[:-4], pdb)
    else:
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb[:-4], pdb)
    A = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    A.append(np.asarray(res['CA'].get_coord())) # C-alpha coordinates are extracted in residue order.
                except:
                    continue
    return distance_matrix(A,A) 

def getCMSize(pdb):
    if pdb.endswith("cif"):
        structure = cif_parser.get_structure(pdb[:-4], pdb)
    else:
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb[:-4], pdb)
    A = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    A.append(np.asarray(res['CA'].get_coord())) # C-alpha coordinates are extracted in residue order.
                except:
                    continue
    return len(A) 

# Resize Function
def resizeCM(input_arr, target_size=512, upper_tol=8, in_size=None): # size as an optional parameter if pre-calculated.
    # Note: return sizes. Use this to calculate the valid range in the assignment map.
    upper_lim = target_size + upper_tol
    if not in_size:
        in_size = np.shape(input_arr)[0]
        assert in_size == np.shape(input_arr)[1]
    if in_size > upper_lim:
        return None 
    elif in_size <= target_size:
        if in_size % 2 == 0:
            pad = ((target_size - in_size)//2,)*2
        else:
            pad = ((target_size - (in_size-1))//2,(target_size - (in_size+1))//2)
        return np.pad(input_arr, pad, mode='constant'), (target_size, in_size)
    else: # if above target size, but within tolerance, center crop.
        temp_size = in_size
        temp_arr = input_arr
        if temp_size % 2 != 0:
            temp_arr = temp_arr[:temp_size-1, :temp_size-1] # truncate off the C-terminus if 
            temp_size -= 1
        trunc = (temp_size - target_size)//2
        return temp_arr[trunc:(temp_size-trunc), trunc:(temp_size-trunc)], (target_size, in_size)

def get_pdb_info(file_name):
    ppb = PPBuilder()  # convert Structure object to polypeptide object.
    numbering = []
    handle = open(file_name, 'r')
    if file_name.endswith("cif"):
        structure = cif_parser.get_structure(file_name[:-4], handle)
    else:
        structure = parser.get_structure(file_name[:-4], handle)
    handle.close()
    # Get residue numbering.
    seq_len=0
    for model in structure:
        for chain in model:
            for residue in chain:
                try:
                    residue['CA'].get_coord()
                    seq_len+=1
                    numbering.append(residue.id[1])
                except:
                    continue
                    print("Non-Acceptable Residue Found: " + residue.get_resname())
                    print("Structure Name: " + file_name)
    assert(getCMSize(file_name) == seq_len)
    assert seq_len == len(numbering)
    return seq_len, numbering

def getAdjustedNumbering(cm, numbering, ignore_index=-9999):
    cm = cm.squeeze()
    mid_row = list(cm[int(cm.shape[0]/2),:])
    del cm
    left_pad = 0
    for i in range(len(mid_row)):
        if mid_row[i] == 0:
            left_pad += 1
        else:
            break
    right_pad = 0
    for i in range(len(mid_row)-1, -1, -1):
        if mid_row[i] == 0:
            right_pad += 1
        else:
            break
    new_numbering = left_pad*[ignore_index] + numbering + right_pad*[ignore_index]
    assert len(new_numbering) == len(mid_row)
    return new_numbering

def makeContactMapTensor(file_name, chain_len, numbering, target_size=512, upper_tol=8, scale=-100, ignore_index=-9999):
    chain_name = file_name[:-4]
    cm = getContactMap(file_name)
    try:
        assert cm.shape[0] == chain_len # check that the size of the unpadded cm and assignment length match up.
    except:
        print(chain_name)
        print(chain_len)
        print(cm.shape[0])
        raise Exception("Unpadded CM and Assignment Length aren't matching.")
    # Resize.
    cm, sizes = resizeCM(cm, target_size=target_size, upper_tol=upper_tol)
    target_size, in_size = sizes
    trunc = target_size - in_size < 0
    if trunc:
        temp_size = in_size
        if temp_size % 2 != 0:
            temp_size -= 1
        trunc = (temp_size - target_size)//2 # the size of the margins for the center crop.
        new_numbering = numbering[trunc : temp_size - trunc]
    else:
        new_numbering = numbering
    try:
        assert len(new_numbering) == in_size or len(new_numbering) == target_size # check that assignment length adjustments are good
    except AssertionError:
        print(chain_name)
        print(len(new_numbering))
        print(in_size)
        raise Exception("Assignment Length Adjustment Failed.")
    new_numbering = getAdjustedNumbering(cm, new_numbering, ignore_index=ignore_index)
    # Write Contact Map
    return torch.from_numpy(cm*scale).unsqueeze(0).unsqueeze(0).float(), new_numbering

