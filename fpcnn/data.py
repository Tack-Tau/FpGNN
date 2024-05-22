#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import os
import random
import warnings
import json
import csv
from functools import reduce, lru_cache

import numpy as np
import fplib
from ase.io import read as ase_read
from ase.neighborlist import NeighborList
#from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.multiprocessing import get_context
from sklearn.utils.class_weight import compute_class_weight

def get_train_val_test_loader(dataset, classification=False,
                              collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1,
                              return_test=False, num_workers=1,
                              pin_memory=False, persistent_workers=False,
                              multiprocessing_context=get_context('fork'),
                              shuffle=False, drop_last=True,
                              **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    classification: bool
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    shuffle: bool
    drop_last: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    if classification:
        train_indices = indices[:train_size]
        train_targets = [np.array(dataset[i][1].numpy(),
                                  dtype=np.int32).item() for i in train_indices]
        class_weights = compute_class_weight(
            class_weight = 'balanced',
            classes = np.unique(train_targets),
            y = train_targets)
        class_weights = class_weights/np.linalg.norm(class_weights)
        loss_weights = torch.tensor(class_weights, dtype=torch.float32)
        class_weights_dict = {class_idx: weight for class_idx,
                              weight in zip(np.unique(train_targets), class_weights)}
        class_weights_tensor = torch.tensor([class_weights_dict[class_idx]
                                             for class_idx in train_targets], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(
            weights = class_weights_tensor[train_indices],
            num_samples = train_size,
            replacement = False)
    else:
        train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn,
                              shuffle=shuffle,
                              drop_last=drop_last,
                              persistent_workers=persistent_workers,
                              multiprocessing_context=multiprocessing_context,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            persistent_workers=persistent_workers,
                            multiprocessing_context=multiprocessing_context,
                            pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 shuffle=shuffle,
                                 drop_last=drop_last,
                                 persistent_workers=persistent_workers,
                                 multiprocessing_context=multiprocessing_context,
                                 pin_memory=pin_memory)
    if classification:
        if return_test:
            return class_weights, train_loader, val_loader, test_loader
        else:
            return class_weights, train_loader, val_loader
    else:
        if return_test:
            return None, train_loader, val_loader, test_loader
        else:
            return None, train_loader, val_loader

def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      struct_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_struct_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_struct_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, struct_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_struct_ids.append(struct_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_struct_ids

def kron_delta(i, j):
    if i == j:
        m = 1.0
    else:
        m = 0.0
    return m

def get_gom(r_disp, dist, rcov, lmax=0):

    if lmax == 0:
        lseg = 1
    else:
        lseg = 4

    NC = 2

    assert (len(r_disp) == len(dist) and len(dist) == len(rcov)), \
    "Your r_disp_list length is different from rcov_list length!"
    nat, n_nbr = rcov.shape
    om = np.empty((nat, n_nbr-1, lseg*lseg), dtype = np.float64)
    mamp = np.empty((nat, n_nbr-1, lseg*lseg), dtype = np.float64)
    if lseg == 1:
        # s orbital only lseg == 1
        for iat in range(nat):
            cutoff = np.max(dist[iat])
            wc = cutoff / np.sqrt(2.* NC)
            fc = 1.0 / (2.0 * NC * wc**2)
            for jat in range(n_nbr-1):
                d = r_disp[iat][jat+1]
                d2 = np.vdot(d, d)
                assert np.allclose(np.sqrt(d2), dist[iat][jat+1])
                t1 = ( 0.5 / rcov[iat][0]**2 ) * ( 0.5 / rcov[iat][jat+1]**2 )
                t2 = ( 0.5 / rcov[iat][0]**2 ) + ( 0.5 / rcov[iat][jat+1]**2 )
                om[iat][jat][0] = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                mamp[iat][jat][0]=(1.0-d2*fc)**NC
    else:
        # for both s and p orbitals
        for iat in range(nat):
            cutoff = np.max(dist[iat])
            wc = cutoff / np.sqrt(2.* NC)
            fc = 1.0 / (2.0 * NC * wc**2)
            for jat in range(n_nbr-1):
                tmp_om = np.empty((lseg, lseg), dtype = np.float64)

                d = r_disp[iat][jat+1]
                d2 = np.vdot(d, d)
                assert np.allclose(np.sqrt(d2), dist[iat][jat+1])
                t1 = ( 0.5 / rcov[iat][0]**2 ) * ( 0.5 / rcov[iat][jat+1]**2 )
                t2 = ( 0.5 / rcov[iat][0]**2 ) + ( 0.5 / rcov[iat][jat+1]**2 )
                tmp_mamp = (1.0-d2*fc)**NC * np.ones((lseg, lseg), dtype = np.float64)

                # <s_i | s_j>
                sij = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)
                tmp_om[0][0] = np.sqrt(2.0*np.sqrt(t1)/t2)**3 * np.exp(-t1/t2*d2)

                # <s_i | p_j>
                stv = 2.0 * (1/np.sqrt(0.5 / rcov[iat][jat+1]**2)) * (t1/t2) * sij
                tmp_om[0][1] = stv * d[0]
                tmp_om[0][2] = stv * d[1]
                tmp_om[0][3] = stv * d[2]

                # <p_i | s_j>
                stv = -2.0 * (1/np.sqrt(0.5 / rcov[iat][0]**2)) * (t1/t2) * sij
                tmp_om[1][0] = stv * d[0]
                tmp_om[2][0] = stv * d[1]
                tmp_om[3][0] = stv * d[2]

                # <p_i | p_j>
                stv = 2.0 * np.sqrt(t1)/t2 * sij
                sx = -2.0*t1/t2

                for i_pp in range(3):
                    for j_pp in range(3):
                        tmp_om[i_pp+1][j_pp+1] = stv * (sx * d[i_pp] * d[j_pp] + \
                                                        kron_delta(i_pp, j_pp))

                tmp_om_vec = tmp_om.ravel()
                tmp_mamp_vec = tmp_om.ravel()

                for k in range(lseg*lseg):
                    om[iat][jat][k] = tmp_om_vec[k]
                    mamp[iat][jat][k] = tmp_mamp_vec[k]

    return om*mamp

class StructData(Dataset):
    """
    The StructData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── struct_id_0.vasp
    ├── struct_id_0.vasp
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal (struct_id), and the second column recodes
    the value of target property.

    struct_id_X.vasp: a CIF file that recodes the crystal structure, where
    struct_id_X is the unique ID for the crystal structure X.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors, unit in Angstroms
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    nx: int
        Maximum number of neighbors to construct the Gaussian overlap matrix for atomic Fingerprint
    lmax: int
        Integer to control whether using s orbitals only or both s and p orbitals for
        calculating the Guassian overlap matrix (0 for s orbitals only, other integers
        will indicate that using both s and p orbitals)
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (nat, atom_fp_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    struct_id: str or int
    """
    def __init__(self,
                 root_dir,
                 max_num_nbr=12,
                 radius=8.0,
                 dmin=0.5,
                 step=0.1,
                 var=1.0,
                 nx=256,
                 lmax=0,
                 random_seed=42):
        self.cache = {}
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.nx = nx
        self.lmax = lmax
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.isfile(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f, delimiter = ',')
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
    def __len__(self):
        return len(self.id_prop_data)
    
    def read_types(self, cell_file):
        buff = []
        with open(cell_file) as f:
            for line in f:
                buff.append(line.split())
        try:
            typt = np.array(buff[5], int)
        except:
            del(buff[5])
            typt = np.array(buff[5], int)
        types = []
        for i in range(len(typt)):
            types += [i+1]*typt[i]
        types = np.array(types, int)
        return types
    
    def get_fp_mat(self, cell_file):
        atoms = ase_read(cell_file)
        lat = atoms.cell[:]
        rxyz = atoms.get_positions()
        chem_nums = list(atoms.numbers)
        znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
        ntyp = len(znucl_list)
        znucl = np.array(znucl_list, int)
        types = self.read_types(cell_file)
        cell = (lat, rxyz, types, znucl)
        contract = False
        natx = int(self.nx)
        lmax = int(self.lmax)
        cutoff = np.float64(int(np.sqrt(self.radius))*3) # Shorter cutoff for GOM

        if lmax == 0:
            lseg = 1
            orbital='s'
        else:
            lseg = 4
            orbital='sp'

        if len(rxyz) != len(types) or len(set(types)) != len(znucl):
            print("Structure file: " +
                  str(cell_file.split('/')[-1]) +
                  " is erroneous, please double check!")
            if contract:
                fp = np.zeros((len(rxyz), 20), dtype = np.float64)
            else:
                fp = np.zeros((len(rxyz), lseg*natx), dtype = np.float64)
        else:
            if contract:
                fp = fplib.get_sfp(cell,
                                   cutoff=cutoff,
                                   natx=natx,
                                   log=False,
                                   orbital=orbital) # Contracted FP
                tmp_fp = []
                for i in range(len(fp)):
                    if len(fp[i]) < 20:
                        tmp_fp_at = fp[i].tolist() + [0.0]*(20 - len(fp[i]))
                        tmp_fp.append(tmp_fp_at)
                fp = np.array(tmp_fp, dtype=np.float64)
            else:
                fp = fplib.get_lfp(cell,
                                   cutoff=cutoff,
                                   natx=natx,
                                   log=False,
                                   orbital=orbital) # Long FP
        return fp
    
    def get_atom_nbr_feature(self, cell_file, radius=8.0, max_num_nbr=12):
        
        max_num_nbr, radius = self.max_num_nbr, self.radius
        
        # Read structure using ASE
        atoms = ase_read(cell_file)
        positions = atoms.get_positions()
        cell = atoms.get_cell(complete=True)
        atomic_symbols = atoms.get_chemical_symbols()
        chem_nums = list(atoms.numbers)
        max_atomic_number = max(max(chem_nums), 112)
        one_hot_encodings = []

        nat = len(atoms)
        for atom in atoms:
            encoding = [0] * (max_atomic_number + 1)
            encoding[atom.number] = 1
            one_hot_encodings.append(encoding)

        atom_fea_arr = np.array(one_hot_encodings, dtype = np.float64)

        # Convert ASE structure to Pymatgen Structure
        pymatgen_structure = Structure(cell, atomic_symbols, positions)

        # Pymatgen unit cell vectors
        S = pymatgen_structure.lattice.matrix

        # Create ASE NeighborList
        # Note: (self_interaction=True, bothways=True) will double count center atom
        nl = NeighborList([radius / 2] * len(atoms), self_interaction=True, bothways=True)
        nl.update(atoms)

        nbr_idx, nbr_dis, rcov, disp, d = [], [], [], [], []

        # Looping over POSCAR
        for iat, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(iat)
            i_center_pos = atoms[iat].position
            local_nbr_idx, local_nbr_dis, local_rcov, local_disp, local_d = [], [], [], [], []
            # Looping over neighbor list
            for j, idx in enumerate(indices):
                j_nbr_pos = atoms[idx].position
                local_nbr_idx.append(idx)
                local_nbr_dis.append(offsets[j].tolist())
                local_rcov.append(CovalentRadius.radius[atomic_symbols[idx]])

                disp_vec = j_nbr_pos - i_center_pos + np.dot(S, offsets[j])
                d2 = np.vdot(disp_vec, disp_vec)
                local_disp.append(disp_vec.tolist())
                local_d.append(np.sqrt(d2))
            # Sort neighbors for i_atom from nearest to furtherest
            combined = list(zip(local_nbr_idx, local_nbr_dis, local_rcov, local_disp, local_d))
            combined_sorted = sorted(combined, key=lambda x: x[-1])
            nbr_idx_sorted, nbr_dis_sorted, rcov_sorted, disp_sorted, d_sorted = zip(*combined_sorted)

            nbr_idx.append(list(nbr_idx_sorted))
            nbr_dis.append(list(nbr_dis_sorted))
            rcov.append(list(rcov_sorted))
            disp.append(list(disp_sorted))
            d.append(list(d_sorted))

        trim_nbr_idx, trim_nbr_dis, trim_rcov, trim_disp, trim_d = [], [], [], [], []
        for iat in range(nat):
            n_nbr = len(nbr_idx[iat])
            if n_nbr < (max_num_nbr + 2):
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cell_file))
                local_nbr_idx, local_nbr_dis, local_rcov, local_disp, local_d = \
                nbr_idx[iat], nbr_dis[iat], rcov[iat], disp[iat], d[iat]
                for i_iter in range(max_num_nbr + 2 - n_nbr):
                    local_nbr_idx.append(sorted_nbr_idx[iat][-1])
                    local_nbr_dis.append(sorted_nbr_dis[iat][-1])
                    local_rcov.append(sorted_rcov[iat][-1])
                    local_disp.append(sorted_disp[iat][-1])
                    local_d.append(sorted_d[iat][-1])
            else:
                local_nbr_idx, local_nbr_dis, local_rcov, local_disp, local_d = \
                nbr_idx[iat][:max_num_nbr+2], nbr_dis[iat][:max_num_nbr+2], \
                rcov[iat][:max_num_nbr+2], disp[iat][:max_num_nbr+2], d[iat][:max_num_nbr+2]

            trim_nbr_idx.append(local_nbr_idx)
            trim_nbr_dis.append(local_nbr_dis)
            trim_rcov.append(local_rcov)
            trim_disp.append(local_disp)
            trim_d.append(local_d)

        nbr_idx_arr, nbr_dis_arr, rcov_arr, disp_arr, d_arr = \
        np.array(trim_nbr_idx), np.array(trim_nbr_dis), np.array(trim_rcov), \
        np.array(trim_disp), np.array(trim_d)
        # Do not include center atom for nbr_idx_arr, nbr_idx_arr
        nbr_idx_arr = nbr_idx_arr[:, 2:]
        nbr_dis_arr = nbr_dis_arr[:, 2:]
        # Include center atom for nbr_idx_arr, nbr_idx_arr
        rcov_arr = rcov_arr[:, 1:]
        disp_arr = disp_arr[:, 1:]
        d_arr= d_arr[:, 1:]
        nbr_fea_arr = get_gom(disp_arr, d_arr, rcov_arr, lmax=1)

        return atom_fea_arr, nbr_fea_arr, nbr_idx_arr
    
    
    @lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        
        struct_id, target = self.id_prop_data[idx]
        cell_file = os.path.join(self.root_dir, struct_id+'.vasp')
        assert os.path.isfile(cell_file)
        
        # fp_mat = self.get_fp_mat(cell_file)
        # fp_mat[np.abs(fp_mat) < 1.0e-10] = 0.0
        # f_mat = fp_mat / np.linalg.norm(fp_mat, axis=-1, keepdims=True)
        # atom_fea = np.hstack((one_hot_encodings, fp_mat))
        
        atom_fea, nbr_fea, nbr_fea_idx = self.get_atom_nbr_feature(cell_file)
        # nbr_fea[np.abs(nbr_fea) < 1.0e-10] = 0.0
        # nbr_fea = nbr_fea / np.linalg.norm(nbr_fea, axis=-1, keepdims=True)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor([float(target)])
        
        return (atom_fea, nbr_fea, nbr_fea_idx), target, struct_id
