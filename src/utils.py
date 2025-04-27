import os
import random
import torch
import dgl
from copy import deepcopy
from rdkit import Chem
import numpy as np


def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_oligomer_smiles(num_repeat_units, smiles, replace_star_atom=True):
    if smiles is None:
        return None
    monomer = Chem.MolFromSmiles(smiles)
    repeat_points = []
    cnct_points = []
    bond_type = []
    num_star_atoms = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == '*':
            repeat_points.append(atom.GetIdx())
            neis = atom.GetNeighbors()
            assert len(neis) == 1, f"*atom has more than one neighbor: {smiles}"
            cnct_points.append(atom.GetNeighbors()[0].GetIdx())
            bond_type.append(monomer.GetBondBetweenAtoms(atom.GetIdx(), atom.GetNeighbors()[0].GetIdx()).GetBondType())
            num_star_atoms += 1

    assert num_star_atoms == 2, "molecule has more than 2 *atoms"
    assert bond_type[0] == bond_type[1], "bond type of 2 *atoms are not same"

    num_atoms = monomer.GetNumAtoms()
    oligomer = deepcopy(monomer)
    for i in range(num_repeat_units-1):
        oligomer = Chem.CombineMols(oligomer, monomer)

    # create index list
    REPEAT_LIST, CNCT_LIST = np.zeros([num_repeat_units, 2]), np.zeros([num_repeat_units, 2])
    for i in range(num_repeat_units):
        REPEAT_LIST[i], CNCT_LIST[i] = np.array(repeat_points) + i * num_atoms, np.array(cnct_points) + i * num_atoms

    # add single bond between monomers
    ed_oligomer = Chem.EditableMol(oligomer)
    removed_atoms_idx = []
    for i in range(num_repeat_units - 1):
        ed_oligomer.AddBond(int(CNCT_LIST[i, 1]), int(CNCT_LIST[i + 1, 0]), order=bond_type[0])
        removed_atoms_idx.extend([int(REPEAT_LIST[i, 1]), int(REPEAT_LIST[i + 1, 0])])

    # Replace the atoms at both ends using H
    if replace_star_atom:
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[0, 0]), Chem.Atom(1))
        ed_oligomer.ReplaceAtom(int(REPEAT_LIST[num_repeat_units - 1, 1]), Chem.Atom(1))

    # Remove * atoms
    for i in sorted(removed_atoms_idx, reverse=True):
        ed_oligomer.RemoveAtom(i)

    final_mol = ed_oligomer.GetMol()
    final_mol = Chem.RemoveHs(final_mol)

    return Chem.MolToSmiles(final_mol)
