"""
Use to get intuition on how structural perturbations are reflected in the SHAPES metrics
 - Gaussian noise at various levels
 - Expansion and compression (residue frame rotations are still correct, but bond distances are not)
"""

from copy import deepcopy
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Structure, Selection
import numpy as np


pdb_parser = PDBParser(QUIET=True)
io = PDBIO()


def save_pdb(structure: Structure, fp: Path):
    io.set_structure(structure)
    io.save(str(fp))


def structure_to_xyz(structure: Structure.Structure) -> np.ndarray:
    coords = []
    for atom in Selection.unfold_entities(structure, "A"):
        coords.append(atom.coord)
    return np.array(coords)


def pdb_to_xyz(fp: Path) -> tuple[Structure.Structure, np.ndarray]:
    """Load PDB into numpy array

    Parameters
    ----------
    fp : Path
        Path to PDB
    idx : dict[str, list]
        Dictionary containing chain -> residue -> atom info to subset the structure

    Returns
    -------
    np.ndarray
        Atomic coordinates
    """
    structure = pdb_parser.get_structure("s", fp)
    xyz = structure_to_xyz(structure)
    return structure, xyz


def center_xyz(xyz: np.ndarray) -> np.ndarray:
    return xyz - np.mean(xyz, axis=0, keepdims=True)


def xyz_to_pdb(structure, xyz) -> Structure:
    """modifies the xyz coordinates of a pdb structure"""
    structure = deepcopy(structure)
    for i, atom in enumerate(Selection.unfold_entities(structure, "A")):
        atom.coord = xyz[i]
    return structure


noise_levels = np.linspace(0.01, 2.0, 10)


def gaussian_noise(pdb_dir: Path):
    save_dir = pdb_dir / "gaussian_perturb"
    save_dir.mkdir(parents=True, exist_ok=True)
    for level in noise_levels:
        for fp in pdb_dir.glob("*.pdb"):
            structure, xyz = pdb_to_xyz(fp)
            xyz = xyz + level * np.random.randn(*xyz.shape)
            structure = xyz_to_pdb(structure, xyz)
            save_fp = save_dir / f"{fp.stem}_noise_{level:.2f}.pdb"
            save_pdb(structure, save_fp)


def expansion_noise(pdb_dir: Path):
    save_dir = pdb_dir / "expansion_perturb"
    save_dir.mkdir(parents=True, exist_ok=True)
    for level in noise_levels:
        for fp in pdb_dir.glob("*.pdb"):
            structure, xyz = pdb_to_xyz(fp)
            xyz = xyz + level * xyz
            structure = xyz_to_pdb(structure, xyz)
            save_fp = save_dir / f"{fp.stem}_noise_{level:.2f}.pdb"
            save_pdb(structure, save_fp)


pdb_dir = Path("tests/test_data")

gaussian_noise(pdb_dir)

expansion_noise(pdb_dir)
