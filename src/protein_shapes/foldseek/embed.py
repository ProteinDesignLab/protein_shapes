# helper functions to generate foldseek embeddings

import os
from pathlib import Path
import time
import json

from tqdm import tqdm


# Get structural seqs from single pdb file
def get_struc_seq_single(
    foldseek,
    path,
    chains: list = None,
    process_id: int = 0,
    foldseek_verbose: bool = False,
) -> dict:

    assert os.path.exists(path), f"PDB file not found: {path}"

    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq, _ = line.strip().split("\t")

            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            if chains is None or chain in chains:
                if chain not in seq_dict:
                    seq_dict[chain] = struc_seq

    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


# Get structural seq dicts from directory of pdb files
def get_struc_seq(foldseek, directory, json_file_path) -> None:
    struc_seqs = []
    for filename in tqdm(os.listdir(directory)):
        if not Path(filename).is_dir():
            file_path = os.path.join(directory, filename)
            try:
                seq_dict = get_struc_seq_single(foldseek, file_path)
                struc_seqs.extend(seq_dict.values())
            except Exception as e:
                print(
                    f"Error occurred while generating foldseek embedding for {file_path}: {e}"
                )

    with open(json_file_path, "w") as f:
        json.dump(struc_seqs, f)
    print(f"Stored foldseek embeddings in {json_file_path}")
