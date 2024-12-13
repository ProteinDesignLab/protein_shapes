import os
from pathlib import Path
import pickle
import torch
import tqdm 
import warnings

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import torch.nn as nn
import torch

warnings.filterwarnings("ignore")


if "HUGGINGFACE_TOKEN" in os.environ.keys():
    login(token=os.environ['HUGGINGFACE_TOKEN'])
else:
    login()


if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
else:
    device = "cpu"

model = ESM3.from_pretrained("esm3_sm_open_v1").to(device)
vqvae_encoder = model.get_structure_encoder()
vqvae_encoder = vqvae_encoder.to(device)


def esm3_embed(pdb_dir: Path, embedding_path: Path):
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_subdirs = [x[0] for x in os.walk(pdb_dir)]
    pdb_files = []
    for pdb_subdir in pdb_subdirs:
        pdb_files += [pdb_subdir + "/" + x for x in os.listdir(pdb_subdir) if x.endswith(".pdb")]
    embedding_dict = {}

    with torch.no_grad():   
        for i in tqdm.tqdm(range(len(pdb_files))):
            try:
                if pdb_files[i] in embedding_dict:
                    continue
                pdb_file = pdb_files[i]
                protein = ESMProtein.from_pdb(pdb_file)
                coords = protein.coordinates.unsqueeze(0).to(device)
                z_q, min_encoding_indices = vqvae_encoder.encode(coords = coords)

                embedding_dict[pdb_file] = (z_q.to("cpu"), min_encoding_indices.to("cpu"))

                del z_q, min_encoding_indices, coords
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print("error at ", pdb_files[i])
                print(e)
                continue

    with open(embedding_path, "wb") as f:
        pickle.dump(embedding_dict, f)

