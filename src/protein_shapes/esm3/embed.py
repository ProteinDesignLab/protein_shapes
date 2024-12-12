import os
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


login(token="$HUGGINGFACE_TOKEN")


torch.cuda.empty_cache()
model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
vqvae_encoder = model.get_structure_token_encoder()
vqvae_encoder = vqvae_encoder.to("cuda")


def esm3_embed(pdb_dir, embedding_path):
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
                pdb_path = os.path.join(pdb_dir, pdb_file)
                protein = ESMProtein.from_pdb(pdb_path)
                coords = protein.coordinates.unsqueeze(0).to("cuda")
                z_q, min_encoding_indices = vqvae_encoder.encode(coords = coords)

                embedding_dict[pdb_file] = (z_q.to("cpu"), min_encoding_indices.to("cpu"))

                del z_q, min_encoding_indices, coords
                torch.cuda.empty_cache()

            except Exception as e:
                print("error at ", pdb_files[i])
                print(e)
                continue

    with open(embedding_path, "wb") as f:
        pickle.dump(embedding_dict, f)
