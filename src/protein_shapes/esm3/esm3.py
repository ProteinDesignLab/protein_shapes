import sys
from pathlib import Path
import pickle
import traceback
import subprocess

import numpy as np
import pandas as pd

from protein_shapes.paths import DATA_DIR
from protein_shapes.utils import load_embeddings, fpd
from protein_shapes.esm3.embed import esm3_embed


if len(sys.argv) != 6:
    print(
        "Usage: python esm3.py  <reference_structures> <sampled_structures> <output_dir> <embed_reference> <embed_samples>"
    )
    sys.exit(1)

reference_structures = Path(sys.argv[1])
sampled_structures = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
output_dir.mkdir(parents=True, exist_ok=True)
embed_reference = sys.argv[4]
embed_samples = sys.argv[5]

ref_suffix = (
    str(reference_structures)[1:]
    if str(reference_structures).startswith("/")
    else str(reference_structures)
).replace("/", "-")
samp_suffix = (
    str(sampled_structures)[1:]
    if str(sampled_structures).startswith("/")
    else str(sampled_structures)
).replace("/", "-")

cur_dir = Path(__file__).parent

if str(reference_structures) == "cath":
    gt_mu = np.load(DATA_DIR / "esm3" / "cath_mu.npy")
    gt_sigma = np.load(DATA_DIR / "esm3" / "cath_sigma.npy")
else:
    ref_embed_dir = output_dir / f"embeddings_{ref_suffix}"
    esm3_embed(reference_structures, ref_embed_dir / "esm3_embed.pkl")
    _, ref_embeds = load_embeddings(ref_embed_dir)

samp_embed_dir = output_dir / f"embeddings_{samp_suffix}"

if embed_samples != "precomputed":
    esm3_embed(sampled_structures, samp_embed_dir / "esm3_embed.pkl")
else:
    samp_embed_dir = sampled_structures

_, samp_embeds = load_embeddings(samp_embed_dir)


fpd_scores = []

try:
    if str(reference_structures) == "cath":
        fpd_score = fpd(samp_embeds, gt_mu=gt_mu, gt_sigma=gt_sigma)
    else:
        fpd_score = fpd(samp_embeds, ref_embeds)
    fpd_scores.append(fpd_scores)
    print(f"FPD score using ESM3: {fpd_score:.4f}")
except Exception as e:
    print(f"Error during FPD calculation: {e}")
    print(traceback.format_exc())


fpd_df = pd.DataFrame()
fpd_df["fpd"] = fpd_scores

csv_filename = f"fpd_values_{samp_suffix}_against_{ref_suffix}.csv"
save_fp = output_dir / csv_filename

fpd_df.to_csv(save_fp, index=False, float_format='%.4f')

print(f"FPD values saved to {save_fp}")
