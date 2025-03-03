from joblib import load
from pathlib import Path
import traceback
import subprocess
import sys

import numpy as np
import pandas as pd

from protein_shapes.paths import DATA_DIR
from protein_shapes.utils import load_embeddings, fpd


if len(sys.argv) != 6:
    print(
        "Usage: python ProteinMPNN.py <reference_structures> <sampled_structures> <output_dir> <embed_reference> <embed_samples>"
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

gt_mus, gt_sigmas = [], []

if str(reference_structures) == "cath":
    for i in range(1, 4):  # go over each encoder layer
        with open(DATA_DIR / "proteinmpnn" / f"cath_layer_{i}.pkl", "rb") as fp:
            gt_mu, gt_sigma = load(fp)[0]
            gt_mus.append(gt_mu)
            gt_sigmas.append(gt_sigma)
else:
    ref_embed_dir = output_dir / f"embeddings_{ref_suffix}"
    subprocess.run(
        [sys.executable, cur_dir / "embed.py", reference_structures, ref_embed_dir]
    )
    _, ref_embeds = load_embeddings(ref_embed_dir)
    ref_embeds = np.transpose(ref_embeds, (1, 0, 2))

samp_embed_dir = output_dir / f"embeddings_{samp_suffix}"

if embed_samples != "precomputed":
    subprocess.run([sys.executable, cur_dir / "embed.py", sampled_structures, samp_embed_dir])
else:
    samp_embed_dir = sampled_structures

_, samp_embeds = load_embeddings(samp_embed_dir)
samp_embeds = np.transpose(samp_embeds, (1, 0, 2))


fpd_scores = []

for i in range(3):
    try:
        if str(reference_structures) == "cath":
            fpd_score = fpd(samp_embeds[i], gt_mu=gt_mus[i], gt_sigma=gt_sigmas[i])
        else:
            fpd_score = fpd(samp_embeds[i], ref_embeds[i])
        fpd_scores.append(fpd_score)
        print(f"FPD score for ProteinMPNN Layer {i+1}: {fpd_score:.4f}")
    except Exception as e:
        print(f"Error during FPD calculation for ProteinMPNN Layer {i+1}: {e}")
        print(traceback.format_exc())


fpd_df = pd.DataFrame()
fpd_df["layer"] = list(range(1, 4))
fpd_df["fpd"] = fpd_scores

csv_filename = f"fpd_values_{samp_suffix}_against_{ref_suffix}.csv"
save_fp = output_dir / csv_filename

fpd_df.to_csv(save_fp, index=False, float_format='%.4f')

print(f"FPD values saved to {save_fp}")
