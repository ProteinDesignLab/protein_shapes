"""
Given CATH embeddings, plot the mean and variance of various percentages of subsampling
and error relative to the FPD value computed on the full dataset.

In this example, we compute FPD using ground truth structure embeddings against itself,
so FPD should approach zero given the full set of samples.

Routine should be repeated if custom dataset is provided as it gives a possible tradeoff
between compute time and FPD estimation accuracy
"""

from joblib import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from protein_shapes.paths import DATA_DIR
from protein_shapes.utils import fpd, load_embeddings


gt_mus, gt_sigmas = [], []

for i in range(1, 4):  # go over each encoder layer
    with open(DATA_DIR / "proteinmpnn" / f"cath_layer_{i}.pkl", "rb") as fp:
        gt_mu, gt_sigma = load(fp)[0]
        gt_mus.append(gt_mu)
        gt_sigmas.append(gt_sigma)

samp_dir = Path("embeddings/proteinmpnn/cath")

_, samp_embeds = load_embeddings(samp_dir)
samp_embeds = np.transpose(samp_embeds, (1, 0, 2))

num_samples = samp_embeds.shape[1]

subsample_thresholds = list(np.linspace(0.001, 0.01, 10)) + list(np.linspace(0.01, 0.1, 10))

num_reps = 3

colors = ['cornflowerblue', 'plum', 'goldenrod']

fig, ax = plt.subplots(1)

for layer_idx in range(3):
    results = [[] for _ in subsample_thresholds]

    for i, threshold in tqdm(enumerate(subsample_thresholds)):
        num_subsamples = int(threshold * num_samples)

        for rep in range(num_reps):
            subsample_idx = np.random.choice(num_samples, num_subsamples)

            subsampled_embeds = samp_embeds[layer_idx, subsample_idx]
            fpd_score = fpd(subsampled_embeds, gt_mu=gt_mus[layer_idx], gt_sigma=gt_sigmas[layer_idx])

            results[i].append(fpd_score)

    results = np.array(results)

    mu1, sigma1 = np.mean(results, axis=1), np.std(results, axis=1)

    ax.plot(subsample_thresholds, mu1, lw=2, label=f'Layer {layer_idx+1}', color=colors[layer_idx])
    ax.fill_between(subsample_thresholds, mu1+sigma1, mu1-sigma1, facecolor=colors[layer_idx], alpha=0.5)
ax.legend(loc='upper right')
ax.set_xlabel('Fraction of Dataset')
ax.set_ylabel('Error')
ax.grid()

plt.tight_layout()
plt.savefig("plots/subsample_proteinmpnn.png", dpi=300)
