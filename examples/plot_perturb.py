from joblib import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from protein_shapes.paths import DATA_DIR
from protein_shapes.utils import fpd, load_embeddings


gt_mus, gt_sigmas = [], []

for i in range(1, 4):  # go over each encoder layer
    with open(DATA_DIR / "proteinmpnn" / f"cath_layer_{i}.pkl", "rb") as fp:
        gt_mu, gt_sigma = load(fp)[0]
        gt_mus.append(gt_mu)
        gt_sigmas.append(gt_sigma)


noise_levels = np.linspace(0.01, 2.0, 10)

colors = ['cornflowerblue', 'plum', 'goldenrod']

for noise_type in ["gaussian", "expansion"]:
    fig, ax = plt.subplots(1)

    all_fpd = []

    perturb_dir = Path(f"out_dir/proteinmpnn/embeddings_tests-test_data-{noise_type}_perturb")

    for level in noise_levels:
        embeds_fps = list(perturb_dir.glob(f"*_noise_{level:.2f}_*"))
        _, perturbed_embeds = load_embeddings(embeds_fps)
        perturbed_embeds = np.transpose(perturbed_embeds, (1, 0, 2))

        for layer_idx in range(3):

            fpd_score = fpd(perturbed_embeds[layer_idx], gt_mu=gt_mus[layer_idx], gt_sigma=gt_sigmas[layer_idx])

            all_fpd.append(fpd_score)

    all_fpd = np.array(all_fpd).reshape(-1, 3).T

    for layer_idx in range(3):
        ax.plot(noise_levels, all_fpd[layer_idx], lw=2, label=f'Layer {layer_idx+1}', color=colors[layer_idx])

    ax.legend(loc='upper left')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('FPD')
    ax.grid()

    plt.tight_layout()
    plt.savefig(f"plots/perturb_proteinmpnn_{noise_type}.png", dpi=300)
