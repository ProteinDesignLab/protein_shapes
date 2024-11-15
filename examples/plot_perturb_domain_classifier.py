from joblib import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from protein_shapes.paths import DATA_DIR
from protein_shapes.utils import fpd, load_embeddings


gt_mu = np.load(DATA_DIR / "prot_domain_classifier" / "cath_mu.npy")
gt_sigma = np.load(DATA_DIR / "prot_domain_classifier" / "cath_sigma.npy")


noise_levels = np.linspace(0.01, 2.0, 10)

for noise_type in ["gaussian", "expansion"]:
    fig, ax = plt.subplots(1)

    all_fpd = []

    perturb_dir = Path(f"out_dir/prot_domain_classifier/embeddings_tests-test_data-{noise_type}_perturb")

    for level in tqdm(noise_levels):
        embeds_fps = list(perturb_dir.glob(f"*_noise_{level:.2f}_*"))
        _, perturbed_embeds = load_embeddings(embeds_fps)

        fpd_score = fpd(perturbed_embeds, gt_mu=gt_mu, gt_sigma=gt_sigma)

        all_fpd.append(fpd_score)

    ax.plot(noise_levels, all_fpd, lw=2, color='cornflowerblue')

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('FPD')
    ax.grid()

    plt.tight_layout()
    plt.savefig(f"plots/perturb_prot_domain_classifier_{noise_type}.png", dpi=300)
    plt.clf()
