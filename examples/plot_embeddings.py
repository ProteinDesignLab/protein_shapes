from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import umap

from protein_shapes.utils import calculate_frechet_distance, load_embeddings


plt.rcParams["figure.dpi"] = 300


data_dir = Path("/home/pdl/code/protein_shapes/embeddings")

data_prefix = "embeddings_home-pdl-results-fpd_paper_results-samples-"


# default sampling routine for non-Protpardelle models
titles = ["RFdiffusion", "Multiflow", "Chroma", "Genie2", "Protpardelle Public", "Protpardelle Stepscale 0.8", "Protpardelle Stepscale 1.0", "Protpardelle Stepscale 1.2"]

# temperature-adjusted sampling for non-Protpardelle models
titles = ["RFdiffusion Noise Scale 2", "RFdiffusion Noise Scale 3", "Chroma Inverse Temperature 4", "Chroma Inverse Temperature 3", "Genie2 Scale 0.8", "Genie2 Scale 1.0"]

# default sampling routine for non-Protpardelle models
model_types = [
    "rf_diffusion", "multiflow", "chroma", "genie2", 
    "public", "exp_80", "step_1.0", "step_1.2"
]

# temperature-adjusted sampling for non-Protpardelle models
model_types = ["rfdiffusion-noise_scale_2", "rfdiffusion-noise_scale_3", "chroma-samples4", "chroma-samples3", "genie2-scale_08", "genie2-scale_1"]

#! adjust based on the number of plots
rows = 3
cols = 2


def runner(embed_type: str = "prot_domain_classifier", layer: int = 0, cache: bool = False):
    """Plot UMAP of embeddings of sampled structures overlayed on reference structures

    Parameters
    ----------
    embed_type, optional
        Embedding type (prot_domain_classifier, proteinmpnn), by default "prot_domain_classifier"
    layer, optional
        ProteinMPNN encoder layer number (only applies when embed_type == "proteinmpnn"), by default 0
    cache, optional
        If False (default), computes UMAP, otherwise loads from cached UMAP coordinates
    """

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    axes = axes.flatten()
    
    gt_embed_dir = data_dir / f"{embed_type}/cath"
    gt_pdbs, gt_embeds,  = load_embeddings(gt_embed_dir)
    if embed_type == 'proteinmpnn':
        gt_embeds = np.transpose(gt_embeds, (1, 0, 2))
        gt_embeds = gt_embeds[layer]
    gt_mu = np.mean(gt_embeds, axis=0)
    gt_cov = np.cov(gt_embeds, rowvar=False)

    for i, (model_type, title) in enumerate(zip(model_types, titles)):

        samp_embed_dir = data_dir / f"{embed_type}/{data_prefix}{model_type}"

        samp_pdbs, samp_embeds = load_embeddings(samp_embed_dir)
        if embed_type == 'proteinmpnn':
            samp_embeds = np.transpose(samp_embeds, (1, 0, 2))
            samp_embeds = samp_embeds[layer]

        samp_mu = np.mean(samp_embeds, axis=0)
        samp_cov = np.cov(samp_embeds, rowvar=False)

        fpd = calculate_frechet_distance(gt_mu, gt_cov, samp_mu, samp_cov)

        if not cache:
            csv_fp = f"cache/cath_{embed_type}_{model_type}_plot_layer_{layer}.csv"

            reducer = umap.UMAP(random_state=42).fit(gt_embeds)
            gt_umap = reducer.transform(gt_embeds)
            samp_umap = reducer.transform(samp_embeds)

            df_plot = pd.DataFrame()
            gt_pdbs = [pdb.split('_')[0] for pdb in gt_pdbs]
            x = list(gt_umap[:, 0]) + list(samp_umap[:, 0])
            y = list(gt_umap[:, 1]) + list(samp_umap[:, 1])

            df_plot["UMAP 1"] = x
            df_plot["UMAP 2"] = y

            df_plot["Label"] = ["Ground Truth"] * len(gt_umap) + ["Sample"] * len(samp_umap)

            df_plot.to_csv(csv_fp, index=False)

        df_plot = pd.read_csv(csv_fp)

        sns.scatterplot(ax=axes[i], data=df_plot, x="UMAP 1", y="UMAP 2", hue="Label", alpha=0.2, s=2, palette="Set2")

        axes[i].text(x=0.95, y=0.95, s=f"FPD = {fpd:.2f}", transform=axes[i].transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

        axes[i].set_title(title, fontsize='x-large')
        axes[i].set_xlabel("UMAP 1")
        axes[i].set_ylabel("UMAP 2")
        axes[i].get_legend().remove()

    plt.tight_layout()

    plt.savefig(f"plots/umap_cath_{embed_type}_{'-'.join(model_types)}_layer_{layer}.png", dpi=300)


if __name__ == '__main__':
    typer.run(runner)

