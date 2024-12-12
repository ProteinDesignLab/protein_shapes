import subprocess
from pathlib import Path

from hydra import main
from omegaconf import DictConfig


@main(version_base=None, config_path="configs", config_name="default")
def run_protein_shapes(cfg: DictConfig):
    embedding_types = cfg.embedding_type
    output_dir = Path(cfg.output_dir)
    reference_structures = cfg.reference_structures
    sampled_structures = Path(cfg.sampled_structures)

    embed_gt = "precomputed" if not cfg.embed_reference else "to_compute"
    embed_samp = "precomputed" if not cfg.embed_samples else "to_compute"

    valid_embeddings = {"foldseek", "proteinmpnn", "prot_domain_classifier", "esm3"}
    for emb in embedding_types:
        if emb not in valid_embeddings:
            raise ValueError(
                f"Invalid embedding type: {emb}. Must be one of {valid_embeddings}."
            )

    for emb_type in embedding_types:
        print(f"Computing FPD using {emb_type} embeddings...")
        subprocess.run(
            [
                "python",
                f"src/protein_shapes/{emb_type}/{emb_type}.py",
                reference_structures,
                sampled_structures,
                output_dir / emb_type,
                embed_gt,
                embed_samp,
            ]
        )


if __name__ == "__main__":
    run_protein_shapes()
