# generates prot domain classifier embeddings in specified output directory for all pdbs in specified directory

import gzip
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

from protein_shapes.prot_domain_classifier.domain_classifier.domain_segmentor import (
    DomainSegmentor,
)


segmentor = DomainSegmentor()  # Initialize model.


def domain_runner(pdb_dir: Path, save_dir: Path):
    print(f"Processing directory: {pdb_dir}")
    for fp in tqdm(pdb_dir.glob("*.pdb")):
        save_fp = save_dir / f"{fp.stem}_domain_embedding.npy.gz"
        if not save_fp.exists():  # Check if embedding already exists
            try:
                embeds, numbering = segmentor.predict(str(fp), return_embeds=True)
                # Reshape and compute the mean embedding
                embeds = embeds.reshape(-1, 512)
                residx = np.array(numbering) != -9999
                embeds = np.mean(embeds[:, residx], axis=1)

                # Save the embedding
                with gzip.GzipFile(str(save_fp), "w") as f:
                    np.save(file=f, arr=np.half(embeds))
            except Exception as e:
                print(
                    f"Error while generating ProtDomainClassifier embedding for {save_fp}: {e}"
                )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python eval.py <path_to_pdb_directory> <path_to_save_directory>")
        sys.exit(1)

    pdb_dir = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    if not pdb_dir.is_dir():
        print(f"Error: {pdb_dir} is not a directory.")
        sys.exit(1)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    domain_runner(pdb_dir, save_dir)
