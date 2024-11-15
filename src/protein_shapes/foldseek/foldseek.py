# generates foldseek embeddings for all pdbs in specified 1-2 directories and dumps in json by calling get_struc_seq() function from get_struc_seq.py; dumps token counts and distribution statistics in jsons by calling get_distribution_stats from eval.py

import os
import sys
from pathlib import Path

from protein_shapes.paths import DATA_DIR
from protein_shapes.foldseek.embed import get_struc_seq
from protein_shapes.foldseek.eval import get_distribution_stats


if len(sys.argv) != 6:
    print(
        "Usage: python foldseek.py <reference_structures> <sampled_structures> <output_dir> <embed_reference> <embed_samples>"
    )
    sys.exit(1)


reference_structures = Path(sys.argv[1])
sampled_structures = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
output_dir.mkdir(parents=True, exist_ok=True)
embed_reference = sys.argv[4]
embed_samples = sys.argv[5]


if str(reference_structures) == "cath":
    ref_json_fp = DATA_DIR / "foldseek" / "cath.json"
else:
    ref_json_fp = os.path.join(
        output_dir,
        f"foldseek_{(str(reference_structures)[1:] if str(reference_structures).startswith('/') else str(reference_structures)).replace('/', '-')}.json",
    )
samp_json_fp = os.path.join(
    output_dir,
    f"foldseek_{(str(sampled_structures)[1:] if str(sampled_structures).startswith('/') else str(sampled_structures)).replace('/', '-')}.json",
)


if str(reference_structures) != "cath" and embed_reference != "precomputed":
    get_struc_seq("foldseek", reference_structures, ref_json_fp)

if embed_samples != "precomputed":
    get_struc_seq("foldseek", sampled_structures, samp_json_fp)


get_distribution_stats(ref_json_fp, samp_json_fp, output_dir)
