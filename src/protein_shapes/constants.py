from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent  # directory from git clone

DATA_DIR = ROOT_DIR / "data"

PLOT_DIR = ROOT_DIR / "plots"

EMBED_DIR = ROOT_DIR / "embeddings"

SOURCE_DIR = Path(__file__).parent  # src/protein_shapes
