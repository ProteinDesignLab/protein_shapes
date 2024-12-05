# SHAPES

Structural and Hierarchical Assessment of Proteins with Embedding Similarity.

## Install

Clone the repo, then

```
cd protein_shapes
conda env create -n protein-shapes --file environment.yaml
pip install -e .
```

The environment was created with

```
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install numpy scipy scikit-learn pandas
conda install -c conda-forge -c bioconda foldseek
pip install ProDy hydra-core ml-collections dm-tree tqdm matplotlib seaborn typer-cli umap-learn
pip install -e .
```

which can be adapted for different PyTorch / CUDA versions.

## Usage

Specify the embedding type to use and paths to reference/sampled structures in `configs/default.yaml`, then run with

```
python run.py
```

The provided embeddings are `"prot_domain_classifier", "proteinmpnn", "foldseek",`, spanning different protein structure hierarchies, from global features by the domain classifier to residue environments with varying sizes by ProteinMPNN encoder layers and strictly local features with Foldseek tokens.

Save time by setting `embed_reference: False` to reuse precomputed reference structure embeddings or `embed_samples: False` to reuse precomputed sampled structure embeddings.

Precomputed statistics for CATH are stored in `data/` and can be loaded and used as the reference statistics by setting `reference_structures: cath`.

Outputs are logged in the specified `output_dir`.

## Plots

For continuous embeddings, visualize sampled structures overlayed on reference set structures with UMAP with 
```
python examples/plot_embeddings.py
```
E.g. to visualize ProteinMPNN embeddings at the first layer, use
```
python examples/plot_embeddings.py --embed-type proteinmpnn --layer 0
```
Plots are by default saved in `plots/`.

## Acknowledgements

 - [ProtDomainSegmentor](https://github.com/egurapha/prot_domain_segmentor)
    ```
    @article{eguchi2020multi,
    title={Multi-scale structural analysis of proteins by deep semantic segmentation},
    author={Eguchi, Raphael R and Huang, Po-Ssu},
    journal={Bioinformatics},
    volume={36},
    number={6},
    pages={1740--1749},
    year={2020},
    publisher={Oxford University Press}
    }

    ```
 - [ProteinMPNN](https://github.com/dauparas/LigandMPNN)
    ```
    @article{dauparas2022robust,
    title={Robust deep learning--based protein sequence design using ProteinMPNN},
    author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
    journal={Science},
    volume={378},
    number={6615},
    pages={49--56},
    year={2022},
    publisher={American Association for the Advancement of Science}
    }
    ```
 - [Foldseek](https://www.nature.com/articles/s41587-023-01773-0)
    ```
    @article{van2024fast,
    title={Fast and accurate protein structure search with Foldseek},
    author={Van Kempen, Michel and Kim, Stephanie S and Tumescheit, Charlotte and Mirdita, Milot and Lee, Jeongjae and Gilchrist, Cameron LM and S{\"o}ding, Johannes and Steinegger, Martin},
    journal={Nature biotechnology},
    volume={42},
    number={2},
    pages={243--246},
    year={2024},
    publisher={Nature Publishing Group US New York}
    }

    ```
