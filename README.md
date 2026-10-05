# Interpretation of graph-based machine learning models reveals network-level patterns in the gut microbiome of inflammatory bowel diseases

Code for the manuscript of the same title.

We infer one microbial association network per individual from longitudinal gut metagenomes, train graph neural networks on these networks to distinguish IBD from non-IBD individuals, and interpret the models with Integrated Gradients to find the species and associations that carry the signal.

This repository contains code only. Data and results are not tracked.

## Repository layout

Scripts are numbered in the order they are run.

| Folder | What it does |
| --- | --- |
| [`code/01_datasets`](code/01_datasets) | Download the cohorts from `curatedMetagenomicData`, filter species and individuals, export abundance tables |
| [`code/02_networks`](code/02_networks) | Infer per-individual SparCC networks (with linear detrending) and export them |
| [`code/03_machine_learning`](code/03_machine_learning) | Build node attributes and PyTorch Geometric datasets, train the traditional and graph-based models, run Integrated Gradients |
| [`code/04_analyses`](code/04_analyses) | Diversity, network metrics, classification performance and model interpretation figures |
| [`code/05_ablation`](code/05_ablation) | Random Forest comparison, module ablation experiments, module and *E. coli* neighborhood analyses |
| [`code/common`](code/common) | Shared functions, model definitions and paths |

All input and output locations are defined in [`code/common/paths.R`](code/common/paths.R) and [`code/common/paths.py`](code/common/paths.py).

## Data

Both cohorts are public and are downloaded by [`code/01_datasets/01_download_cmd_datasets.R`](code/01_datasets/01_download_cmd_datasets.R) from the `curatedMetagenomicData v.3.14.0` R package:

- **Discovery:** HMP2/iHMP (Lloyd-Price *et al.*, 2019), `HMP_2019_ibdmdb`
- **Validation:** Hall *et al.* (2017), `HallAB_2017`

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT, see [`LICENSE`](LICENSE).
