# README figures

| Figure | Editable source | README export |
| --- | --- | --- |
| Study overview | [`study_overview.svg`](study_overview.svg) | [`study_overview.png`](study_overview.png), 3000 × 2740 |
| Selected results | [`results_summary.svg`](results_summary.svg) | [`results_summary.png`](results_summary.png), 3000 × 2080 |

The overview uses vector text, geometry and arrows, with two embedded source photographs. The small matrices and sampling tokens illustrate the training mechanisms; their colors and counts are schematic. The transformed thumbnails illustrate augmentation. The potential extensions panel is separate from the five implemented configurations.

The results figure was generated with Matplotlib from [`results.csv`](results.csv). It preserves the six aggregate result rows from the original thesis README. Each metric uses the same axis scale for both classification tasks, and every point carries its exact reported value. Horizontal bands connect each strategy to its three results. Matching circles, squares and diamonds distinguish the baseline, combined supervised training and contrastive pretraining configurations respectively.

## Photographs

Both source photographs belong to lesion **IL_9026324**, labelled **BKL** (benign keratinocytic lesion) in the official MILK10k training ground truth.

| View | Original photograph |
| --- | --- |
| Clinical close-up | [`ISIC_7780391.jpg`](examples/ISIC_7780391.jpg) |
| Dermoscopic | [`ISIC_8180214.jpg`](examples/ISIC_8180214.jpg) |

The original JPEGs are retained unchanged. Complete photographs are scaled uniformly in the dataset and classification blocks. The augmentation block additionally shows a flip, rotation and affine transformation of the dermoscopic view. The pair was selected for its discreet appearance, little visible hair and absence of an open wound, following visual review of candidates ranked by the MONET hair and ulceration/crust annotations.

Source: [official training image ZIP](https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Input.zip), [metadata](https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Metadata.csv) and [ground truth](https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_GroundTruth.csv). File identifiers and SHA-256 checksums are recorded in [`examples/manifest.csv`](examples/manifest.csv).

Attribution: **MILK study team. MILK10k. ISIC Archive, 2025. [doi:10.34970/648456](https://doi.org/10.34970/648456)**. Photographs are licensed under [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/), as recorded in their embedded XMP metadata. The repository's MIT code license does not cover these photographs.
