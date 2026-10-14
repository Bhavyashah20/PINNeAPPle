# Zoo catalogue

Every pre-trained model and dataset of `pp.zoo` / `pp.datasets`, generated from the zoo manifest by `pp.zoo.catalogue()`. Each item passed the zoo's gate; the candidates that did not, and why, are in [the inventory](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/dev/model_dataset_inventory.md). How to use them: [Pre-trained models and datasets](model_zoo.md).

## Models

| Model | Version | Task | Held-out metrics (reference) | Trust report | Size | Hugging Face |
|---|---|---|---|---|---|---|
| `airfoil_cst_rans_mlp` | 1.0.0 | Airfoil aerodynamic coefficients (Cl, Cd, Cm) from shape and angle of attack, 2-D steady RANS | test/Cl_mae 0.00285, test/Cd_mape_pct 0.545, test/Cm_mae 0.000454 | APPROVED | 0.42 MB | [PINNeAPPle-Labs/airfoil-cst-rans-mlp](https://huggingface.co/PINNeAPPle-Labs/airfoil-cst-rans-mlp/tree/v1.0.0) |
| `heatsink_forced_mlp` | 1.0.0 | Plate-fin heat sink in ducted forced convection: base-to-air thermal resistance and pressure drop | test/R_mean_abs_pct 0.712, test/dP_mean_abs_pct 1.04 | APPROVED | 0.22 MB | [PINNeAPPle-Labs/heatsink-forced-mlp](https://huggingface.co/PINNeAPPle-Labs/heatsink-forced-mlp/tree/v1.0.0) |
| `heatsink_natural_mlp` | 1.0.0 | Plate-fin heat sink in natural convection (vertical fins, free air): base-to-air thermal resistance | test/R_mean_abs_pct 0.469 | APPROVED | 0.22 MB | [PINNeAPPle-Labs/heatsink-natural-mlp](https://huggingface.co/PINNeAPPle-Labs/heatsink-natural-mlp/tree/v1.0.0) |
| `rover_contact_grc1_viper` | 1.1.0 | Wheel-soil contact forces of a rigid grousered wheel on lunar regolith simulant (terramechanics) | test/Fx_rmse_N 0.22, test/Fz_rmse_N 0.47, test/My_rmse_Nm 0.0707 | APPROVED | 0.33 MB | [PINNeAPPle-Labs/rover-contact-grc1-viper](https://huggingface.co/PINNeAPPle-Labs/rover-contact-grc1-viper/tree/v1.1.0) |

Metric names are `<split>/<metric>`; the full set, the validity range and the physics checks are in each model card (`pp.zoo.info(name)`).

## Datasets

| Dataset | Version | Content | Splits (rows) | Licence | Size | Hugging Face |
|---|---|---|---|---|---|---|
| `airfoil_rans_cst_re4e6` | 1.0.0 | Airfoil RANS campaign, CST shapes, Re 4e6 (OpenFOAM, k-omega SST) | train 340, test 60, naca_reference 36, xfoil_naca2412 9, grid_study 3 | cc-by-4.0 | 29 kB (bundled) | [PINNeAPPle-Labs/airfoil-rans-cst-re4e6](https://huggingface.co/datasets/PINNeAPPle-Labs/airfoil-rans-cst-re4e6/tree/v1.0.0) |
| `heatsink_forced_convection` | 1.0.0 | Plate-fin heat sinks in ducted forced convection (6000 physics-engine solves) | test 900, cal 600, val 600, train 3900 | cc-by-4.0 | 0.75 MB (bundled) | [PINNeAPPle-Labs/heatsink-forced-convection](https://huggingface.co/datasets/PINNeAPPle-Labs/heatsink-forced-convection/tree/v1.0.0) |
| `heatsink_natural_convection` | 1.0.0 | Plate-fin heat sinks in natural convection (5000 physics-engine solves) | test 750, cal 500, val 500, train 3250 | cc-by-4.0 | 0.55 MB (bundled) | [PINNeAPPle-Labs/heatsink-natural-convection](https://huggingface.co/datasets/PINNeAPPle-Labs/heatsink-natural-convection/tree/v1.0.0) |
| `rover_contact_grc1_viper` | 1.1.0 | Wong-Reece wheel-soil contact forces, GRC-1 simulant, VIPER wheel | train 3600, test 2000 | cc-by-4.0 | 0.19 MB (bundled) | [PINNeAPPle-Labs/rover-contact-grc1-viper](https://huggingface.co/datasets/PINNeAPPle-Labs/rover-contact-grc1-viper/tree/v1.1.0) |
| `single_wheel_tests` | 1.0.0 | Published single-wheel testbed points (drawbar pull, sinkage, torque vs slip) | all 27 | mit | 3 kB (bundled) | [PINNeAPPle-Labs/single-wheel-tests](https://huggingface.co/datasets/PINNeAPPle-Labs/single-wheel-tests/tree/v1.0.0) |
| `burgers_1d_li2021` | 1.0.0 | 1-D viscous Burgers, Li et al. (2021) operator-learning benchmark (1200 trajectories, 1024 points) | train 1000, test 200 | cc-by-4.0 | 55 MB (Hub-only) | [PINNeAPPle-Labs/burgers-1d-li2021](https://huggingface.co/datasets/PINNeAPPle-Labs/burgers-1d-li2021/tree/v1.0.0) |
| `darcy_2d_li2021` | 1.0.0 | 2-D Darcy flow, Li et al. (2021) operator-learning benchmark (1200 coefficient / pressure pairs, 128 x 128) | train 1000, test 200 | cc-by-4.0 | 71 MB (Hub-only) | [PINNeAPPle-Labs/darcy-2d-li2021](https://huggingface.co/datasets/PINNeAPPle-Labs/darcy-2d-li2021/tree/v1.0.0) |

4 models and 7 datasets, 128.4 MB in total (2.7 MB bundled with the package; the rest is downloaded on first use and checked against its SHA-256).

## Cite

Each Hub card carries its own citation. For the software, cite PINNeAPPle (Zenodo DOI [10.5281/zenodo.23343062](https://doi.org/10.5281/zenodo.23343062), [CITATION.cff](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/CITATION.cff)).

