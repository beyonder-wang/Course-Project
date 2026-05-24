# Course-Project

EEG brain-signal classification project for four task-specific datasets plus one consolidated result summary. The repository is organized so visitors can inspect the final outputs first and then trace each dataset-specific workflow if needed.

## Overview

This repository contains separate subprojects for each dataset, along with the corresponding data folders, training notebooks, evaluation scripts, and final prediction files.

If you only want the final deliverables, start with [Results_Summary](Results_Summary).

## Repository Layout

- [Results_Summary](Results_Summary): final text summaries for all datasets
- [BCIC2A_code](BCIC2A_code): BCIC2A notebook and submission pipeline
- [MDD_code](MDD_code): MDD training, evaluation, and ensemble results
- [SEED_code](SEED_code): SEED training scripts, benchmark runs, and experiment outputs
- [Sleep_code](Sleep_code): SLEEP notebook-based workflow and final result file
- [data](data): shared dataset directory used by the project

## Final Results

The quickest way to inspect the project outcome is to open the dataset summaries below:

- [Results_Summary/BCIC2A.txt](Results_Summary/BCIC2A.txt)
- [Results_Summary/MDD.txt](Results_Summary/MDD.txt)
- [Results_Summary/SEED.txt](Results_Summary/SEED.txt)
- [Results_Summary/SLEEP.txt](Results_Summary/SLEEP.txt)

These files collect the final results in a compact, human-readable format.

## Dataset Workflows

### BCIC2A

The end-to-end submission notebook is [BCIC2A_code/BCIC2A_final_submission.ipynb](BCIC2A_code/BCIC2A_final_submission.ipynb). It contains the data checks, preprocessing, model selection, and submission file generation steps.

### MDD

The main result file is [MDD_code/MDD.txt](MDD_code/MDD.txt). For a more detailed look at the ensemble experiments, see [MDD_code/reports/ensemble_eval.md](MDD_code/reports/ensemble_eval.md) and [MDD_code/reports/ensemble_eval.csv](MDD_code/reports/ensemble_eval.csv).

### SEED

The SEED project is centered around the scripts in [SEED_code](SEED_code). Useful starting points include [SEED_code/AGENTS.md](SEED_code/AGENTS.md), [SEED_code/0_run_train.py](SEED_code/0_run_train.py), and the output folders under [SEED_code/Results](SEED_code/Results).

### SLEEP

The main reference files are [Sleep_code/SLEEP.txt](Sleep_code/SLEEP.txt), [Sleep_code/train_B.ipynb](Sleep_code/train_B.ipynb), and [Sleep_code/train.ipynb](Sleep_code/train.ipynb).

## Data Convention

All datasets are stored under [data](data), but did not be pushed on GitHub. The project does not rely on a separate external data root, and the notebook/script paths have already been aligned to this structure.

## Outputs

Typical experiment outputs include:

- `predictions.txt` for test-set labels
- `model.pt` for trained weights
- `config.json` for run configuration
- `metrics.json` for evaluation history
- `summary.txt` for a readable run summary
- `run.log` for the full training log

Many of these files live inside per-run folders under `Results/` or the corresponding dataset-specific output directory.

## Notes

- Prediction text files are treated as important deliverables in this repository.
- If you want to reproduce a specific dataset result, follow the notebook or script inside that dataset folder rather than editing the shared paths manually.
- The repository is structured for inspection first, then reproduction.

## Suggested Entry Point

For most visitors, the best order is:

1. Open [Results_Summary](Results_Summary).
2. Open the dataset-specific notebook or script you care about.
3. Inspect the corresponding output files in the dataset folder.
