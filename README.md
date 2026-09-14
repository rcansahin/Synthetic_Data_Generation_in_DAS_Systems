# Synthetic Data Generation in DAS Systems

This repository explores synthetic data generation and signal modeling for distributed acoustic sensing (DAS) systems. It combines augmentation utilities, configuration files, a dummy data reader, and notebooks for encoding, time-series generation, and residual GAN-style offline excavation workflows.

## Folder contents

- `encoder.ipynb` — notebook for encoding and feature representation experiments.
- `timesnet-gen.ipynb` — notebook for time-series synthetic generation with a TimesNet-style workflow.
- `residual_gan_excavation_offline.ipynb` — notebook for residual GAN or excavation-style synthetic data generation experiments.
- `augmentations.py` — custom augmentation routines.
- `conf_mat_thr.py` — confusion matrix or threshold-related utility code.
- `config.py` — project configuration constants.
- `dummy_data_reader.py` — lightweight data reader for synthetic or dummy input samples.

## Typical workflow

1. Open the notebook that matches the intended experiment, such as `encoder.ipynb` or `timesnet-gen.ipynb`.
2. Use `config.py` and `augmentations.py` to configure and transform the synthetic data pipeline.
3. Load data through `dummy_data_reader.py` or notebook-specific cells.
4. Train, evaluate, or inspect model outputs using the GAN or time-series notebook cells.

## Purpose

The goal is to develop and study synthetic DAS data generation workflows, including augmentation, encoding, residual GAN-style exploration, and TimesNet-inspired time-series generation.
