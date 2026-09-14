# Synthetic Data Generation in DAS Systems

This project implements a synthetic distributed acoustic sensing (DAS) data-generation and modeling pipeline centered on a peak-masked encoder, generator, and discriminator architecture. The workflow combines sequence preprocessing, synthetic trace augmentation, TimesNet-style encoding, and a GAN-style training loop for generating realistic DAS-like signal patterns.

## Folder contents

- `training_gan.ipynb` — main notebook for the GAN-style training pipeline and synthetic signal reconstruction workflow.
- `timesnet_encoder.py` — TimesNet-inspired encoder for sequence and temporal representation.
- `sequences_peaks.py` — sequence and peak extraction utilities used for masking and signal modeling.
- `utils.py` — shared data and plotting utilities.
- `dummy_data_reader.py` — lightweight reader for synthetic or dummy DAS inputs.
- `conf_mat_thr.py` — confusion-matrix and threshold evaluation support.
- `README.md` — project documentation.

## Peak-mask encoder-generator-discriminator pipeline

The project is organized around a three-part signal modeling flow:

1. `Peak mask` extraction and sequence preparation using `sequences_peaks.py`.
2. `Encoder` path implemented through `timesnet_encoder.py` to capture temporal structure and feature representation.
3. `Generator` and `Discriminator` training in `training_gan.ipynb` for synthetic DAS signal reconstruction or image-like waveform synthesis.

The peak mask acts as a spatial or temporal attention gate that guides the encoder-generator-discriminator loop to concentrate on the most informative DAS events rather than the full noisy background.

## Typical workflow

1. Open `training_gan.ipynb`.
2. Prepare or load synthetic sequence samples with the helper scripts and the dummy reader.
3. Apply peak masking and sequence extraction using `sequences_peaks.py`.
4. Pass the masked sequences through the TimesNet encoder representation layer.
5. Train or evaluate the generator and discriminator on the encoded signal samples.
6. Use the confusion-matrix and threshold utilities for diagnostic evaluation.

## Purpose

The repository demonstrates a synthetic DAS signal generation pipeline that emphasizes peak masking, TimesNet-style sequence encoding, and adversarial learning through an encoder-generator-discriminator structure for event-focused signal reconstruction and generation.

