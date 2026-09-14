# Synthetic Data Generation in DAS Systems

This project implements a synthetic distributed acoustic sensing (DAS) data-generation and modeling workflow based on a GAN-style encoder-generator-discriminator pipeline with a peak-mask anti-leakage design. The model receives noisy raw DAS signals, masks and suppresses peak locations through a widened peak mask, compresses the remaining background into a latent style vector, and then reconstructs synthetic walking peaks through an energy-aware generator.

## Folder contents

- `training_gan.ipynb` — main notebook for the encoder-generator-discriminator training loop, losses, synthetic reconstruction, and visualization.
- `timesnet_encoder.py` — TimesNet-style encoder and temporal representation builder.
- `sequences_peaks.py` — sequence and peak extraction utilities, including the peak mask generation and wavelet denoising routine.
- `utils.py` — shared utilities for GPU setup and result support.
- `dummy_data_reader.py` — synthetic or dummy data loading support.
- `conf_mat_thr.py` — confusion-matrix and threshold evaluation utilities.

## Architecture summary

The implemented architecture follows the notebook’s model version tag: GAN v4.6, an energy-aware style transfer and inpainting workflow built around a reference-conditioned conditional GAN.

1. Encoder (Style Extractor)
   - Input: masked signal in Z-score form and the peak mask.
   - Goal: remove walking peak information from the real signal by zeroing the peak region with a 31-sample expanded mask. The encoder only sees the silent or masked background and compresses it into a 512-dimensional latent style DNA vector `z_encoded`.

2. Generator (Synthesizer)
   - Input: `z_encoded` and the peak mask.
   - Goal: build realistic peaks in the masked areas using the style signal from the encoder.
   - Improvement: an energy loss and amplitude loss are used to prevent generator regression to low-energy or flat outputs. The generator is encouraged to create aggressive, high-amplitude synthetic peaks.

3. Discriminator (Conditional Judge)
   - Input: full signal plus instance noise and the peak mask.
   - Goal: judge whether the signal appears realistic in DAS format and whether the event timing is plausible.
   - Defensive design: TTUR, label smoothing, and instance noise are used to prevent the discriminator from overpowering the generator.

## Data and visualization logic

- The model is trained on raw noisy DAS signals so that the classifier can recognize the desired class behavior correctly.
- Visualization uses wavelet denoising `_wavelet_filter` to make the generated patterns easier to inspect at the end of each epoch.

## Typical workflow

1. Open `training_gan.ipynb`.
2. Load sequence data and train/validation CSV files using `sequences_peaks.py` and `dummy_data_reader.py`.
3. Generate or update the peak mask and convert the signal into the masked encoder input.
4. Train the encoder-generator-discriminator loop with the TimesNet encoder and generator/discriminator builders.
5. Use the classifier and threshold utilities for final assessment of the synthetic signal quality.

## Purpose

The repository demonstrates a peak-masked synthetic DAS signal generation pipeline built around an encoder-generator-discriminator design, with emphasis on anti-leakage masking, high-energy peak reconstruction, classifier-aware signal quality, and wavelet-assisted visualization.

