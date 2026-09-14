# Synthetic Data Generation in DAS Systems

This repository implements a synthetic distributed acoustic sensing (DAS) data-generation and diffusion model pipeline for the CAR class. It follows the notebook’s latest architecture definition: Diffusion v6 (Heavy-Duty 1D U-Net & Classifier-Free Guidance / SOTA Sismik Akustik Difüzyon Modeli).

The project emphasizes an iterative denoising pipeline in which the model starts from a white-noise initialization and reconstructs smooth synthetic acoustic events from noisy DAS signals. The codebase is organized around a conditional 1D U-Net, a compact latent style encoder, Gaussian masking, randomized erasing, and time-frequency signal reconstruction.

## Repository files

- `training_gan.ipynb` — main training notebook for the diffusion model, encoder, encoder-style conditioning, and iterative noise-removal workflow.
- `testing_gan.ipynb` — inference/testing notebook for generated or synthetic signal evaluation.
- `generate_synthetic.ipynb` — notebook for synthetic signal generation and sample reproduction.
- `data_generation_peaks_walk.ipynb` — notebook for data generation and peak boundary generation logic.
- `diffusion_encoder.py` — conditional 1D U-Net and diffusion encoder/model builder utilities.
- `timesnet_encoder.py` — temporal encoder and TimesNet-style feature extraction utilities.
- `sequences_peaks.py` — sequence loading, signal row loading, peak boundary generation, sequence configuration, and signal utilities.
- `synthetic_hdf5_creator.py` — synthetic HDF5 dataset creator.
- `dummy_data_reader.py` — lightweight dummy/synthetic data reader.
- `utils.py` — GPU setup and shared utility support.
- `conf_mat_thr.py` — threshold and confusion-matrix helper functions.

## Architecture summary

### 1. Encoder — Style Extractor

- Input: a randomly masked DAS signal in Z-score form and a Gaussian peak mask.
- The latent style bottleneck is preserved at 128 dimensions. This keeps the model focused on the clean acoustic background character rather than memorizing irrelevant detail.
- A classifier-free guidance update is implemented through style dropout. During training, the 128-dimensional style DNA is randomly zeroed with a probability of 10%. This enables the U-Net to learn both conditional and unconditional synthesis behavior.
- A randomized eraser is used at the data-loader level. Instead of a fixed 61-sample eraser, the encoder mask (`enc_mask`) is created by extending the signal boundary randomly on the left and right by 20–80 samples, preventing leakage and forcing the model to learn boundary-aware synthesis.

### 2. Conditional 1D U-Net — Synthesis Engine

- Input: noisy signal `x_t`, normalized timestep `t`, latent style DNA `z_encoded`, and a Gaussian mask.
- The model begins from a pure white-noise prior and denoises the signal over a full `T = 1000` diffusion schedule. It learns to remove stochastic Gaussian noise step by step and recover clean synthetic CAR acoustic events.
- Heavy-duty residual blocks replace simple convolutions. They use double 1D convolutions and FiLM-style conditioning to carry time and latent-style information through the model depth.
- The filter path is increased to `[48, 96, 192, 384]` across the residual stack.
- A bottleneck self-attention layer with 4 heads is added at the deepest point where the signal has been compressed to 625 samples. This allows the model to connect the beginning and end of the long 5000-sample signal and learn global acoustic dependencies such as echo coherence, decay, and damping behavior.

### 3. Classifier-Free Guidance (CFG)

At inference time, the diffusion model produces two predictions:

- a conditional prediction using the Gaussian template and latent style
- an unconditional prediction using a zero-style or zero-mask condition

The model then combines them using:

`pred_noise = uncond + GUIDANCE_SCALE * (cond - uncond)`

A guidance scale such as `w = 4.0` forces the model to follow the target mask more strictly. Regions outside the mask become silent or clean, while the masked area is sharpened into high-energy, high-amplitude synthetic seismic peaks.

## Data and modeling philosophy

- The older adversarial GAN setup, including discriminator instability, TTUR, and label smoothing, is removed in favor of a mathematically stable denoising diffusion scheme.
- The dataset process no longer tries to create a perfect shape in one shot. Instead, the model follows an iterative denoising process and learns natural acoustic decay and time-frequency structure through the statistics of the inverse diffusion chain.
- Hard, square masks are removed. Gaussian soft envelopes and wide 5/7 kernel convolutions smooth the synthetic signal and reduce artificial artifacts.
- The project now emphasizes time-frequency analysis rather than only time-domain signal shape. The model is forced to learn the acoustic DNA and frequency response of the target event, not just the waveform envelope.

## Workflow

1. Open `training_gan.ipynb`.
2. Load the training and validation CSV files from the DAS dataset using `SequenceConfig`, `TrainSequence`, and `ValidationSequence` from `sequences_peaks_v2.py`.
3. Generate the `enc_mask` and `peak_mask` boundaries with the sequence utilities.
4. Pass the randomly masked Z-score signal and Gaussian mask to the encoder style extractor.
5. Train the conditional 1D U-Net through the diffusion denoising loop with classifier-free guidance support.
6. Use `testing_gan.ipynb` and `generate_synthetic.ipynb` to inspect generated signals and compare them to the target distribution.
7. Export synthetic HDF5 or data summaries using `synthetic_hdf5_creator.py` when needed.

## Purpose

The repository demonstrates a modern diffusion-based synthetic DAS generation pipeline for the CAR class. It combines a latent style encoder, a heavy-duty conditional 1D U-Net, Gaussian soft masking, randomized erasing, and classifier-free guidance to synthesize acoustic events that preserve the waveform and frequency fingerprint of the original DAS data while reducing artifacts from hard square masks.
