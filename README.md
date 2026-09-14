# Synthetic Data Generation in DAS Systems

This project implements a synthetic distributed acoustic sensing (DAS) data-generation and modeling workflow based on the GAN v4.7 architecture described in the training notebook. The architecture moves away from rigid square masks and complex conflicting constraints toward a simpler KISS-style design using Gaussian soft masking, randomized erasing, spectral loss, and a diffusion-style conditional workflow.

## Folder contents

- `training_gan.ipynb` — primary notebook containing the current diffusion-style GAN pipeline, training loop, STFT spectral loss, encoder path, and visualization.
- `timesnet_encoder.py` — temporal encoder and TimesNet-style building blocks.
- `sequences_peaks.py` — sequence handling, peak extraction, peak-mask generation, and waveform utilities.
- `utils.py` — GPU configuration and shared utility functions.
- `dummy_data_reader.py` — lightweight synthetic or dummy dataset reader.
- `conf_mat_thr.py` — confusion-matrix and threshold evaluation support.

## Architecture summary

The current model version is described as GAN v4.7 with a Gaussian dynamic masking, randomized eraser, and spectral/STFT loss pipeline. It is also framed as a diffusion model architecture in the notebook.

1. Encoder (Style Extractor)
   - Input: randomly masked signal in Z-score form and a Gaussian peak mask.
   - Latent bottleneck: the latent vector is compressed from 512 dimensions down to 128 dimensions so the model focuses on the core style DNA rather than memorizing background detail.
   - Latent noise: Gaussian noise with `std=0.1` is injected into the 128-dimensional latent vector during training to smooth the latent space and reduce memorization.
   - Randomized Eraser: instead of static 61-sample masking, the DataLoader now integrates a dynamic randomized eraser (`enc_mask`) that extends randomly from the left and right of the same walking template by 20–80 units per iteration, preventing leakage and memorization.

2. Generator (Synthesizer)
   - Input: noisy `z_encoded` latent DNA and the Gaussian peak mask template.
   - Objective: synthesize realistic walking peaks dynamically while adapting the generated signal to the target mask and the surrounding background.
   - Spectral/STFT Loss: the generator is now penalized not only in the time waveform domain but also in the frequency domain using Short-Time Fourier Transform comparisons. This helps the model preserve the acoustic texture, tonality, and friction signatures of the real walking signal.
   - Soft Gaussian Masking: the target mask is no longer a hard square binary window; instead, it becomes a soft Gaussian envelope centered around the peak and fading smoothly at the edges. This reduces artificial edge spikes and ensures a smoother fade-in and fade-out behavior.
   - Dynamic Physical Loss: amplitude, energy, gradient, and STFT losses are applied directly inside the soft Gaussian mask boundaries, allowing energy to peak at the center and decay naturally through ring-down.

3. Discriminator (Conditional Judge)
   - Input: full noisy signal plus instance noise (`std=0.05`) and the Gaussian peak mask.
   - Objective: check whether the signal is a realistic DAS-like vibration and whether it matches the soft Gaussian mask.
   - Defenses: TTUR with different learning rates and label smoothing (`0.9 / 0.1`) prevent the discriminator from overpowering the generator.

## Data and modeling philosophy

- The old complex, conflicting constraints are replaced by a simpler engineering philosophy based on KISS.
- The randomized eraser mask and the Gaussian target mask are separated before entering the model so the DataLoader prevents leakage instead of carrying the complexity inside TensorFlow graph operations.
- Soft Gaussian envelopes replace square masks to match the natural acoustic decay pattern of DAS signals.
- The model has moved from strict time-only constraints to time-frequency analysis, forcing the architecture to learn not only waveform shape but also the acoustic DNA of the signal.

## Typical workflow

1. Open `training_gan.ipynb`.
2. Load the training or validation sequence data using `SequenceConfig`, `TrainSequence`, and `ValidationSequence` from `sequences_peaks.py`.
3. Generate the `enc_mask` and `peak_mask` objects in the DataLoader path and feed the masked signal and Gaussian mask into the encoder.
4. Train the conditional encoder-diffusion unet path using noise injection and STFT-aware reconstruction objectives.
5. Use the provided classifier and threshold tools if needed to validate synthetic signal realism and walking-class behavior.

## Purpose

The project demonstrates a synthetic DAS signal generation pipeline that emphasizes randomized masking, Gaussian soft masking, STFT/frequency-domain loss, diffusion-style denoising, and a noise-regularized latent style encoder path. The end goal is to synthesize realistic walking peaks while preserving the acoustic fingerprint and event structure of the raw DAS signal.

