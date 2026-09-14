from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _crop_or_pad_1d(x: tf.Tensor, target_len: int) -> tf.Tensor:
    """Ensure generator output has exact target length (crop first, then pad if needed)."""
    t = x
    tgt = tf.cast(target_len, tf.int32)
    # Crop to target length (safe even if shorter)
    t = t[:, :tgt, :]
    # Pad only when needed
    curr_len = tf.shape(t)[1]
    pad_len = tf.maximum(tgt - curr_len, 0)
    paddings = tf.convert_to_tensor([[0, 0], [0, pad_len], [0, 0]], dtype=tf.int32)
    return tf.pad(t, paddings, mode="CONSTANT")


def _conv1d_transpose(
    x: tf.Tensor, filters: int, kernel_size: int, strides: int, name: str
) -> tf.Tensor:
    x = layers.Conv1DTranspose(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        name=name,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x


def build_wavegan_generator(sample_length: int, latent_dim: int = 128) -> keras.Model:
    """
    Lightweight WaveGAN-style 1D generator.
    Pure 1D convolutions (no spectrograms).
    """
    target_len = int(sample_length)
    init_len = max(target_len // 32, 16)

    z = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(init_len * 256, name="dense_proj")(z)
    x = layers.Reshape((init_len, 256), name="reshape")(x)

    x = _conv1d_transpose(x, 256, kernel_size=25, strides=2, name="up1")
    x = _conv1d_transpose(x, 128, kernel_size=25, strides=2, name="up2")
    x = _conv1d_transpose(x, 96, kernel_size=25, strides=2, name="up3")
    x = _conv1d_transpose(x, 64, kernel_size=25, strides=2, name="up4")
    x = _conv1d_transpose(x, 32, kernel_size=25, strides=2, name="up5")

    x = layers.Conv1DTranspose(
        1, kernel_size=25, strides=2, padding="same", activation="tanh", name="output"
    )(x)
    x = layers.Lambda(lambda t: _crop_or_pad_1d(t, target_len), name="pad_or_crop")(x)
    return keras.Model(z, x, name="wavegan_generator_1d")


def build_specgan_generator(sample_length: int, latent_dim: int = 128) -> keras.Model:
    """
    SpecGAN-inspired generator but kept strictly 1D (no mel/spectrogram use).
    """
    target_len = int(sample_length)
    init_len = max(target_len // 16, 16)

    z = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(init_len * 128, name="dense_proj")(z)
    x = layers.Reshape((init_len, 128), name="reshape")(x)

    for i, filters in enumerate([128, 96, 64, 48], start=1):
        x = _conv1d_transpose(x, filters, kernel_size=15, strides=2, name=f"up{i}")

    x = layers.Conv1DTranspose(
        1, kernel_size=15, strides=2, padding="same", activation="tanh", name="output"
    )(x)
    x = layers.Lambda(lambda t: _crop_or_pad_1d(t, target_len), name="pad_or_crop")(x)
    return keras.Model(z, x, name="specgan_generator_1d")


def build_dcgan_generator(sample_length: int, latent_dim: int = 128) -> keras.Model:
    """
    Simple 1D DCGAN generator (conv-transpose stack).
    """
    target_len = int(sample_length)
    init_len = max(target_len // 8, 8)

    z = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(init_len * 128, name="dense_proj")(z)
    x = layers.Reshape((init_len, 128), name="reshape")(x)

    x = _conv1d_transpose(x, 128, kernel_size=11, strides=2, name="up1")
    x = _conv1d_transpose(x, 96, kernel_size=11, strides=2, name="up2")
    x = _conv1d_transpose(x, 64, kernel_size=11, strides=2, name="up3")

    x = layers.Conv1DTranspose(
        1, kernel_size=11, strides=2, padding="same", activation="tanh", name="output"
    )(x)
    x = layers.Lambda(lambda t: _crop_or_pad_1d(t, target_len), name="pad_or_crop")(x)
    return keras.Model(z, x, name="dcgan_generator_1d")


# --------------------------------------------------------------------------- #
# Augmentor
# --------------------------------------------------------------------------- #


@dataclass
class GANAugmentorConfig:
    method: str
    classes: Iterable[str]
    sample_length: int
    latent_dim: int = 128
    weights_dir: Optional[Path] = None


class Base1DGANAugmentor:
    """
    Thin wrapper to produce synthetic 1D signals from class-conditional generators.
    """

    def __init__(self, name: str, config: GANAugmentorConfig, builder):
        self.name = name
        self.config = config
        self.classes = list(config.classes)
        self.latent_dim = int(config.latent_dim)
        self.sample_length = int(config.sample_length)
        self.generators = {
            cls: builder(self.sample_length, self.latent_dim) for cls in self.classes
        }
        self._load_weights_if_available()

    def _load_weights_if_available(self):
        if self.config.weights_dir is None:
            return
        weights_dir = Path(self.config.weights_dir)
        for cls, gen in self.generators.items():
            candidate_paths = [
                weights_dir / f"{self.name}_{cls}.keras",
                weights_dir / f"{cls}.keras",
                weights_dir / f"{self.name}_{cls}.h5",
                weights_dir / f"{cls}.h5",
            ]
            for path in candidate_paths:
                if path.exists():
                    gen.load_weights(path)
                    break

    def sample_for_label(
        self,
        label: str,
        length: Optional[int] = None,
        base_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        target_len = int(length or self.sample_length)
        generator = self.generators.get(label) or next(iter(self.generators.values()))
        noise = tf.random.normal((1, self.latent_dim))
        # Force single-input call; ignore base_signal unless a conditional model is detected.
        try:
            if len(generator.inputs) == 1:
                out = generator(noise, training=False)
            else:
                # If a conditional generator is loaded unexpectedly, broadcast base_signal as condition.
                cond = (
                    tf.reshape(tf.convert_to_tensor(base_signal, dtype=noise.dtype), (1, -1, 1))
                    if base_signal is not None
                    else tf.zeros((1, target_len, 1), dtype=noise.dtype)
                )
                out = generator([noise, cond], training=False)
        except Exception:
            # Fallback to predict to avoid signature confusion
            if len(generator.inputs) == 1:
                out = generator.predict(noise, verbose=0)
            else:
                cond = (
                    tf.reshape(tf.convert_to_tensor(base_signal, dtype=noise.dtype), (1, -1, 1))
                    if base_signal is not None
                    else tf.zeros((1, target_len, 1), dtype=noise.dtype)
                )
                out = generator.predict([noise, cond], verbose=0)
        arr = out.numpy()[0, :, 0]
        if arr.shape[0] > target_len:
            arr = arr[:target_len]
        elif arr.shape[0] < target_len:
            pad = target_len - arr.shape[0]
            arr = np.pad(arr, (0, pad), mode="constant")
        return arr.astype(np.float32)


@dataclass
class RandomSignalAugmentorConfig:
    jitter_std: float = 0.01         # additive Gaussian noise std (relative to signal std later)
    scale_min: float = 0.8           # multiplicative scale min
    scale_max: float = 1.2           # multiplicative scale max
    shift_max_frac: float = 0.05     # max circular shift as fraction of length
    dropout_prob: float = 0.02       # probability of zeroing a sample
    cutout_frac: float = 0.05        # fraction length for contiguous mask


class RandomSignalAugmentor:
    """
    Simple, deterministic 1D signal augmentations applied on top of the real sample.
    Keeps raw 1D shape; avoids spectrogram/mel transforms.
    """

    def __init__(self, config: Optional[RandomSignalAugmentorConfig] = None):
        self.cfg = config or RandomSignalAugmentorConfig()

    def _scale(self, x: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(self.cfg.scale_min, self.cfg.scale_max)
        return x * factor

    def _jitter(self, x: np.ndarray) -> np.ndarray:
        noise_std = self.cfg.jitter_std * (np.std(x) + 1e-6)
        return x + np.random.normal(0.0, noise_std, size=x.shape)

    def _shift(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.shift_max_frac <= 0.0:
            return x
        max_shift = int(len(x) * self.cfg.shift_max_frac)
        if max_shift < 1:
            return x
        k = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, k)

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.dropout_prob <= 0.0:
            return x
        mask = np.random.rand(*x.shape) < self.cfg.dropout_prob
        if not mask.any():
            return x
        x = x.copy()
        min_val = float(np.min(x))
        x[mask] = min_val
        return x

    def _cutout(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.cutout_frac <= 0.0:
            return x
        width = int(len(x) * self.cfg.cutout_frac)
        if width < 1:
            return x
        start = np.random.randint(0, max(len(x) - width + 1, 1))
        end = min(len(x), start + width)
        x = x.copy()
        min_val = float(np.min(x))
        x[start:end] = min_val
        return x

    def augment_signal(
        self, signal: np.ndarray, label: Optional[str] = None, target_len: Optional[int] = None
    ) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float32)
        if target_len is not None and x.shape[0] != target_len:
            x = _crop_or_pad_1d(tf.convert_to_tensor(x[None, :, None]), int(target_len)).numpy()[0, :, 0]

        # Order: scale -> jitter -> shift -> dropout -> cutout
        x = self._scale(x)
        x = self._jitter(x)
        x = self._shift(x)
        x = self._dropout(x)
        x = self._cutout(x)
        return x

    # Compatibility: GAN-style API
    def sample_for_label(self, label: Optional[str], length: Optional[int] = None) -> np.ndarray:
        dummy = np.zeros(int(length or 1), dtype=np.float32)
        return self.augment_signal(dummy, label=label, target_len=length)


def create_gan_augmentor(
    method: Optional[str],
    classes: Iterable[str],
    sample_length: int,
    latent_dim: int = 128,
    weights_dir: Optional[str | Path] = None,
) -> Optional[Base1DGANAugmentor]:
    """
    Factory to build 1D augmentors. Returns None when method is falsy.
    """
    if method is None or str(method).lower() in {"none", "no", "off"}:
        return None

    method = str(method).lower()
    cfg = GANAugmentorConfig(
        method=method,
        classes=list(classes),
        sample_length=sample_length,
        latent_dim=latent_dim,
        weights_dir=Path(weights_dir) if weights_dir else None,
    )

    if method == "wavegan":
        return Base1DGANAugmentor("wavegan", cfg, build_wavegan_generator)
    if method == "specgan":
        return Base1DGANAugmentor("specgan", cfg, build_specgan_generator)
    if method in {"dcgan", "dcgan1d"}:
        return Base1DGANAugmentor("dcgan", cfg, build_dcgan_generator)
    if method in {
        "basic",
        "classic",
        "random1d",
        "random",
        "randomsignalaugmentor",
        "jitter",
    }:
        return RandomSignalAugmentor()

    raise ValueError(f"Bilinmeyen augmentasyon yöntemi: {method}")
