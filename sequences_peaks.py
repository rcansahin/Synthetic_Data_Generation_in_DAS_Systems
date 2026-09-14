from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional
import pywt
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence, to_categorical
from scipy import signal
REQUIRED_COLUMNS = ["file", "channel", "event", "window_start", "window_end", "peaks"]

# def _generate_peak_mask(peaks_str, window_size, min_radius=15, max_radius=100):
#     """
#     Rastgele yayılım genişliğinde maske üretir. 
#     Radius 15-50 arası demek, toplam genişlik 30-100 indeks arası olacak demektir.
#     """
#     mask = np.zeros((window_size, 1), dtype=np.float32)
#     if pd.isna(peaks_str) or peaks_str == "":
#         return mask
    
#     try:
#         peak_indices = [int(p) for p in str(peaks_str).split(",") if p.strip()]
#         for p in peak_indices:
#             if p < window_size:
#                 # Hocanın istediği tamamen rastgele maske genişliği
#                 random_sigma = np.random.randint(min_radius, max_radius + 1)
                
#                 start = max(0, p - random_sigma)
#                 end = min(window_size, p + random_sigma + 1)
#                 mask[start:end] = 1.0
#     except ValueError:
#         pass
#     return mask
import numpy as np
import pandas as pd

def _generate_peak_mask(peaks_str, window_size, event_type="walking", min_width=120, max_width=160):
    """
    Sınıfa (event_type) göre koşullandırma maskesi ve encoder körleştirme silgisi üretir.
    1. Araçlar (car/vehicle) için: İlk iki değeri (başlangıç, bitiş) devasa blok aralık olarak kullanır.
    2. Yürüme/Kazı için: Değişken zaman ölçekli keskin (Square) ve asimetrik maske üretir.
    """
    peak_mask = np.zeros((window_size, 1), dtype=np.float32)
    enc_mask = np.zeros((window_size, 1), dtype=np.float32)
    
    if pd.isna(peaks_str) or peaks_str == "":
        return peak_mask, enc_mask
    
    try:
        # Gelen veriyi virgül ile parçalayıp tam sayı listesi yapıyoruz
        parts = [int(p) for p in str(peaks_str).split(",") if p.strip()]
        if len(parts) == 0:
            return peak_mask, enc_mask
        
        # =========================================================
        # --- A. ARAÇ GEÇİŞİ (CAR / VEHICLE) MANTIĞI ---
        # =========================================================
        if "car" in event_type:
            if len(parts) >= 2:
                # Virgülle ayrılmış string'in ilk iki elemanı başlangıç ve bitiştir (Örn: 5000, 15000)
                v_start = parts[0]
                v_end = parts[1]
                
                valid_start = max(0, v_start)
                valid_end = min(window_size, v_end)
                
                if valid_end > valid_start:
                    # 1. Jeneratör için Devasa Keskin Blok Maske (1.0)
                    peak_mask[valid_start:valid_end] = 1.0
                    
                    # 2. Encoder için Körleştirme 
                    # Araç enerjisi geniş yayıldığı için padding daha büyük tutulur (200-600)
                    random_padding = np.random.randint(200, 600) 
                    enc_start = max(0, v_start - random_padding)
                    enc_end = min(window_size, v_end + random_padding)
                    enc_mask[enc_start:enc_end] = 1.0

        # =========================================================
        # --- B. YÜRÜME VE KAZI (WALKING / DIGGING) MANTIĞI ---
        # =========================================================
        else:
            for p in parts:
                if p < window_size:
                    # Gaussian (Çan Eğrisi) yerine, CFG ile tam uyumlu çalışan 
                    # "Değişken Genişlikli Asimetrik Keskin Maske" (Square) kullanıyoruz.
                    width = np.random.randint(min_width, max_width + 1)
                    left_w = int(width * 0.85)  # Asimetrik: Topuk %85
                    right_w = int(width * 0.15) # Asimetrik: Burun %15
                    
                    start = p - left_w
                    end = p + right_w
                    
                    valid_start = max(0, start)
                    valid_end = min(window_size, end)
                    
                    if valid_end > valid_start:
                        # 1. Jeneratör için Keskin Maske (Genlik her zaman 1.0)
                        peak_mask[valid_start:valid_end] = 1.0
                        
                    # 2. Encoder için Rastgele Körleştirme (Padding: 20-80)
                    random_padding = np.random.randint(20, 81)
                    enc_start = max(0, p - left_w - random_padding)
                    enc_end = min(window_size, p + right_w + random_padding)
                    enc_mask[enc_start:enc_end] = 1.0
                    
    except ValueError:
        pass
        
    return peak_mask, enc_mask
    
def _generate_tukey_peak_mask(peaks_str, window_size, min_width=120, max_width=160):
    """
    1. Jeneratör için Tukey (Flat-Top Gaussian) hedef maskesi üretir.
    2. Encoder için RASTGELE boyutta, geniş bir körleştirme silgisi üretir.
    """
    peak_mask = np.zeros((window_size, 1), dtype=np.float32)
    enc_mask = np.zeros((window_size, 1), dtype=np.float32)
    
    if pd.isna(peaks_str) or peaks_str == "":
        return peak_mask, enc_mask
    
    try:
        peak_indices = [int(p) for p in str(peaks_str).split(",") if p.strip()]
        for p in peak_indices:
            if p < window_size:
                # --- A. TUKEY PEAK MASK (Jeneratör İçin) ---
                # Test scriptiyle tamamen aynı genişlik ve asimetri sınırları
                width = np.random.randint(min_width, max_width + 1)
                
                # Asimetrik ayak izi (Topuk %85, Burun %15)
                left_w = int(width * 0.85)
                right_w = int(width * 0.15)
                
                start = p - left_w
                end = p + right_w
                
                valid_start = max(0, start)
                valid_end = min(window_size, end)
                actual_width = valid_end - valid_start
                
                if actual_width > 1:
                    # alpha=0.4 ile %60 Kutu, %40 Yumuşak Sönümlenen Tukey Window oluştur
                    tapered_box = signal.windows.tukey(actual_width, alpha=0.4).reshape(-1, 1)
                    
                    # Sınırların 1.0'ı geçmemesini sağla
                    tapered_box = tapered_box * 1.0
                    
                    # Maskeye ekle (Üst üste binerse max olanı al)
                    peak_mask[valid_start:valid_end] = np.maximum(
                        peak_mask[valid_start:valid_end], tapered_box
                    )
                else:
                    peak_mask[p] = 1.0
                    
                # --- B. RASTGELE KÖRLEŞTİRME MASKESİ (Encoder İçin) ---
                # Tukey şablonunun sağından ve solundan 
                # rastgele 20 ile 80 birim arası ekstra silgi ekliyoruz.
                random_padding = np.random.randint(20, 81)
                enc_start = max(0, p - left_w - random_padding)
                enc_end = min(window_size, p + right_w + random_padding)
                
                enc_mask[enc_start:enc_end] = 1.0
                
    except ValueError:
        pass
        
    return peak_mask, enc_mask
def normalize_walking_data(x_batch):
    """
    x_batch: (Batch, 20000, 2)
    Kanal 0: Ham Sinyal -> [-1, 1] arasına çekilir.
    Kanal 1: Peak Maskesi -> Zaten 0-1 arasında.
    """
    x_norm = x_batch.copy()
    signals = x_norm[:, :, 0]
    
    # Her örneği kendi içinde normalize et (Batch-wise Min-Max)
    s_min = np.min(signals, axis=1, keepdims=True)
    s_max = np.max(signals, axis=1, keepdims=True)
    
    # 0'a bölme hatasını engelle
    denom = s_max - s_min + 1e-9
    # Sinyali [-1, 1] arasına çekiyoruz (tanh'a uygun)
    normalized_signals = 2 * (signals - s_min) / denom - 1
    
    x_norm[:, :, 0] = normalized_signals
    return x_norm
    
def normalize_batch(x_batch):
    # Kanal 0: Sinyal -> [-1, 1]
    signals = x_batch[:, :, 0]
    s_min = np.min(signals, axis=1, keepdims=True)
    s_max = np.max(signals, axis=1, keepdims=True)
    x_batch[:, :, 0] = 2 * (signals - s_min) / (s_max - s_min + 1e-9) - 1
    return x_batch
    
def _wavelet_filter(data):
    wavelet = 'sym5'
    maxlev = 5
    coeffs = pywt.wavedec(data, wavelet, level=maxlev, mode='symmetric')
    # Detay katsayılarını (gürültüyü) thresholding ile temizle
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [coeffs[0]]
    for i in range(1, maxlev + 1):
        new_coeffs.append(pywt.threshold(coeffs[i], uthresh, mode='soft'))
    return pywt.waverec(new_coeffs, wavelet, mode='symmetric')
    
def _load_sequence_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """
    Load metadata CSV and retain the required five columns. Prefer named columns,
    otherwise fall back to the first five columns in order.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df[REQUIRED_COLUMNS].copy()
    if df.shape[1] >= len(REQUIRED_COLUMNS):
        trimmed = df.iloc[:, : len(REQUIRED_COLUMNS)].copy()
        trimmed.columns = REQUIRED_COLUMNS
        return trimmed
    raise ValueError(
        f"{csv_path} beklenen kolonları içermiyor. En az {len(REQUIRED_COLUMNS)} kolon olmalı."
    )


def _slice_or_pad_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Trim/pad 1D signal to the desired window length."""
    sig = signal
    if sig.shape[0] > window_size:
        sig = sig[:window_size]
    if sig.shape[0] < window_size:
        missing = window_size - sig.shape[0]
        min_value = sig.min() if sig.size > 0 else 0.0
        sig = np.pad(sig, (0, missing), "constant", constant_values=(min_value))
    return sig


def _load_signal_from_row(row: pd.Series, window_size: int) -> np.ndarray:
    """
    Read a single-channel P-polarization magnitude slice from an HDF5 row description.
    Keeps the exact 1D workflow (no spectrogram/mel usage).
    """
    file = row["file"]
    channel = str(row["channel"])
    window_start = int(row["window_start"])
    window_end = int(row["window_end"])

    with h5py.File(file, "r") as f:
        s = f[channel][:]["P"]

        if "raw_augment" not in str(file):
            if s.dtype.names and "re" in s.dtype.names and "im" in s.dtype.names:
                s_re = s["re"][window_start:window_end]
                s_im = s["im"][window_start:window_end]
                s = np.hypot(s_re, s_im)
            else:
                s = s[window_start:window_end]
            #s = _wavelet_filter(s)
            s = _slice_or_pad_signal(s, window_size)
        else:
            # Augmented files are assumed to be already time-aligned; pad for safety.
            s = _slice_or_pad_signal(np.asarray(s), window_size)

    return s

def _(row: pd.Series, window_size: int) -> np.ndarray:
    file = row["file"]

@dataclass
class SequenceConfig:
    csv_path: Path
    batch_size: int
    n_classes: int
    window_size: int
    shuffle: bool = True
    balance: bool = False
    include_meta: bool = False
    augmentor: Optional[Any] = None  # expects sample_for_label(label, length) -> np.ndarray
    augment_prob: float = 0.0


class BaseSignalSequence(Sequence):
    """
    Shared 1D signal sequence (train/val/test) with optional GAN-based augmentation.
    Does not touch underlying CSV/HDF5 files, only reads slices.
    """

    def __init__(self, config: SequenceConfig):
        self.config = config
        self.data = _load_sequence_dataframe(config.csv_path)
        self.data_size = self.data.shape[0]
        self.encoder = LabelEncoder().fit(self.data.event)

        self.indices = np.arange(self.data_size)
        self._rebalance_if_needed()
        if self.config.shuffle:
            np.random.shuffle(self.indices)

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @property
    def window_size(self) -> int:
        return self.config.window_size

    @property
    def n_classes(self) -> int:
        return self.config.n_classes

    @property
    def n_batches(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def __len__(self) -> int:
        return self.n_batches

    # ---- Augmentation helpers -------------------------------------------------
    def _maybe_augment(self, label: str, base_signal: np.ndarray) -> Optional[np.ndarray]:
        """Call augmentor if enabled; supports both sample_for_label and augment_signal APIs."""
        if self.config.augmentor is None or self.config.augment_prob <= 0.0:
            return None
        if np.random.rand() > self.config.augment_prob:
            return None

        try:
            if hasattr(self.config.augmentor, "augment_signal"):
                generated = self.config.augmentor.augment_signal(
                    base_signal, label=label, target_len=self.window_size
                )
            elif hasattr(self.config.augmentor, "sample_for_label"):
                try:
                    generated = self.config.augmentor.sample_for_label(
                        label, self.window_size, base_signal=base_signal
                    )
                except TypeError:
                    generated = self.config.augmentor.sample_for_label(label, self.window_size)
            else:
                return None

            if generated is None:
                return None
            generated = np.asarray(generated).reshape(-1)
            return _slice_or_pad_signal(generated, self.window_size)
        except Exception as exc:  # keep training running if augmentation fails
            print(f"Augmentation skipped for label {label}: {exc}")
            return None

    # ---- Data loading ---------------------------------------------------------
    def _load_batch_indices(self, index: int) -> Iterable[int]:
        start = index * self.batch_size
        end = min(len(self.indices), (index + 1) * self.batch_size)
        if start >= end:
            raise IndexError(
                f"Index {index} out of bounds for indices of size {len(self.indices)}"
            )
        return self.indices[start:end]

    def __getitem__(self, index: int):
        batch_x, batch_y_class, batch_y_peaks = [], [], []
        files, channels = [], []

        for ind in self._load_batch_indices(index):
            row = self.data.iloc[ind]
            event = row["event"]
            peaks_str = row.get("peaks", "")

            try:
                # 1. Ham Sinyali Yükle
                signal = _load_signal_from_row(row, self.window_size).astype(np.float32)
                
                # 2. Peak Maskesini (Ground Truth) Oluştur
                # Bu maske hem girişte kanal olacak hem de çıkışta hedef (target)
                peak_mask, enc_mask = _generate_peak_mask(peaks_str, self.window_size, event_type=event)
            except (KeyError, OSError, ValueError) as e:
                print(f"Dosya hatası ({row['file']}): {e}")
                continue

            # Augmentation (Yalnızca sinyal kanalına uygulanır)
            augmented = self._maybe_augment(event, signal)
            if augmented is not None:
                signal = augmented

            # --- GÜNCELLEME: İKİ KANALLI GİRİŞ (INPUT) OLUŞTURMA ---
            # signal: (20000,) , peak_mask: (20000, 1) 
            # Önce sinyali (20000, 1) yapalım, sonra yan yana koyalım
            signal_2d = np.expand_dims(signal, axis=-1)
            triple_channel_input = np.concatenate([signal_2d, peak_mask, enc_mask], axis=-1) # Shape: (20000, 2)

            batch_x.append(triple_channel_input)
            batch_y_class.append(event)
            batch_y_peaks.append(peak_mask)
            
            files.append(row["file"])
            channels.append(row["channel"])

        # x Shape: (Batch, Window, 2)
        x = np.array(batch_x)
        
        # Sınıf Etiketi (Classification)
        y_enc = self.encoder.transform(batch_y_class)
        y_class = to_categorical(y_enc, num_classes=self.n_classes)
        
        # Peak Etiketi (Segmentation Target)
        y_peaks = np.array(batch_y_peaks) # Shape: (Batch, Window, 1)

        # Çoklu çıktı (Multi-output) desteği
        targets = {"class_output": y_class, "peak_output": y_peaks}

        if self.config.include_meta:
            return x, targets, files, channels
        return x, targets

    # ---- Epoch housekeeping ---------------------------------------------------
    def _rebalance_if_needed(self):
        if not self.config.balance:
            return
        counts = self.data["event"].value_counts()
        if counts.empty:
            return
        min_samples = counts.min()
        balanced_indices = []
        for cls in self.data["event"].unique():
            class_data = self.data[self.data["event"] == cls]
            if len(class_data) > min_samples:
                sampled = np.random.choice(class_data.index, size=min_samples, replace=False)
            else:
                sampled = class_data.index
            balanced_indices.extend(sampled)
        self.indices = np.array(balanced_indices)

    def on_epoch_end(self):
        self._rebalance_if_needed()
        if self.config.shuffle:
            np.random.shuffle(self.indices)


class TrainSequence(BaseSignalSequence):
    """Balanced + shuffled sequence for training."""

    def __init__(self, config: SequenceConfig):
        cfg = replace(config, shuffle=True, balance=True, include_meta=False)
        super().__init__(cfg)


class ValidationSequence(BaseSignalSequence):
    """No balancing, optional shuffle off by default."""

    def __init__(self, config: SequenceConfig):
        cfg = replace(config, shuffle=False, balance=False, include_meta=False)
        super().__init__(cfg)


class TestSequence(BaseSignalSequence):
    """No balancing, returns file/channel metadata for downstream evaluation."""

    def __init__(self, config: SequenceConfig):
        cfg = replace(config, shuffle=False, balance=False, include_meta=True)
        super().__init__(cfg)

