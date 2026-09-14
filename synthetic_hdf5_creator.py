import os
import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt  # Çizimler için eklendi
from timesnet_encoder import TimesBlock
from diffusion_encoder import build_conditional_unet_1d
from scipy import signal
from tqdm import tqdm
from utils import setup_gpu

setup_gpu(memory_fraction=0.7)

def _slice_or_pad_signal(s, window_size=10000):
    if len(s) > window_size:
        return s[:window_size]
    elif len(s) < window_size:
        return np.pad(s, (0, window_size - len(s)), mode='constant')
    return s

def generate_random_peak_mask(window_size, min_peaks=8, max_peaks=15):
    """Her kanal için rastgele sayılarda ve konumlarda yürüme maskesi üretir."""
    peak_mask = np.zeros((1, window_size, 1), dtype=np.float32)
    
    # Sinyalin başından ve sonundan 500 sample boşluk bırakarak rastgele noktalar seç
    num_peaks = np.random.randint(min_peaks, max_peaks + 1)
    adim_noktalari = np.random.choice(range(500, window_size - 500), size=num_peaks, replace=False)
    adim_noktalari = np.sort(adim_noktalari) # Sıralı olması estetik açıdan iyidir
    for p in adim_noktalari:
        width = np.random.randint(120, 160) 
        
        start = max(0, p - int(width * 0.85))
        end = min(window_size, p + int(width * 0.15))
        actual_width = end - start
        
        if actual_width <= 0:
            continue
            
        tapered_box = signal.windows.tukey(actual_width, alpha=0.4)
        peak_mask[0, start:end, 0] = np.maximum(peak_mask[0, start:end, 0], tapered_box)
    # for p in adim_noktalari:
    #     width = np.random.randint(120, 160) 
        
    #     start = max(0, p - int(width * 0.85))
    #     end = min(window_size, p + int(width * 0.15))
    #     if end > start:
    #         # Tukey vb. yok, dümdüz keskin 1.0 veriyoruz (Square Masking)
    #         peak_mask[0, start:end, 0] = 1.0

    return tf.convert_to_tensor(peak_mask, dtype=tf.float32), peak_mask

# --- 1. GİRDİ VE ÇIKTI PARAMETRELERİ ---
MODEL_FOLDER = "20260605-DiffGAN-v4.7.2"
ENC_WEIGHT_PATH = os.path.join(MODEL_FOLDER, "weights", "encoder_epoch_500.weights.h5")
GEN_WEIGHT_PATH = os.path.join(MODEL_FOLDER, "weights", "diffusion_unet_epoch_500.weights.h5")

# SESSİZ_DATA_PATH = "/home/fotas/data/segment/2023.06.04/record_CIFTLIK_SESSIZ_202306040006_30min_raw.bin.hdf5"
SESSİZ_DATA_PATH = "/tf/start_training/TIMESNET_AUG/record_tpao_sessiz_202607081119_raw.hdf5"
OUTPUT_HDF5_PATH = os.path.join(MODEL_FOLDER, "synthetic_walking_tpao_ch500-600.hdf5")

WINDOW_START = 0
WINDOW_SIZE = 5000
WINDOW_END = WINDOW_START + WINDOW_SIZE
T = 1000
CFG_SCALE = 4.0  # +++ CFG Yönlendirme Gücü +++

# --- 2. MODELLERİ İNŞA ET ---
print("Modeller inşa ediliyor ve ağırlıklar yükleniyor...")
encoder = TimesBlock.build_encoder_v4(window=WINDOW_SIZE, latent_dim=128)
diffusion_unet = build_conditional_unet_1d(window=WINDOW_SIZE, latent_dim=128)

encoder.load_weights(ENC_WEIGHT_PATH)
diffusion_unet.load_weights(GEN_WEIGHT_PATH)
print("Ağırlıklar başarıyla yüklendi! ✅")

# DİFÜZYON TAKVİMİ (Sadece 1 kere hesaplanır)
betas = np.linspace(1e-4, 0.02, T, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

# --- 3. BATCH GENERATION (KANALLAR ÜZERİNDE DÖNGÜ) ---
print(f"\n🚀 Sentetik Üretim Başlıyor! Çıktı Dosyası: {OUTPUT_HDF5_PATH}")

with h5py.File(SESSİZ_DATA_PATH, "r") as f_in, h5py.File(OUTPUT_HDF5_PATH, "w") as f_out:
    
    # --- FOTAS HDF5 ZORUNLU METADATA (ÖZNİTELİKLER) ---
    f_out.attrs["polarization"] = 1
    f_out.attrs["port"] = 1
    f_out.attrs["prf"] = 2000
    f_out.attrs["duration"] = WINDOW_SIZE / f_out.attrs["prf"]
    f_out.attrs["channels"] =  101
    f_out.attrs["resolution"] = 0.68        
    f_out.attrs["pulsewidth"] = 25       
    f_out.attrs["sampletype"] = "RealUInt16" 
    
    HDF5_OPTS = {'compression': 'gzip', 'compression_opts': 9, 'shuffle': True}
    
    # 130'dan 150'ye kadar olan kanalları dön (150 dahil)
    for ch in range(500, 601):
        channel_str = str(ch)
        
        if channel_str not in f_in:
            print(f"⚠️ Kanal {channel_str} kaynak dosyada bulunamadı, atlanıyor...")
            continue
            
        print(f"\n--- İşlenen Kanal: {channel_str} ---")
        
        # 3A. Veriyi Oku
        s = f_in[channel_str][:]["P"]
        if s.dtype.names and "re" in s.dtype.names and "im" in s.dtype.names:
            sessiz_ham = np.hypot(s["re"][WINDOW_START:WINDOW_END], s["im"][WINDOW_START:WINDOW_END])
        else:
            sessiz_ham = s[WINDOW_START:WINDOW_END]
            
        sessiz_ham = _slice_or_pad_signal(sessiz_ham, WINDOW_SIZE)
        sessiz_ham = sessiz_ham.reshape(1, WINDOW_SIZE, 1).astype(np.float32)
        
        # 3B. Z-Score Normalizasyonu
        mean = np.mean(sessiz_ham, axis=1, keepdims=True)
        std = np.std(sessiz_ham, axis=1, keepdims=True) + 1e-8
        sessiz_norm = (sessiz_ham - mean) / std
        
        # 3C. Bu Kanala Özel Rastgele Maske Üret
        peak_mask_tf, peak_mask_np = generate_random_peak_mask(WINDOW_SIZE, min_peaks=8, max_peaks=15)
        
        # 3D. Encoder ile Stil Analizi
        encoder_input = tf.concat([sessiz_norm, peak_mask_tf], axis=-1)
        z_encoded_numpy = encoder.predict(encoder_input, verbose=0)
        z_encoded = tf.convert_to_tensor(z_encoded_numpy, dtype=tf.float32)
        
        # 3E. Difüzyon Döngüsü (CFG Destekli)
        x = tf.random.normal(shape=(1, WINDOW_SIZE, 1))
        
        for t in tqdm(reversed(range(T)), total=T, desc=f"Ch:{channel_str} Denoising", leave=False):
            t_normalized = tf.ones((1, 1), dtype=tf.float32) * (t / float(T))
            
            # CFG için hem koşullu hem koşulsuz tahmin
            pred_noise_cond = diffusion_unet([x, t_normalized, z_encoded, peak_mask_tf], training=False)
            pred_noise_uncond = diffusion_unet([x, t_normalized, tf.zeros_like(z_encoded), tf.zeros_like(peak_mask_tf)], training=False)
            
            pred_noise = pred_noise_uncond + CFG_SCALE * (pred_noise_cond - pred_noise_uncond)
            
            alpha_t = alphas[t]
            alpha_t_cumprod = alphas_cumprod[t]
            beta_t = betas[t]
            
            if t > 0:
                noise = tf.random.normal(shape=tf.shape(x))
            else:
                noise = tf.zeros_like(x)
                
            x = (1.0 / np.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / np.sqrt(1.0 - alpha_t_cumprod)) * pred_noise) + np.sqrt(beta_t) * noise

        # 3F. Üretilen Veriyi Orijinal Genliğine Çevir
        fake_yurume_norm = x.numpy()
        fake_yurume_orijinal = (fake_yurume_norm * std) + mean
        signal_1d = fake_yurume_orijinal[0, :, 0]
        
        # --- ÜRETİMİ GÖRSELLEŞTİRME (PLOT) ---
        plt.figure(figsize=(12, 4))
        plt.plot(signal_1d, label=f"Synthetic Signal (Channel {channel_str})", color='#1f77b4')
        
        # Maskeyi sinyalin genliğine göre ölçeklendirip gölge olarak ekliyoruz
        max_amp = np.max(np.abs(signal_1d))
        mask_1d = peak_mask_np[0, :, 0]
        plt.fill_between(range(WINDOW_SIZE), -max_amp * mask_1d, max_amp * mask_1d, 
                         color='red', alpha=0.2, label='Peak Mask Template')
        
        plt.title(f"Channel {channel_str} - Synthetic Signal after Diffusion")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.show()  # Çizimi ekranda göster
        save_path = os.path.join(MODEL_FOLDER, f"tpao_synthetic_walking_ch{channel_str}.png")
        plt.savefig(save_path, bbox_inches='tight')
        # 3G. Çıktıları FOTAS Yapısal Dizi Formatında Kaydet
        dt = np.dtype([('P', np.float32)])
        structured_signal = np.zeros(signal_1d.shape, dtype=dt)
        structured_signal['P'] = signal_1d
        
        f_out.create_dataset(channel_str, data=structured_signal, **HDF5_OPTS)

    # 4. Döngü bittikten sonra Labels'ı BOŞ olarak kaydet
    f_out.create_dataset('labels', shape=(0, 6), dtype='u4', **HDF5_OPTS)

print(f"\n🎉 Tüm kanallar başarıyla üretildi, görselleştirildi ve FOTAS formatında kaydedildi!")
print(f"📁 Dosya konumu: {OUTPUT_HDF5_PATH}")