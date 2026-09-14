import tensorflow as tf
from tensorflow.keras import layers as L

# --- YARDIMCI BLOK: 1D RESNET BLOĞU ---
def residual_block_1d(x, filters, cond_emb):
    """
    Hem sinyali işleyen hem de Zaman+Stil (cond_emb) bilgisini 
    sinyalin içine derinlemesine enjekte eden Residual Blok.
    """
    # Shortcut (Atlama bağlantısı) - Boyut uyuşmazlığı varsa eşitle
    shortcut = x if x.shape[-1] == filters else L.Conv1D(filters, 1, padding='same')(x)

    # 1. Konvolüsyon
    x = L.GroupNormalization(groups=8)(x)
    x = L.Activation('swish')(x)
    x = L.Conv1D(filters, kernel_size=5, padding='same')(x)

    # Koşul (Condition) Enjeksiyonu
    c = L.Dense(filters)(cond_emb)
    c = L.Reshape((1, filters))(c)
    x = L.Add()([x, c])

    # 2. Konvolüsyon
    x = L.GroupNormalization(groups=8)(x)
    x = L.Activation('swish')(x)
    x = L.Conv1D(filters, kernel_size=5, padding='same')(x)

    # Çıkışta orijinal sinyali ekle (Residual mantığı)
    return L.Add()([shortcut, x])


# --- YARDIMCI BLOK: 1D SELF-ATTENTION ---
def attention_block_1d(x, filters):
    """
    Modelin 5000 birimlik uzun sinyalin farklı noktaları arasındaki 
    akustik ilişkiyi (örneğin yankıların uyumunu) görmesini sağlar.
    """
    shortcut = x
    x = L.GroupNormalization(groups=8)(x)
    # 4 Kafalı (Head) Öz-Dikkat mekanizması
    x = L.MultiHeadAttention(num_heads=4, key_dim=filters)(x, x)
    return L.Add()([shortcut, x])


# --- ANA MİMARİ: HEAVY-DUTY CONDITIONAL U-NET ---
def build_conditional_unet_1d(window=5000, latent_dim=128):
    noisy_signal_in = L.Input(shape=(window, 1), name="noisy_signal")
    time_step_in = L.Input(shape=(1,), name="time_step")
    latent_style_in = L.Input(shape=(latent_dim,), name="latent_style")
    peak_mask_in = L.Input(shape=(window, 1), name="peak_mask")
    
    # 1. ZAMAN (TIME) EMBEDDING (Difüzyonun kalbi)
    t_emb = L.Dense(256, activation="swish")(time_step_in)
    t_emb = L.Dense(256)(t_emb)
    
    # 2. STİL DNA EMBEDDING
    style_emb = L.Dense(256, activation="swish")(latent_style_in)
    
    # Zaman ve Stili birleştir
    cond_emb = L.Add()([t_emb, style_emb])
    
    # Gürültülü sinyal ile hedef Gaussian şablonunu birleştir
    x = L.Concatenate(axis=-1)([noisy_signal_in, peak_mask_in]) 
    
    # İlk genişletme (Sinyali ağın içine al)
    x = L.Conv1D(64, kernel_size=7, padding='same')(x)
    
    skips = []
    # Daha derin ve daha geniş filtreler!
    filters = [64, 128, 256, 512]
    
    # --- ENCODER (AŞAĞI İNİŞ) ---
    for i, f in enumerate(filters):
        # Her seviyede 2 adet ResNet bloğu
        x = residual_block_1d(x, f, cond_emb)
        x = residual_block_1d(x, f, cond_emb)
        
        # Sadece dar alanlarda (256 ve 512 filtrelerde) Attention kullan (OOM yememek için)
        if f >= 256:
            x = attention_block_1d(x, f)
            
        skips.append(x)
        
        # Son katman hariç MaxPool ile boyutu küçült
        if i < len(filters) - 1:
            x = L.MaxPooling1D(pool_size=2)(x)
            
    # --- BOTTLENECK (DARBOĞAZ - EN DERİN NOKTA) ---
    x = residual_block_1d(x, 512, cond_emb)
    x = attention_block_1d(x, 512)
    x = residual_block_1d(x, 512, cond_emb)
    
    # --- DECODER (YUKARI ÇIKIŞ) ---
    for i, (f, skip) in enumerate(zip(reversed(filters), reversed(skips))):
        # Son katman hariç UpSample ile boyutu büyüt
        if i > 0:
            x = L.UpSampling1D(size=2)(x)
            
        x = L.Concatenate(axis=-1)([x, skip]) # Skip connection
        
        # Yine her seviyede 2 adet ResNet bloğu
        x = residual_block_1d(x, f, cond_emb)
        x = residual_block_1d(x, f, cond_emb)
        
        if f >= 256:
            x = attention_block_1d(x, f)

    # --- ÇIKIŞ (NOISE TAHMİNİ) ---
    x = L.GroupNormalization(groups=8)(x)
    x = L.Activation('swish')(x)
    # Sinyale eklenen o incecik gürültüyü lineer olarak tahmin et
    predicted_noise = L.Conv1D(1, kernel_size=5, padding='same', activation='linear', name="pred_noise")(x)
    
    return tf.keras.Model(
        inputs=[noisy_signal_in, time_step_in, latent_style_in, peak_mask_in], 
        outputs=predicted_noise, 
        name="Heavy_Conditional_Diffusion_UNet"
    )