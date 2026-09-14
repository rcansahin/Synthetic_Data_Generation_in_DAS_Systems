import tensorflow as tf
from tensorflow.keras import layers as L

class TimesBlock(L.Layer):
    def __init__(self, k_periyot, d_model):
        super(TimesBlock, self).__init__()
        self.k = k_periyot  # En baskın kaç periyoda bakılacağı
        self.d_model = d_model
        
        # 2D Evrişim: Reshape edilen sinyali "akustik bir resim" gibi işler
        self.conv = tf.keras.Sequential([
            L.Conv2D(d_model, kernel_size=(3, 3), padding='same', activation='relu'),
            L.BatchNormalization(),
            L.Conv2D(d_model, kernel_size=(3, 3), padding='same')
        ])
        
    def call(self, x):
        # x: (Batch, Time, Features)
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]
        
        # 1. Dinamik Padding Hesaplama
        # T'nin self.k'ya tam bölünmesi için ne kadar eklemeliyiz?
        pad_len = (self.k - (T % self.k)) % self.k
        
        # Veriyi sona sıfır ekleyerek uzatıyoruz
        # paddings: [[batch], [time_padding], [feature]]
        x_padded = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
        
        # Yeni zaman boyutu (Artık self.k'ya tam bölünür)
        T_new = T + pad_len
        period = T_new // self.k 
        
        # 2. 1D -> 2D Reshape (Artık güvenli)
        x_2d = tf.reshape(x_padded, (B, self.k, period, D))
        
        # 3. 2D Inception-like Convolution
        res = self.conv(x_2d)
        
        # 4. 2D -> 1D Geri Dönüş
        res = tf.reshape(res, (B, T_new, D))
        
        # 5. Orijinal boyuta geri kırpma (Padding'i kaldır)
        res = res[:, :T, :]
        
        return L.Add()([x, res]) # Residual connection
        
    def build_timesnet_unsupervised(window=20000, latent_dim=128):
        # INPUT: [Raw Signal, Peak Mask] -> (20000, 2)
        inputs = L.Input(shape=(window, 2), name="walking_input")
        
        # --- ENCODER ---
        # İlk projeksiyon
        x = L.Conv1D(64, kernel_size=11, padding='same')(inputs)
        
        # TimesBlocklar: DAS verisindeki farklı yürüme hızlarını yakalar
        x = TimesBlock(k_periyot=20, d_model=64)(x) # Uzun periyot
        x = TimesBlock(k_periyot=50, d_model=64)(x) # Orta periyot
        
        # Global temsil (Sinyalin karakteristiği buraya sıkışır)
        gap = L.GlobalAveragePooling1D()(x)
        latent = L.Dense(latent_dim, activation='relu', name="latent_representation")(gap)
        
        # --- GENERATOR (DECODER) ---
        # Latent vektörden sinyali tekrar inşa et
        d = L.Dense(window * 4)(latent)
        d = L.Reshape((window, 4))(d)
        
        d = TimesBlock(k_periyot=50, d_model=4)(d)
        d = TimesBlock(k_periyot=20, d_model=4)(d)
        
        # ÇIKIŞ: 2 Kanal (Sinyal ve Peak Maskesi)
        # Sinyal için Tanh, Maske için Sigmoid
        out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='tanh')(d)
        out_peak = L.Conv1D(1, kernel_size=11, padding='same', activation='sigmoid')(d)
        
        outputs = L.Concatenate(axis=-1)([out_sig, out_peak])
        
        return tf.keras.Model(inputs, outputs, name="TimesNet_Unsupervised_Generator")

    def build_timesnet_unsupervised_v2(window=20000, latent_dim=128):
        inputs = L.Input(shape=(window, 2), name="walking_input")
        
        # --- ENCODER ---
        x = L.Conv1D(64, kernel_size=11, padding='same')(inputs)
        x = TimesBlock(k_periyot=20, d_model=64)(x)
        x = TimesBlock(k_periyot=50, d_model=64)(x)
        
        gap = L.GlobalAveragePooling1D()(x)
        latent = L.Dense(latent_dim, activation='relu', name="latent_representation")(gap)
        
        # --- GENERATOR (DECODER) ---
        d = L.Dense(window * 4)(latent)
        d = L.Reshape((window, 4))(d)
        d = TimesBlock(k_periyot=50, d_model=4)(d)
        d = TimesBlock(k_periyot=20, d_model=4)(d)
        
        # İki Ayrı Çıktı Kanallı Yapı
        out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='tanh', name="signal_output")(d)
        out_peak = L.Conv1D(1, kernel_size=11, padding='same', activation='sigmoid', name="peak_output")(d)
        
        # Model artık bir liste döndürüyor
        return tf.keras.Model(inputs, [out_sig, out_peak], name="TimesNet_V2")
    
    def build_v3_pro(window=20000, latent_dim=512):
            inputs = L.Input(shape=(window, 2), name="walking_input")
            
            # ENCODER (Downsampling ile özellik yakalama)
            x = L.Conv1D(64, kernel_size=25, strides=4, padding='same')(inputs)
            x = L.BatchNormalization()(x)
            x = L.LeakyReLU(0.2)(x)
            
            for p in [10, 50, 100]: # Farklı periyotlar
                x = TimesBlock(k_periyot=p, d_model=64)(x)
                
            x = L.GlobalMaxPooling1D()(x) # Detayları koruyan havuzlama
            latent = L.Dense(latent_dim, activation='relu')(x)
            
            # DECODER (Upsampling ile sinyal inşa etme)
            d = L.Dense(5000 * 64)(latent)
            d = L.Reshape((5000, 64))(d)
            
            # 5000 -> 10000 -> 20000
            for f in [64, 32]:
                d = L.Conv1DTranspose(f, kernel_size=25, strides=2, padding='same')(d)
                d = L.BatchNormalization()(d)
                d = L.LeakyReLU(0.2)(d)
                d = TimesBlock(k_periyot=50, d_model=f)(d)
                
            out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='tanh', name="signal_output")(d)
            out_peak = L.Conv1D(1, kernel_size=11, padding='same', activation='sigmoid', name="peak_output")(d)
            
            return tf.keras.Model(inputs, [out_sig, out_peak])
        
    def build_v3_sf(window=5000, latent_dim=512):
        inputs = L.Input(shape=(window, 2), name="walking_input")
        
        # --- ENCODER ---
        # Giriş: 5000 -> Çıkış: 625 (8 kat küçülme)
        x = L.Conv1D(64, kernel_size=41, strides=8, padding='same')(inputs)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU(0.2)(x)
        
        for p in [10, 50]:
            x = TimesBlock(k_periyot=p, d_model=64)(x)
            
        x = L.GlobalMaxPooling1D()(x)
        latent = L.Dense(latent_dim, activation='relu', name="latent")(x)
        
        # --- DECODER (Dinamik Hesaplama) ---
        # Bottleneck boyutu girişin tam 8'de biri olmalı
        bottleneck_size = window // 8  # 5000 / 8 = 625
        
        d = L.Dense(bottleneck_size * 128)(latent)
        d = L.Reshape((bottleneck_size, 128))(d)
        
        # Kademeli Upsampling (Toplam 8 kat büyüme: 2 * 2 * 2 = 8)
        # 625 -> 1250 -> 2500 -> 5000
        for f in [128, 64, 32]:
            d = L.Conv1DTranspose(f, kernel_size=11, strides=2, padding='same')(d)
            d = L.LeakyReLU(0.2)(d)
            d = TimesBlock(k_periyot=50, d_model=f)(d)
        
        # DİKKAT: Sondaki strides=10 katmanını sildik! 
        # Çünkü o katman 5.000'i 50.000 yapıyordu.
        
        # Attention
        q = L.Dense(32)(d)
        v = L.Dense(32)(d)
        d = L.Attention()([q, v])
        
        # Çıkış Katmanları (Artık boyutu [Batch, 5000, 1])
        out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='tanh', name="signal_output")(d)
        out_peak = L.Conv1D(1, kernel_size=11, padding='same', activation='sigmoid', name="peak_output")(d)
        
        return tf.keras.Model(inputs, [out_sig, out_peak])
    
    def build_encoder_v4(window=5000, latent_dim=512):
        inputs = L.Input(shape=(window, 2), name="signal_input_only") # Sadece sinyal

        x = L.Conv1D(64, kernel_size=41, strides=8, padding='same')(inputs)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU(0.2)(x)
        
        for p in [10, 50]:
            x = TimesBlock(k_periyot=p, d_model=64)(x)
            
        x = L.GlobalMaxPooling1D()(x)
        latent = L.Dense(latent_dim, activation='relu', name="embedding")(x)
        
        return tf.keras.Model(inputs, latent, name="Encoder_V4")
    
    def build_generator_v4(window=5000, latent_dim=512):
        # İki Giriş: 1. Encoder'dan gelen embedding, 2. Senin vereceğin Peak Maskesi
        latent_in = L.Input(shape=(latent_dim,), name="latent_in")
        peak_in = L.Input(shape=(window, 1), name="peak_info_input") 
        
        # 1. Embedding'i sinyal iskeletine dönüştür
        bottleneck_size = window // 8 # 625
        d = L.Dense(bottleneck_size * 128)(latent_in)
        d = L.Reshape((bottleneck_size, 128))(d)
        
        # 2. Upsampling (625 -> 5000)
        for f in [128, 64, 32]:
            d = L.Conv1DTranspose(f, kernel_size=11, strides=2, padding='same')(d)
            d = L.LeakyReLU(0.2)(d)
            d = TimesBlock(k_periyot=50, d_model=f)(d)
        
        # 3. KRİTİK ADIM: Üretilen iskelet ile Peak Bilgisini birleştir (Concatenate)
        # d şu an (5000, 32), peak_in ise (5000, 1). Birleşince (5000, 33) olur.
        merged = L.Concatenate(axis=-1)([d, peak_in])
        
        # Attention katmanını birleşmiş veri üzerinde çalıştırıyoruz
        q = L.Dense(33)(merged)
        v = L.Dense(33)(merged)
        d_final = L.Attention()([q, v])
        
        # Sinyal Çıkışı: Artık peak noktalarına göre şekillenmiş bir yürüme sinyali
        out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='linear', name="gen_signal")(d_final)
        
        return tf.keras.Model([latent_in, peak_in], out_sig, name="Generator_V4")

    def build_generator_v4_3(window=10000, latent_dim=128):
        # Girişler
        latent_in = L.Input(shape=(latent_dim,), name="latent_in")
        peak_in = L.Input(shape=(window, 1), name="peak_info_input") 
        
        # 1. Başlangıç (10000 / 16 = 625)
        # 10.000 uzunluğa ulaşmak için 4 kez strides=2 kullanacağız.
        bottleneck_size = window // 16 
        d = L.Dense(bottleneck_size * 128)(latent_in)
        d = L.Reshape((bottleneck_size, 128))(d)
        
        # --- İLK UPSAMPLING: 625 -> 1250 ---
        d = L.Conv1DTranspose(128, kernel_size=11, strides=2, padding='same')(d)
        d = L.LeakyReLU(0.2)(d)
        
        # --- ATTENTION STRATEJİSİ (Hafıza Dostu) ---
        # 10.000'lik maskeyi 1.250'ye düşürerek Jeneratöre "bak buraya peak gelecek" diyoruz
        peak_downsampled = L.AveragePooling1D(pool_size=8)(peak_in) # 10000 / 8 = 1250
        
        # Düşük çözünürlükte birleştir (1250, 128 + 1)
        merged_low_res = L.Concatenate(axis=-1)([d, peak_downsampled])
        
        # Attention'ı 1250 uzunluğunda çalıştır (1250x1250 matris GPU'yu yormaz)
        q = L.Dense(64)(merged_low_res)
        v = L.Dense(64)(merged_low_res)
        d = L.Attention()([q, v])
        d = TimesBlock(k_periyot=50, d_model=64)(d) # Fiziksel sinyal blokları
    
        # --- UPSAMPLING DEVAM: 1250 -> 10000 ---
        # Sırasıyla: 2500, 5000, 10000
        for f in [64, 32, 16]:
            d = L.Conv1DTranspose(f, kernel_size=11, strides=2, padding='same')(d)
            d = L.LeakyReLU(0.2)(d)
            # Sadece 2500 ve 5000 aşamalarında TimesBlock kullanabiliriz (opsiyonel)
            if f > 16:
                d = TimesBlock(k_periyot=50, d_model=f)(d)
    
        # 3. NİHAİ BİRLEŞTİRME (10.000 seviyesinde)
        # Artık attention değil, sadece basit bir birleştirme ve son fırça darbesi
        final_merged = L.Concatenate(axis=-1)([d, peak_in])
        
        # Sinyal Çıkışı
        out_sig = L.Conv1D(1, kernel_size=11, padding='same', activation='linear', name="gen_signal")(final_merged)
        
        return tf.keras.Model([latent_in, peak_in], out_sig, name="Generator_V4_3")
        
    def build_discriminator_v4(window=5000):
        inputs = L.Input(shape=(window, 1), name="disc_input")
        
        x = L.Conv1D(32, kernel_size=21, strides=4, padding='same')(inputs)
        x = L.LeakyReLU(0.2)(x)
        x = L.Conv1D(64, kernel_size=21, strides=4, padding='same')(x)
        x = L.LeakyReLU(0.2)(x)
        
        x = L.GlobalMaxPooling1D()(x)
        out = L.Dense(1, activation='sigmoid', name="real_fake_score")(x)
        
        return tf.keras.Model(inputs, out, name="Discriminator_V4")

    def build_discriminator_v4_1(window=5000):
        inputs = L.Input(shape=(window, 2), name="disc_input")
    
        # 1. Aşama: Sinyale Geniş Bakış
        x = L.Conv1D(64, kernel_size=41, strides=4, padding='same')(inputs)
        x = L.LeakyReLU(0.2)(x)
        x = L.Dropout(0.3)(x) # Hakemin ezberlemesini engellemek için Dropout şart!
        
        # 2. Aşama: Jeneratörün Silahıyla Vurma (TimesBlock)
        # Hakem de periyotları ve frekansları inceleyebilsin
        x = TimesBlock(k_periyot=50, d_model=64)(x)
        
        # 3. Aşama: Derinleşme ve Detay Avı
        x = L.Conv1D(128, kernel_size=21, strides=4, padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU(0.2)(x)
        x = L.Dropout(0.3)(x)
        
        # 4. Aşama: İnce Frekans Dokusu Analizi
        x = L.Conv1D(256, kernel_size=11, strides=4, padding='same')(x)
        x = L.BatchNormalization()(x)
        x = L.LeakyReLU(0.2)(x)
        
        # 5. Karar Aşaması
        x = L.GlobalMaxPooling1D()(x)
        
        x = L.Dense(64)(x)
        x = L.LeakyReLU(0.2)(x)
        out = L.Dense(1, activation='sigmoid', name="real_fake_score")(x)
        
        return tf.keras.Model(inputs, out, name="Discriminator_V4_1")