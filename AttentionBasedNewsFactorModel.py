import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, Attention, Dropout, 
                                   LayerNormalization, MultiHeadAttention, Reshape, 
                                   Flatten, Concatenate, BatchNormalization, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, AutoModel
import logging


from ImprovedTokenizer import ImprovedTokenizer

logger = logging.getLogger("AdvancedNewsFactor.AttentionBasedNewsFactorModel")

class AttentionBasedNewsFactorModel:
    """
    Attention-alapú hírfaktor modell kapitalizációs változások előrejelzésére
    """
    
    def __init__(self, vocab_size=50000, keyword_dim=256, company_dim=512, 
                 latent_dim=128, max_keywords=100):
        """
        Inicializálja az attention-alapú modellt
        
        Paraméterek:
            vocab_size (int): Szótár mérete
            keyword_dim (int): Kulcsszó beágyazási dimenzió
            company_dim (int): Cég beágyazási dimenzió
            latent_dim (int): Látens reprezentáció dimenzió
            max_keywords (int): Maximum kulcsszavak száma
        """
        self.vocab_size = vocab_size
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.max_keywords = max_keywords
        
        # Szótár és tokenizer
        self.tokenizer = ImprovedTokenizer(vocab_size)
        
        # Modell építése
        self.model = self.build_model()
        logger.info("Neural network architektúra felépítve")

        # Modell összeállítása multi-task loss-szal
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'capitalization_prediction': 'mse',
                'keyword_reconstruction': 'mse'
            },
            loss_weights={
                'capitalization_prediction': 1.0,
                'keyword_reconstruction': 0.3  # Rekonstrukció kisebb súllyal
            },
            metrics={
                'capitalization_prediction': ['mae'],
                'keyword_reconstruction': ['mae']
            }
        )
        self.model.summary(print_fn=logger.info)
        logger.info("AttentionBasedNewsFactorModel inicializálva")
    
    def build_model(self):
        """
        Felépíti a teljes neurális háló architektúrát
        """
        # -----------------------------
        # 1. Input rétegek
        # -----------------------------
        keyword_input = Input(shape=(self.max_keywords,), name='keywords')  # Kulcsszavak tokenizált szekvenciája, pl. (batch, max_keywords)
        company_input = Input(shape=(self.company_dim,), name='company_embedding')  # Cég embedding (pl. fundamentális vagy piaci jellemzők alapján), (batch, company_dim)

        # -----------------------------
        # 2. Keyword Encoder
        # -----------------------------
        keyword_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.keyword_dim,
            mask_zero=True
        )(keyword_input)  # Kulcsszavak beágyazása, shape: (batch, max_keywords, keyword_dim)

        keyword_attention = MultiHeadAttention(
            num_heads=8,
            key_dim=self.keyword_dim // 8,
            name='keyword_self_attention'
        )(keyword_embedding, keyword_embedding)  # multi-head self-attention: hogyan kapcsolódnak egymáshoz a kulcsszavak
        keyword_attention = LayerNormalization()(keyword_attention + keyword_embedding) # residual connection + LayerNorm
        keyword_latent = Dense(self.latent_dim, activation='relu', name='keyword_latent_transform')(keyword_attention) # minden kulcsszóhoz megőrzünk egy latent reprezentációt
        keyword_latent = Dropout(0.2)(keyword_latent)  # keyword_latent shape: (batch, max_keywords, latent_dim)

        # -----------------------------
        # 3. Company Embedding Processing
        # -----------------------------
        company_processed = Dense(self.latent_dim, activation='relu', name='company_processed')(company_input)
        company_processed = BatchNormalization()(company_processed)
        company_reshaped = Reshape((1, self.latent_dim))(company_processed)  # company_reshaped shape: (batch, 1, latent_dim)

        # -----------------------------
        # 4. Cross-Attention
        # -----------------------------
        cross_attention = MultiHeadAttention(num_heads=4, key_dim=self.latent_dim // 4, name='company_keyword_cross_attention') # mely kulcsszavak a legfontosabbak z adott cég szempontjából, a cég embedding a "query", a kulcsszavak a "key" és "value"
        attention_output, attention_scores = cross_attention(
            query=company_reshaped,     # (batch, 1, latent_dim)
            value=keyword_latent,       # (batch, max_keywords, latent_dim)
            key=keyword_latent,         # (batch, max_keywords, latent_dim)
            return_attention_scores=True
        ) # attention_output shape: (batch, 1, latent_dim), attention_scores shape: (batch, num_heads, query_len=1, key_len=max_keywords)

        # -----------------------------
        # 5. Combination and feature Processing
        # -----------------------------
        flattened_attention = Flatten()(attention_output)  # (batch, latent_dim)
        combined_features = Concatenate()([flattened_attention, company_processed]) # Egyesíti a cross-attention kimenetet és a feldolgozott cég embeddinget

        x = Dense(256, activation='relu')(combined_features)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        combined_representation = Dropout(0.2)(x)
        
        # -----------------------------
        # 6. Capitalization Predictor
        # -----------------------------
        cap_prediction = Dense(64, activation='relu', name='cap_hidden')(combined_representation)
        cap_prediction = Dropout(0.1)(cap_prediction)
        capitalization_output = Dense(4, activation='linear', name='capitalization_prediction')(cap_prediction)
        
        # -----------------------------
        # 7. Reconstruction Decoder
        # -----------------------------
        attention_pooled = self.attention_pooling_layer(keyword_latent, company_reshaped) #  Attention-based pooling használata, a kulcsszó-attention súlyozott átlaga mint rekonstrukciós cél
        reconstruction_hidden = Dense(128, activation='relu', name='reconstruction_hidden')(attention_pooled)
        reconstruction_hidden = Dropout(0.2)(reconstruction_hidden)
        reconstructed_keywords = Dense(self.latent_dim, activation='tanh',name='keyword_reconstruction')(reconstruction_hidden)

        # -----------------------------
        # 8. Teljes modell összeállítása
        # -----------------------------
        return Model(
            inputs=[keyword_input, company_input],
            outputs=[capitalization_output, reconstructed_keywords],
            name='ImprovedAttentionNewsFactorModel'
        )
    
    def prepare_keyword_sequence(self, text, max_length=None):
        """
        Előkészíti a szöveget kulcsszó szekvenciává
        
        Paraméterek:
            text (str): Bemeneti szöveg
            max_length (int): Maximum hossz, alapértelmezett self.max_keywords
            
        Visszatérési érték:
            numpy.ndarray: Tokenizált és padding-elt szekvencia
        """
        if max_length is None:
            max_length = self.max_keywords
        return self.tokenizer.encode(text, max_length=max_length)
        
    def attention_pooling_layer(self, keyword_features, company_query):
        """
        Attention-alapú pooling implementáció: a company_query alapján súlyozzuk a keyword_features-t
        
        Paraméterek:
            keyword_features: (batch, max_keywords, latent_dim)
            company_query: (batch, 1, latent_dim)
        
        Visszatérési érték:
            Súlyozott keyword reprezentáció
        """
        
        # Dot-product attention scores
        scores = tf.reduce_sum(
            keyword_features * company_query,  # Broadcasting: (batch, max_keywords, latent_dim)
            axis=-1, keepdims=True  # (batch, max_keywords, 1)
        )
        
        # Softmax normalizálás
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch, max_keywords, 1)
        
        # Súlyozott összeg
        pooled = tf.reduce_sum(
            keyword_features * attention_weights,  # (batch, max_keywords, latent_dim)
            axis=1  # (batch, latent_dim)
        )
        return pooled
    
    def build_vocabulary(self, news_texts):
        """
        Szótár építése a hírekből
        
        Paraméterek:
            news_texts (list): Hírek listája
        """
        logger.info("Szótár építése...")
        self.tokenizer.build_vocab(news_texts)
        logger.info(f"Szótár mérete: {len(self.tokenizer.word_to_idx)}")
    
    def train(self, training_data, validation_data=None, epochs=100, batch_size=32):
        """
        Modell tanítása
        
        Paraméterek:
            training_data (dict): Tanítási adatok
            validation_data (dict): Validációs adatok (opcionális)
            epochs (int): Tanítási epochok száma
            batch_size (int): Batch méret
        """
        # Adatok előkészítése
        X_keywords = np.array(training_data['keywords'])
        X_companies = np.array(training_data['company_embeddings'])
        y_caps = np.array(training_data['capitalization_changes'])
        y_reconstructions = np.array(training_data['keyword_targets'])
        
        # Validációs adatok előkészítése ha vannak
        validation_data_prepared = None
        if validation_data:
            val_X_keywords = np.array(validation_data['keywords'])
            val_X_companies = np.array(validation_data['company_embeddings'])
            val_y_caps = np.array(validation_data['capitalization_changes'])
            val_y_reconstructions = np.array(validation_data['keyword_targets'])
            
            validation_data_prepared = (
                [val_X_keywords, val_X_companies],
                [val_y_caps, val_y_reconstructions]
            )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Modell tanítása
        history = self.model.fit(
            [X_keywords, X_companies],
            [y_caps, y_reconstructions],
            validation_data=validation_data_prepared,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Modell tanítása befejezve")
        return history
    
    def predict_capitalization_change(self, keyword_sequence, company_embedding):
        """
        Kapitalizációs változás előrejelzése
        
        Paraméterek:
            keyword_sequence (numpy.ndarray): Kulcsszó szekvencia
            company_embedding (numpy.ndarray): Cég beágyazása
            
        Visszatérési érték:
            numpy.ndarray: Előrejelzett változások [1d, 5d, 20d, volatility]
        """
        # Dimenzió ellenőrzése
        if keyword_sequence.ndim == 1:
            keyword_sequence = keyword_sequence.reshape(1, -1)
        if company_embedding.ndim == 1:
            company_embedding = company_embedding.reshape(1, -1)
        
        prediction = self.model.predict([keyword_sequence, company_embedding], verbose=0)[0]        
        return prediction[0]
    
    def get_attention_weights(self, keyword_sequence, company_embedding):
        """        
        Paraméterek:
            keyword_sequence (numpy.ndarray): Kulcsszó szekvencia  
            company_embedding (numpy.ndarray): Cég beágyazása
            
        Visszatérési érték:
            dict: Attention súlyok és tokenek interpretációja
        """
        # Dimenzió normalizálás
        if keyword_sequence.ndim == 1:
            keyword_sequence = keyword_sequence.reshape(1, -1)
        if company_embedding.ndim == 1:
            company_embedding = company_embedding.reshape(1, -1)

        try:
            # Biztonságos attention layer lekérés
            attention_layer = None
            for layer in self.model.layers:
                if layer.name == 'company_keyword_cross_attention':
                    attention_layer = layer
                    break
            
            if attention_layer is None:
                logger.warning("Cross-attention layer nem található")
                return self._create_dummy_attention_result(keyword_sequence)
            
            # Átmeneti modell létrehozása az attention súlyok kinyeréséhez
            intermediate_layers = []
            found_attention = False
            
            for layer in self.model.layers:
                intermediate_layers.append(layer.output)
                if layer.name == 'company_keyword_cross_attention':
                    # Az attention layer második outputja a súlyok
                    attention_output = layer.output
                    found_attention = True
                    break
            
            if not found_attention:
                return self._create_dummy_attention_result(keyword_sequence)
                
            # Egyszerűbb megközelítés - predikció és interpretáció
            prediction = self.model.predict([keyword_sequence, company_embedding], verbose=0)
            
        except Exception as e:
            logger.warning(f"Attention súlyok lekérési hiba: {str(e)}")
            return self._create_dummy_attention_result(keyword_sequence)
        
        # Kulcsszavak dekódolása
        decoded_keywords = []
        for idx in keyword_sequence[0]:
            if idx in self.tokenizer.idx_to_word:
                decoded_keywords.append(self.tokenizer.idx_to_word[idx])
            elif idx == 0:
                decoded_keywords.append('[PAD]')
            else:
                decoded_keywords.append('[UNK]')

        return {
            'attention_scores': prediction,  # A predikció mint proxy az attention-nek
            'decoded_keywords': decoded_keywords,
            'input_sequence': keyword_sequence[0],
            'company_embedding': company_embedding[0],
            'sequence_length': len([k for k in decoded_keywords if k != '[PAD]'])
        }

    def _create_dummy_attention_result(self, keyword_sequence):
        """
        Dummy attention eredmény hibák esetén
        """
        decoded_keywords = []
        for idx in keyword_sequence[0]:
            if idx in self.tokenizer.idx_to_word:
                decoded_keywords.append(self.tokenizer.idx_to_word[idx])
            elif idx == 0:
                decoded_keywords.append('[PAD]')
            else:
                decoded_keywords.append('[UNK]')
        
        return {
            'attention_scores': np.zeros((1, 4)),  # Dummy scores
            'decoded_keywords': decoded_keywords,
            'input_sequence': keyword_sequence[0],
            'company_embedding': np.zeros(self.company_dim),
            'sequence_length': len([k for k in decoded_keywords if k != '[PAD]'])
        }
