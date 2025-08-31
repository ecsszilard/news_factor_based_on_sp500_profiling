import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, Dropout, LayerNormalization, MultiHeadAttention, Reshape, Flatten, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging

from BiDirectionalAttentionLayer import BiDirectionalAttentionLayer
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
        
        # Financial impact tracker inicializálása
        self.financial_tracker = FinancialImpactTracker()
        
        # Modell építése
        self.model = self.build_model()
        logger.info("Neural network architektúra felépítve")

        # Modell összeállítása multi-task loss-szal
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'capitalization_prediction': 'mse',
                'keyword_reconstruction': 'mse',
                'financial_attention_loss': 'categorical_crossentropy'
            },
            loss_weights={
                'capitalization_prediction': 1.0,
                'keyword_reconstruction': 0.3,
                'financial_attention_loss': 0.2
            },
            metrics={
                'capitalization_prediction': ['mae'],
                'keyword_reconstruction': ['mae'],
                'financial_attention_loss': ['accuracy']
            }
        )
        
        logger.info("AttentionBasedNewsFactorModel inicializálva")

    def build_model(self):
        """
        Felépíti a bi-directional attention architektúrát
        """
        # -----------------------------
        # 1. Input rétegek
        # -----------------------------
        keyword_input = Input(shape=(self.max_keywords,), name='keywords')  # Kulcsszavak tokenizált szekvenciája, pl. (batch, max_keywords)
        company_input = Input(shape=(self.company_dim,), name='company_embedding')  # Cég embedding (pl. fundamentális vagy piaci jellemzők alapján), (batch, company_dim)
        financial_weights_input = Input(shape=(self.max_keywords,), name='financial_weights')

        # -----------------------------
        # 2. Keyword Encoder
        # -----------------------------
        keyword_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.keyword_dim,
            mask_zero=True
        )(keyword_input)  # Kulcsszavak beágyazása, shape: (batch, max_keywords, keyword_dim)

        keyword_self_attention = MultiHeadAttention(
            num_heads=8,
            key_dim=self.keyword_dim // 8,
            name='keyword_self_attention'
        )(keyword_embedding, keyword_embedding)  # multi-head self-attention: hogyan kapcsolódnak egymáshoz a kulcsszavak
        keyword_attention = LayerNormalization()(keyword_self_attention + keyword_embedding) # residual connection + LayerNorm
        keyword_latent = Dense(self.latent_dim, activation='relu', name='keyword_latent_transform')(keyword_attention) # minden kulcsszóhoz megőrzünk egy latent reprezentációt
        keyword_latent = Dropout(0.2)(keyword_latent)  # keyword_latent shape: (batch, max_keywords, latent_dim)

        # -----------------------------
        # 3. Company Embedding Processing
        # -----------------------------
        company_processed = Dense(self.latent_dim, activation='relu', name='company_processed')(company_input)
        company_processed = BatchNormalization()(company_processed)
        company_reshaped = Reshape((1, self.latent_dim))(company_processed)  # company_reshaped shape: (batch, 1, latent_dim)

        # -----------------------------
        # 4. Bi-directional attention layer
        # -----------------------------
        bidirectional_attention = BiDirectionalAttentionLayer(
            latent_dim=self.latent_dim,
            num_heads=4,
            name='bidirectional_attention'
        )
        
        attention_results = bidirectional_attention(
            [company_reshaped, keyword_latent],
            financial_weights=financial_weights_input
        )

        # -----------------------------
        # 5. Combination and feature Processing
        # -----------------------------
        flattened_attention = Flatten()(attention_results['combined_output'])  # (batch, latent_dim)
        combined_features = Concatenate()([flattened_attention, company_processed]) # egyesíti a cross-attention kimenetet és a feldolgozott cég embeddinget

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
        attention_pooled = self.attention_pooling_layer(attention_results['weighted_keywords'], company_reshaped) #  Attention-based pooling használata, a kulcsszó-attention súlyozott átlaga mint rekonstrukciós cél
        reconstruction_hidden = Dense(128, activation='relu', name='reconstruction_hidden')(attention_pooled)
        reconstruction_hidden = Dropout(0.2)(reconstruction_hidden)
        reconstructed_keywords = Dense(self.latent_dim, activation='tanh',name='keyword_reconstruction')(reconstruction_hidden)

        # Financial attention prediction
        financial_attention_pred = Dense(64, activation='relu')(combined_representation)
        financial_attention_pred = Dropout(0.1)(financial_attention_pred)
        financial_attention_output = Dense(self.max_keywords, activation='softmax', name='financial_attention_loss')(financial_attention_pred)
        
        # -----------------------------
        # 8. Teljes modell összeállítása
        # -----------------------------
        return Model(
            inputs=[keyword_input, company_input, financial_weights_input],
            outputs=[capitalization_output, reconstructed_keywords, financial_attention_output],
            name='BiDirectionalNewsFactorModel'
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
        
    def attention_pooling_layer(self, weighted_keywords, company_query):
        """
        Attention pooling company_query és weighted_keywords alapján
        
        Paraméterek:
            weighted_keywords: (batch, max_keywords, latent_dim)
            company_query: (batch, 1, latent_dim)
        
        Visszatérési érték:
            Súlyozott keyword reprezentáció
        """
        
        # Dot-product attention scores
        scores = tf.reduce_sum(
            weighted_keywords * company_query,  # Broadcasting: (batch, max_keywords, latent_dim)
            axis=-1, keepdims=True  # (batch, max_keywords, 1)
        )
        
        # Softmax normalizálás
        attention_weights = tf.nn.softmax(scores, axis=1)  # (batch, max_keywords, 1)
        
        # Súlyozott összeg
        pooled = tf.reduce_sum(
            weighted_keywords * attention_weights,  # (batch, max_keywords, latent_dim)
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
    
    def train(self, training_data, price_feedback_data=None, validation_data=None, epochs=100, batch_size=32):
        """
        Modell tanítása
        
        Paraméterek:
            training_data (dict): Tanítási adatok
            validation_data (dict): Validációs adatok (opcionális)
            epochs (int): Tanítási epochok száma
            batch_size (int): Batch méret
        """
        # Training data előkészítése
        X_keywords = np.array(training_data['keywords'])
        X_companies = np.array(training_data['company_embeddings'])
        
        # Financial weights csak ha van price feedback data
        if price_feedback_data:
            X_financial_weights = self.update_financial_weights(price_feedback_data, X_keywords, X_companies, training_data)
        else:
            # Uniform weights alapértelmezettként
            X_financial_weights = np.ones((len(X_keywords), self.max_keywords)) / self.max_keywords
        
        # Target adatok
        y_caps = np.array(training_data['capitalization_changes'])
        y_reconstructions = np.array(training_data['keyword_targets'])
        y_financial_attention = X_financial_weights.copy()
        
        # Validation data előkészítése hasonlóan
        validation_data_prepared = None
        if validation_data:
            val_X_keywords = np.array(validation_data['keywords'])
            val_X_companies = np.array(validation_data['company_embeddings'])
            val_X_financial_weights = np.ones((len(val_X_keywords), self.max_keywords)) / self.max_keywords
            val_y_caps = np.array(validation_data['capitalization_changes'])
            val_y_reconstructions = np.array(validation_data['keyword_targets'])
            val_y_financial_attention = val_X_financial_weights.copy()
            
            validation_data_prepared = (
                [val_X_keywords, val_X_companies, val_X_financial_weights],
                [val_y_caps, val_y_reconstructions, val_y_financial_attention]
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
            [X_keywords, X_companies, X_financial_weights],
            [y_caps, y_reconstructions, y_financial_attention],
            validation_data=validation_data_prepared,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Modell tanítása befejezve")
        return history
        
    def update_financial_weights(self, price_feedback_data, X_keywords, X_companies, training_data):
        """
        Financial weights frissítése training adatokkal
        """
        X_financial_weights = []
        
        # Financial tracker frissítése historikus adatokkal
        for feedback in price_feedback_data:
            words = self.tokenizer.tokenize_text(feedback['news_text'])
            self.financial_tracker.update_impact(
                words=words,
                company=feedback['company'],
                price_change_1d=feedback['price_change_1d'],
                price_change_5d=feedback['price_change_5d']
            )
        
        # Financial weights generálása minden training sample-hoz
        company_symbols = training_data.get('company_symbols', [])
        news_texts = training_data.get('news_texts', [])
        
        for i in range(len(X_keywords)):
            # Alapértelmezett értékek ha nincs elég adat
            if i < len(company_symbols) and i < len(news_texts):
                company = company_symbols[i] 
                text = news_texts[i]
                fin_weights = self._prepare_financial_weights(text, company)
            else:
                # Uniform weights ha nincs adat
                fin_weights = np.ones(self.max_keywords) / self.max_keywords
                
            X_financial_weights.append(fin_weights)

        return np.array(X_financial_weights)
        
    def predict_capitalization_change(self, keyword_sequence, company_embedding, news_text=None, company_symbol=None, return_detailed=False):
        """
        Integrált predikció attention súlyokkal
        
        ParamÉterek:
            keyword_sequence (numpy.ndarray): Kulcsszó szekvencia
            company_embedding (numpy.ndarray): Cég beágyazása
            news_text (str): Eredeti hír szöveg (opcionális)
            company_symbol (str): Cég szimbólum (opcionális)
            return_detailed (bool): Részletes eredmény visszaadása
            
        Returns:
            numpy.ndarray vagy dict: Előrejelzett változások
        """
        # Dimenzió ellenőrzése
        if keyword_sequence.ndim == 1:
            keyword_sequence = keyword_sequence.reshape(1, -1)
        if company_embedding.ndim == 1:
            company_embedding = company_embedding.reshape(1, -1)
        
        # Financial weights előkészítése
        if news_text and company_symbol:
            financial_weights = self._prepare_financial_weights(news_text, company_symbol)
            financial_weights = financial_weights.reshape(1, -1)
        else:
            financial_weights = np.ones((1, self.max_keywords)) / self.max_keywords
        
        # Predikció
        predictions = self.model.predict([keyword_sequence, company_embedding, financial_weights], verbose=0)
        capitalization_pred = predictions[0][0]
        reconstruction_pred = predictions[1][0]
        financial_attention_pred = predictions[2][0]
        
        if return_detailed:
            # Decoded keywords az interpretációhoz
            decoded_keywords = []
            for idx in keyword_sequence[0]:
                if idx in self.tokenizer.idx_to_word:
                    decoded_keywords.append(self.tokenizer.idx_to_word[idx])
                else:
                    decoded_keywords.append('[UNK]')
            
            return {
                'capitalization_prediction': capitalization_pred,
                'learned_financial_attention': financial_attention_pred,
                'input_financial_weights': financial_weights[0],
                'reconstruction_quality': reconstruction_pred,
                'attention_analysis': {
                    'decoded_keywords': decoded_keywords,
                    'top_financial_keywords': self._get_top_keywords(decoded_keywords, financial_attention_pred, top_k=5),
                    'confidence_score': np.mean(np.abs(capitalization_pred))
                }
            }
        else:
            return capitalization_pred
        
    def _prepare_financial_weights(self, text, company):
        """
        Financial attention súlyok előkészítése
        """
        words = self.tokenizer.tokenize_text(text)
        
        # Lekérjük a pénzügyi súlyokat
        financial_weights = self.financial_tracker.get_financial_attention_weights(
            words[:self.max_keywords], company
        )
        
        # Padding/truncation
        if len(financial_weights) < self.max_keywords:
            padding = np.zeros(self.max_keywords - len(financial_weights))
            financial_weights = np.concatenate([financial_weights, padding])
        else:
            financial_weights = financial_weights[:self.max_keywords]
            
        return financial_weights
    
    def _get_top_keywords(self, decoded_keywords, attention_weights, top_k=5):
        """
        Top-k kulcsszavak a legnagyobb attention súllyal
        """
        non_pad_indices = [i for i, word in enumerate(decoded_keywords) if word != '[PAD]' and i < len(attention_weights)]
        
        if not non_pad_indices:
            return []
        
        word_attention_pairs = [(decoded_keywords[i], attention_weights[i]) for i in non_pad_indices]
        word_attention_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return word_attention_pairs[:top_k]
    