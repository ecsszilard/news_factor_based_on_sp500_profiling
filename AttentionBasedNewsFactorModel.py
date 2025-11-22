import logging
import numpy as np
import tensorflow as tf
import mlflow
import os
import warnings

# Suppress TensorFlow/Transformers deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow and JAX classes are deprecated.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

logger = logging.getLogger("AdvancedNewsFactor.AttentionBasedNewsFactorModel")
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers
callbacks = tf.keras.callbacks
metrics = tf.keras.metrics
losses = tf.keras.losses
initializers = tf.keras.initializers

class CompoundProbabilisticLoss(losses.Loss):
    """
    Compact probabilistic loss for correlation prediction with uncertainty.

    Components:
    1. **CRPS (Continuous Ranked Probability Score)** – main distributional loss  
       - Evaluates both accuracy and calibration; no separate calibration penalty needed.  
       - Closed form for Gaussian: CRPS(N(μ,σ²), y) = σ[z(2Φ(z)-1) + 2φ(z) - 1/√π], z=(y–μ)/σ.  
       - Balances mean accuracy and uncertainty without extra tuning.

    2. **Adaptive weighting** – highlights important prediction pairs  
       - Higher weight for pairs involving news-affected companies.  
       - Larger |Δ_corr| gets proportionally higher weight.  
       - Normalized to mean = 1 for stable gradients.

    3. **Variance floor penalty** – prevents overconfident, collapsed σ  
       - Small penalty (λ_var ≈ 1e-4) discourages σ → 0.  
       - Avoids degenerate “fake certainty” solutions while staying minimally intrusive.

    4. **Directional hinge (high-confidence only)** – improves trading usefulness  
       - Activates when σ < threshold (model claims high confidence).  
       - Penalizes wrong direction: sign(Δ_pred) ≠ sign(Δ_actual).  
       - Margin avoids trivial Δ_pred ≈ 0 predictions.  
       - Penalty increases as confidence grows.

    Args:
        min_sigma: Minimum allowed standard deviation.  
        affected_weight: Weight multiplier for news-affected pairs (e.g., 3.3).  
        magnitude_weight: Weight per unit of |Δ_corr| (e.g., 1.2).  
        lambda_var: Variance-floor penalty weight (e.g., 1e-4).  
        lambda_dir: Directional hinge weight (e.g., 1e-2).  
        high_conf_threshold: σ threshold for activating directional loss (e.g., 0.85).  
        directional_margin: Margin for the directional hinge (e.g., 0.05 in Fisher-z space).

    Returns:
        A scalar loss combining weighted CRPS, variance-floor penalty,
        and confidence-conditioned directional hinge.
    """
    def __init__(self, 
                 min_sigma=1e-2,  # ← FONTOS: ne engedjük túl kicsire
                 affected_weight=3.3, 
                 magnitude_weight=1.2,
                 lambda_var=1e-4,
                 lambda_dir=1e-2,
                 high_conf_threshold=0.85,
                 directional_margin=0.04,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_sigma = float(min_sigma)
        self.affected_weight = float(affected_weight)
        self.magnitude_weight = float(magnitude_weight)
        self.lambda_var = float(lambda_var)
        self.lambda_dir = float(lambda_dir)
        self.high_conf_threshold = float(high_conf_threshold)
        self.directional_margin = float(directional_margin)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'min_sigma': self.min_sigma,
            'affected_weight': self.affected_weight,
            'magnitude_weight': self.magnitude_weight,
            'lambda_var': self.lambda_var,
            'lambda_dir': self.lambda_dir,
            'high_conf_threshold': self.high_conf_threshold,
            'directional_margin': self.directional_margin
        })
        return config
    
    def crps_gaussian(self, y_true, mu, sigma):
        z = (y_true - mu) / (sigma + 1e-12)
        cdf_z = 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(2.0)))
        pdf_z = tf.exp(-0.5 * tf.square(z)) / tf.sqrt(2.0 * np.pi)
        return sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / tf.sqrt(np.pi))
    
    def call(self, y_true, y_pred):
        actual_corr = y_true[..., 0]
        baseline_corr = y_true[..., 1]
        affected_mask = tf.cast(y_true[..., 2], tf.float32)
        
        mu = y_pred[..., 0]
        raw_logvar = tf.clip_by_value(y_pred[..., 1], -6.0, 10.0)
        
        # Variance construction with floor
        var = tf.nn.softplus(raw_logvar) + self.min_sigma
        sigma = tf.sqrt(var)
        
        # 1. CRPS Loss
        crps_elem = self.crps_gaussian(actual_corr, mu, sigma)
        
        # Normalized weighting
        affected_weights = 1.0 + (self.affected_weight - 1.0) * affected_mask
        magnitude = tf.abs(actual_corr - baseline_corr)
        total_weights = affected_weights * (1.0 + self.magnitude_weight * magnitude)
        total_weights_norm = total_weights / (tf.reduce_mean(total_weights) + 1e-12)
        
        weighted_crps = tf.reduce_mean(crps_elem * total_weights_norm)
        
        # 2. Minimal variance floor penalty (prevents collapse)
        if self.lambda_var > 0:
            shortfall = tf.nn.relu(self.min_sigma - var)
            var_penalty = self.lambda_var * tf.reduce_mean(shortfall)
        else:
            var_penalty = 0.0
        
        # 3. Directional hinge loss (improved version)
        actual_delta = actual_corr - baseline_corr
        pred_delta = mu - baseline_corr
        
        # Confidence weight: smooth from 0 (uncertain) to 1 (confident)
        conf_weight = tf.clip_by_value(
            (self.high_conf_threshold - sigma) / self.high_conf_threshold, 
            0.0, 1.0
        )
        
        # Hinge: penalize wrong direction proportionally
        dir_violation = tf.nn.relu(-tf.sign(actual_delta) * pred_delta + self.directional_margin)
        
        # Apply confidence-weighted penalty
        dir_loss = self.lambda_dir * tf.reduce_mean(conf_weight * dir_violation)
        return weighted_crps + var_penalty + dir_loss
    
class HighConfidenceDirectionalAccuracy(metrics.Metric):
    """
    Measures how often the model predicts the *correct direction* of correlation change
    (delta = actual_corr - baseline_corr), but ONLY on pairs where the model is confident.
    
    Why this metric?
    - If the model's predicted uncertainty (sigma) is low, it claims to be confident.
    - We then check: did it at least guess the direction correctly?
    - This reveals whether "high confidence" output is actually trustworthy.
    """

    def __init__(self, sigma_threshold, min_sigma, **kwargs):
        super().__init__(**kwargs)
        self.sigma_threshold = sigma_threshold
        self.min_sigma = min_sigma
        self.correct_high_conf = self.add_weight(name='correct_high_conf', initializer='zeros')
        self.total_high_conf   = self.add_weight(name='total_high_conf',   initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        actual_corr   = y_true[..., 0]
        baseline_corr = y_true[..., 1]

        mu        = y_pred[..., 0]
        raw_logvar = y_pred[..., 1]

        # Convert predicted log-variance into standard deviation (sigma)
        # softplus ensures positivity; sqrt converts variance → std dev
        sigma = tf.sqrt(tf.nn.softplus(raw_logvar) + self.min_sigma)

        # Only evaluate pairs where the model claims "low uncertainty"
        high_conf_mask = tf.less(sigma, self.sigma_threshold)

        # Delta = correlation change relative to baseline
        actual_delta = actual_corr - baseline_corr
        pred_delta   = mu          - baseline_corr

        # Ignore "near-zero" true deltas: no real direction exists
        eps = 1e-4
        valid_direction_mask = tf.greater(tf.abs(actual_delta), eps)

        # Check if model predicted the correct direction (sign)
        correct = tf.equal(tf.sign(actual_delta), tf.sign(pred_delta))
        correct = tf.logical_and(correct, valid_direction_mask)

        # Combine: correct AND model was confident
        correct_and_confident = tf.logical_and(correct, high_conf_mask)

        self.correct_high_conf.assign_add(tf.reduce_sum(tf.cast(correct_and_confident, tf.float32)))
        self.total_high_conf.assign_add(tf.reduce_sum(tf.cast(high_conf_mask, tf.float32)))

    def result(self):
        return tf.math.divide_no_nan(self.correct_high_conf, self.total_high_conf)

    def reset_state(self):
        self.correct_high_conf.assign(0.0)
        self.total_high_conf.assign(0.0)

class CalibrationMetric(metrics.Metric):
    """
    Measures how well the predicted uncertainty (σ) matches actual errors
    Good calibration: High σ → High error, Low σ → Low error
    """
    def __init__(self, min_sigma, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        self.min_sigma = float(min_sigma)
        self.num_bins = num_bins
        self.bin_errors = self.add_weight(
            name='bin_errors',
            shape=(num_bins,),
            initializer='zeros'
        )
        self.bin_counts = self.add_weight(
            name='bin_counts', 
            shape=(num_bins,), 
            initializer='zeros'
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_bins': self.num_bins})
        return config
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract actual correlations and predictions (mean, log variance)
        actual_corr = y_true[..., 0]
        mu = y_pred[..., 0]
        raw_logvar = y_pred[..., 1]
        sigma = tf.sqrt(tf.nn.softplus(raw_logvar) + self.min_sigma)  # Calculate standard deviation
        
        # Calculate the absolute errors between actual and predicted values
        errors = tf.abs(actual_corr - mu)
        
        # Flatten the tensors for binning
        sigma_flat = tf.reshape(sigma, [-1])
        errors_flat = tf.reshape(errors, [-1])
        
        # --- Vectorized binning with accumulation ---
        
        # Calculate bin width and normalize sigma values
        sigma_max = tf.reduce_max(sigma_flat)
        sigma_min = tf.reduce_min(sigma_flat)
        bin_width = (sigma_max - sigma_min) / tf.cast(self.num_bins, tf.float32)
        normalized_sigmas = (sigma_flat - sigma_min) / (bin_width + 1e-7)
        
        # Compute bin indices and clamp them to the range [0, num_bins-1]
        bin_indices = tf.cast(normalized_sigmas, tf.int32)
        bin_indices = tf.clip_by_value(bin_indices, 0, self.num_bins - 1)
        
        # Use unsorted_segment_sum for proper accumulation
        errors_to_add = tf.math.unsorted_segment_sum(errors_flat, bin_indices, num_segments=self.num_bins)
        counts_to_add = tf.math.unsorted_segment_sum(tf.ones_like(errors_flat), bin_indices, num_segments=self.num_bins)
        
        # Update bin errors and counts
        self.bin_errors.assign_add(errors_to_add)
        self.bin_counts.assign_add(counts_to_add)
    
    def result(self):
        avg_errors = self.bin_errors / (self.bin_counts + 1e-7)
        return tf.reduce_mean(tf.abs(avg_errors))
    
    def reset_state(self):
        self.bin_errors.assign(tf.zeros_like(self.bin_errors))
        self.bin_counts.assign(tf.zeros_like(self.bin_counts))

class KeywordEncoderBlock(layers.Layer):
    """
    Attention-based keyword encoder:
    - MultiHeadAttention with learnable temperature
    - Residual + LayerNorm
    - Global average + max pooling over tokens
    - Optional projection to latent_dim
    
    Output:
        smooth_global: [batch, latent_dim]
        processed_tokens: [batch, seq_len, feature_dim]
    """
    def __init__(self, num_heads, key_dim, latent_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.layer_norm = layers.LayerNormalization()

        # Learnable scaling of attention output
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer='ones',
            trainable=True
        )

        # Projection for smooth global representation
        self.global_proj = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5)
        )

        super().build(input_shape)

    def call(self, token_embeddings, training=None):
        # 1) Multi-head attention
        attn_out = self.attn(
            token_embeddings, token_embeddings, token_embeddings,
            training=training
        )
        attn_out = attn_out * tf.nn.softplus(self.temperature)

        # 2) Residual + LN
        processed_tokens = self.layer_norm(
            token_embeddings + self.dropout(attn_out, training=training)
        )  # [batch, seq_len, feature_dim]

        # 3) Global features (avg + max)
        global_avg = tf.reduce_mean(processed_tokens, axis=1)
        global_max = tf.reduce_max(processed_tokens, axis=1)
        global_features = tf.concat([global_avg, global_max], axis=-1)

        # 4) Smooth latent projection
        smooth_global = self.global_proj(global_features)

        return smooth_global, processed_tokens

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config

class AdaptiveGate(layers.Layer):
    """
    Dense → learnable-temperature sigmoid gate.
    Controls (0..1) how much each local feature passes through.
    Optional L1 penalty encourages sparse, selective gating.
    """
    def __init__(self, units, initial_temperature=1.0, activity_l1=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.initial_temperature = initial_temperature
        self.activity_l1 = activity_l1

        self.dense = layers.Dense(
            units,
            activation=None,
            kernel_regularizer=regularizers.l1_l2(1e-6),
            name="gate_dense"
        )

    def build(self, input_shape):
        # Learnable temperature parameter
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=initializers.Constant(self.initial_temperature),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # Dense → logits
        logits = self.dense(inputs)  # [batch, seq_len, units]

        # Temperature (softplus ensures positivity)
        temp_pos = tf.nn.softplus(self.temperature)

        # Final gate
        gate = tf.sigmoid(logits * temp_pos)

        # Optional activity regularization on gate activations
        if self.activity_l1 > 0.0:
            reg = tf.reduce_mean(tf.abs(gate)) * self.activity_l1
            self.add_loss(reg)

        return gate

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "initial_temperature": self.initial_temperature,
            "activity_l1": self.activity_l1,
        })
        return config

class LocalEdgeBlock(layers.Layer):
    """
    Local feature refinement block:
    - Elementwise gating of conv features
    - Top-k mean pooling over tokens
    - Dropout
    - Projection to latent_dim
    """
    def __init__(self, latent_dim, top_k=3, dropout_rate=0.4, activity_l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.dropout_rate = dropout_rate
        self.activity_l2 = activity_l2

    def build(self, input_shapes):
        # Final projection after pooling
        self.proj = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5),
            name=f"{self.name}_proj_dense"
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.activity = layers.ActivityRegularization(l2=self.activity_l2)
        super().build(input_shapes)

    def top_k_mean(self, x):
        # x shape [batch, seq_len, filters]
        x_t = tf.transpose(x, [0, 2, 1])  # [batch, filters, seq_len]
        values, _ = tf.nn.top_k(x_t, k=self.top_k)    # [batch, filters, k]
        return tf.reduce_mean(values, axis=-1)        # [batch, filters]

    def call(self, inputs, training=None):
        local_conv, local_gate = inputs

        # 1) Elementwise gating
        gated = local_conv * local_gate   # [..., filters]

        # 2) Top-k mean pooling
        pooled = self.top_k_mean(gated)

        # 3) Dropout
        pooled = self.dropout(pooled, training=training)

        # 4) Projection
        out = self.proj(pooled)

        # 5) Optional L2 activity regularization
        out = self.activity(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "top_k": self.top_k,
            "dropout_rate": self.dropout_rate,
            "activity_l2": self.activity_l2,
        })
        return config

class PairwiseAdaptiveGate(layers.Layer):
    """
    Learns how much to trust the baseline correlation vs. predicted changes.
    """
    def __init__(self, 
                 num_companies, 
                 alpha_min=0.15,     # Minimum trust in baseline
                 hidden_units=64, 
                 **kwargs):
        super().__init__(**kwargs)
        self.num_companies = num_companies
        self.alpha_min = alpha_min
        self.hidden_units = hidden_units

        # Hidden projection
        self.gate_hidden = layers.Dense(
            hidden_units,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5),
            name="gate_hidden"
        )

        # Logits output
        # This starts the gate at sigmoid(0) = 0.5 (50% baseline, 50% delta)
        # instead of sigmoid(1) = 0.73 which was killing the delta branch
        self.gate_logits = layers.Dense(
            1,
            activation=None,
            kernel_regularizer=regularizers.l2(1e-6),
            bias_initializer='zeros',
            name="gate_logits"
        )

        self.reshape = layers.Reshape((num_companies, num_companies), name="gate_reshape")

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_companies": self.num_companies,
            "alpha_min": self.alpha_min,
            "hidden_units": self.hidden_units
        })
        return config

    def call(self, inputs, baseline_corr):        
        # 1. Hidden layer
        gate_hidden = self.gate_hidden(inputs)

        # 2. Gate predicts alpha per pair
        raw_alpha = tf.sigmoid(self.gate_logits(gate_hidden))

        # 3. Clamping: ensure alpha never goes below alpha_min
        alpha = self.reshape(raw_alpha) * (1.0 - self.alpha_min) + self.alpha_min
        
        # 4. Gated Baseline
        gated_baseline = layers.Multiply()([baseline_corr, alpha])
        return gated_baseline

class DeltaHead(layers.Layer):
    """
    Predicts the symmetric mean change (μ_ij) and log-variance (log σ²_ij) for each pair.
    
    Features:
      - Shared non-linear projection with Dropout
      - Symmetric output enforcement (M_ij = M_ji)
      - Diagonal zeroing (correlation change with self is always 0)
      - Optional L1 activity regularization on μ (prevents lazy zero-prediction)
    """

    def __init__(self, 
                 hidden_units=128, 
                 dropout_rate=0.5, 
                 delta_l1=1e-4, # Penalty for non-zero changes (sparsity)
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.delta_l1 = delta_l1

        self.hidden = layers.Dense(
            hidden_units,
            activation='tanh',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name="delta_shared_hidden"
        )

        self.dropout = layers.Dropout(dropout_rate)

        # Heads for mu and log_var
        # Using linear activation to allow full range
        self.mu_head = layers.Dense(1, activation='linear', name="delta_mu_raw")
        self.var_head = layers.Dense(1, activation='linear', name="delta_logvar_raw")

    def call(self, inputs, training=None):
        # inputs: [batch, N, N, features]
        
        # 1. Shared representation
        h = self.hidden(inputs)
        
        # MC Dropout logic: if training=True (during fit), dropout is active.
        # During inference, it's inactive unless we manually force it (for MC sampling).
        h = self.dropout(h, training=training)

        # 2. Raw predictions [batch, N, N, 1]
        mu_raw = self.mu_head(h)
        logvar_raw = self.var_head(h)

        # Remove last dimension -> [batch, N, N]
        mu = tf.squeeze(mu_raw, axis=-1)
        logvar = tf.squeeze(logvar_raw, axis=-1)

        # 3. For stability prevent exploding gradients or numerical instability in Loss
        logvar = tf.clip_by_value(logvar, -6.0, 6.0)

        # 4. Enforce Symmetry: M_sym = 0.5 * (M + M^T)
        # Transpose only the last two dimensions (N, N)
        mu_sym = 0.5 * (mu + tf.linalg.matrix_transpose(mu))
        logvar_sym = 0.5 * (logvar + tf.linalg.matrix_transpose(logvar))

        # 5. Enforce Zero Diagonal (Self-correlation change is 0)
        n = tf.shape(mu_sym)[-1]
        diag_mask = tf.expand_dims(1.0 - tf.eye(n), axis=0)
        
        mu_final = mu_sym * diag_mask
        logvar_final = logvar_sym * diag_mask

        # 6. Activity Regularization
        # Penalize magnitude of changes to encourage sparsity (only significant changes)
        if self.delta_l1 > 0.0:
            self.add_loss(self.delta_l1 * tf.reduce_mean(tf.abs(mu_final)))

        return mu_final, logvar_final

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate,
            "delta_l1": self.delta_l1
        })
        return config

class SharedNewsRepresentation(layers.Layer):
    """
    Builds the final shared news representation by combining:
    - global news features (smooth_global)
    - local edge features (local_edge_proj)

    Applies a Dense → LayerNorm → Dropout pipeline.
    """

    def __init__(self, latent_dim, dropout_rate=0.36, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        self.proj = layers.Dense(
            latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name="shared_news_representation"
        )
        self.norm = layers.LayerNormalization(name="shared_news_layernorm")
        self.drop = layers.Dropout(dropout_rate, name="shared_news_dropout")

    def call(self, smooth_global, local_edge_proj, training=None):
        x = tf.concat([smooth_global, local_edge_proj], axis=-1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.drop(x, training=training)
        return x

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate
        }

class PairwiseFeatureBuilder(layers.Layer):
    """
    Constructs pairwise features and also returns the computed company embeddings.

    Output (tuple):
    1. pairwise: [batch, N, N, 2*company_dim + latent_dim]
    2. company_emb: [batch, N, company_dim]
    """

    def __init__(self, num_companies, company_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_companies = num_companies
        self.company_dim = company_dim

        self.company_embedding = layers.Embedding(
            input_dim=num_companies,
            output_dim=company_dim,
            embeddings_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name="company_embeddings"
        )
        
        self.concat_layer = layers.Concatenate(axis=-1, name='pairwise_concat')

    def call(self, shared_news_rep):
        batch_size = tf.shape(shared_news_rep)[0] 
        
        # 1. Company embeddings
        all_indices = tf.tile(tf.expand_dims(tf.range(self.num_companies, dtype=tf.int32), 0), [batch_size, 1])
        company_emb = self.company_embedding(all_indices)

        # 2. Pairwise expansion
        company_i = tf.tile(tf.expand_dims(company_emb, 2), [1, 1, self.num_companies, 1])
        company_j = tf.tile(tf.expand_dims(company_emb, 1), [1, self.num_companies, 1, 1])
        news = tf.tile(
            tf.expand_dims(tf.expand_dims(shared_news_rep, 1), 1),
            [1, self.num_companies, self.num_companies, 1]
        )

        # 3. Concatenate
        pairwise = self.concat_layer([company_i, company_j, news])

        return pairwise, company_emb

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_companies": self.num_companies,
            "company_dim": self.company_dim,
        })
        return config

class AttentionBasedNewsFactorModel:
    """
    Enhanced multi-task learning model with:
    1. Residual Learning for correlation changes
    2. Probabilistic predictions (μ, σ²) for aleatoric uncertainty
    3. Weighted loss focusing on important pairs
    """

    def __init__(self, tokenizer, bert_model, num_companies, min_sigma=0.01, max_keywords=128, keyword_dim=256, company_dim=128, latent_dim=256):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.num_companies = num_companies
        self.max_keywords = max_keywords
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.min_sigma = min_sigma

        # Simple BERT embedding cache (hash-based)
        self._bert_cache = {}

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("news_correlation_trading_probabilistic")
        
        self.model = self.build_model()
        
        # Custom optimizer with gradient clipping
        optimizer = optimizers.AdamW(
            learning_rate=optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-4,
                first_decay_steps=2000,
                t_mul=2.0,
                m_mul=0.76,
                alpha=0.02
            ),
            weight_decay=2e-4,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'correlation_changes_probabilistic': CompoundProbabilisticLoss(
                    min_sigma=self.min_sigma,
                    affected_weight=3.3,
                    magnitude_weight=1.8,
                    lambda_var=1e-4,
                    lambda_dir=2e-2,
                    high_conf_threshold=0.85,
                    directional_margin=0.04
                ),
                'price_deviations': losses.Huber(delta=1.0),
                'news_reconstruction': 'mae'
            },
            loss_weights={
                'correlation_changes_probabilistic': 5.0,
                'price_deviations': 0.5,
                'news_reconstruction': 0.5
            },
            metrics={
                'correlation_changes_probabilistic': [CalibrationMetric(min_sigma=self.min_sigma), HighConfidenceDirectionalAccuracy(sigma_threshold=0.85, min_sigma=self.min_sigma)],
                'price_deviations': ['mae', 'mse'],
                'news_reconstruction': ['mae']
            }
        )

    def build_model(self, top_k: int = 3):
        """Multi-task model with Residual Probabilistic Residual Learning.
        Hybrid news encoder with controlled local-edge branch
        - top_k: number of top activations to aggregate from local conv (soft top-k mean)
        """

        # -----------------------------
        # 1. Inputs
        # -----------------------------
        # Fisher-z transformed baseline correlation matrix (square NxN)
        keyword_input = layers.Input(shape=(self.max_keywords,), name='keywords')
        baseline_correlation_input = layers.Input(
            shape=(self.num_companies, self.num_companies),
            name='baseline_correlation_input',
            dtype='float32'
        )

        # -----------------------------
        # 2. Keyword encoder (token embedding + attention)
        # -----------------------------
        keyword_embedding = layers.Embedding(
            input_dim=len(self.tokenizer),
            output_dim=self.keyword_dim,
            mask_zero=True,
            embeddings_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_embeddings'
        )(keyword_input)  # [batch, seq_len, keyword_dim]

        smooth_global, processed_keywords = KeywordEncoderBlock(
            num_heads=8,
            key_dim=max(1, self.keyword_dim // 8),
            latent_dim=self.latent_dim,
            dropout_rate=0.2,
            name='keyword_encoder'
        )(keyword_embedding)

        # -----------------------------
        # 3. Controlled local branch
        # -----------------------------
        # small conv filter bank
        local_filters = max(32, self.latent_dim // 4)
        # 1D convolutional layer is able to examine n-grams (e.g. kernel 3 = 3 words) at once, making it much more robust in recognizing "semantic edges" and less prone to noise.
        local_conv = layers.Conv1D(
            filters=local_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5),
            name='local_conv_small'
        )(processed_keywords)  # [batch, seq_len, local_filters]

        # adaptive gating over the local features with L1 activity regularization
        local_gate = AdaptiveGate(
            units=local_filters,
            initial_temperature=1.0,
            activity_l1=1e-3,
            name="local_gate"
        )(processed_keywords)

        # -----------------------------
        # 4. Elementwise multiply conv features with gate
        # -----------------------------
        local_edge_proj = LocalEdgeBlock(
            latent_dim=self.latent_dim,
            top_k=top_k,
            dropout_rate=0.35,
            activity_l2=1e-4,
            name="local_edge_block"
        )([local_conv, local_gate])

        # -----------------------------
        # 5. Combine branches -> shared_news_rep
        # -----------------------------
        shared_news_rep = SharedNewsRepresentation(
            latent_dim=self.latent_dim, 
            dropout_rate=0.35
        )(smooth_global, local_edge_proj) # [batch, latent_dim]

        # -----------------------------
        # 6. Company embeddings and pairwise features
        # -----------------------------
        pairwise_features, all_company_embeddings = PairwiseFeatureBuilder(
            num_companies=self.num_companies,
            company_dim=self.company_dim
        )(shared_news_rep) # [batch, N, N, 2*company_dim + latent]

        # -----------------------------
        # 7. Predict correlation CHANGE (delta)
        # -----------------------------
        # Delta Head: Predicts symmetric μ and log(σ²) using FULL context
        delta_mu_symmetric, delta_log_var_symmetric = DeltaHead(
            hidden_units=128,
            dropout_rate=0.35,
            delta_l1=2e-4, # Sparsity for delta values
            name="delta_head"
        )(pairwise_features)  # Uses baseline for context

        # -----------------------------
        # 8. Probabilistic residual predictions
        # -----------------------------
        # Adaptive Gate: Predicts trust level using ONLY news+companies
        gated_baseline = PairwiseAdaptiveGate(
            num_companies=self.num_companies,
            alpha_min=0.15,   # At least 15% baseline always remains
            hidden_units=64,
            name="pairwise_gate"
        )(pairwise_features, baseline_correlation_input)  # Baseline passed separately

        # Gated residual: α*baseline + delta
        correlation_mean = layers.Add(name='correlation_mean_prediction')([gated_baseline, delta_mu_symmetric])

        # Stack mean and log-variance for probabilistic loss
        correlation_changes_prob = layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=-1), 
            name='correlation_changes_probabilistic'
        )([correlation_mean, delta_log_var_symmetric])

        # -----------------------------
        # 9. Price deviations & reconstruction heads
        # -----------------------------
        # For price deviation, use shared_news_rep combined with flattened company embeddings
        price_dev_input = layers.Concatenate(name='price_dev_input')([shared_news_rep, layers.Flatten()(all_company_embeddings)])
        price_dev_hidden = layers.Dense(128, activation='relu', name='price_dev_hidden')(price_dev_input)
        price_dev_hidden = layers.Dropout(0.25, name='price_dev_dropout')(price_dev_hidden)
        price_deviations = layers.Dense(self.num_companies, activation='tanh', name='price_deviations')(price_dev_hidden) # [batch, N]

        # Encourage shared_news_rep to preserve reconstructible information
        recon_out = layers.Dense(self.latent_dim, activation='tanh', name='news_reconstruction')(shared_news_rep) # [batch, latent_dim]

        # -----------------------------
        # Build and return the Model
        # -----------------------------
        return models.Model(
            inputs=[keyword_input, baseline_correlation_input],
            outputs=[correlation_changes_prob, price_deviations, recon_out],
            name='NewsCorrelationProbabilisticResidualModel'
        )

    def train(self, training_data, validation_data=None, epochs=60, batch_size=16):
        """Train the probabilistic residual correlation model"""
        
        # Create output directory for models
        os.makedirs("./saved_models", exist_ok=True)
        
        with mlflow.start_run():
            mlflow.log_params({
                'epochs': epochs,
                'batch_size': batch_size,
                'keyword_dim': self.keyword_dim,
                'company_dim': self.company_dim,
                'latent_dim': self.latent_dim,
                'max_keywords': self.max_keywords,
                'architecture': 'probabilistic_residual_learning'
            })
            
            class MLflowCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric, value in logs.items():
                            mlflow.log_metric(metric, value, step=epoch)

            # Train data preparation with augmented targets
            
            X_keywords = np.array(training_data['keyword_sequence'])  # Shape: (batch, seq_len)
            X_baseline_correlation = np.array(training_data['baseline_correlation'])
            
            # Augment y_correlation_changes with baseline and affected_companies_mask for weighted loss
            y_correlation_actual = np.array(training_data['correlation_changes'])
            
            # Get affected companies mask from training data
            affected_companies_masks = np.array(training_data['affected_companies_mask'])  # [batch, N, N]
            
            # Create augmented target: [actual, baseline, affected_mask]
            # Shape: [batch, N, N, 3]
            y_correlation_augmented = np.stack([
                y_correlation_actual,
                X_baseline_correlation,
                affected_companies_masks
            ], axis=-1)
            
            y_price_deviations = np.array(training_data['price_deviations'])
            y_news_targets = np.array(training_data['news_targets'])

            # Validation data preparation
            validation_data_prepared = None
            if validation_data:
                val_X_keywords = np.array(validation_data['keyword_sequence'])              
                val_X_baseline_correlation = np.array(validation_data['baseline_correlation'])
                val_y_correlation_actual = np.array(validation_data['correlation_changes'])
                val_affected_companies_masks = np.array(validation_data['affected_companies_mask'])
                
                val_y_correlation_augmented = np.stack([
                    val_y_correlation_actual,
                    val_X_baseline_correlation,
                    val_affected_companies_masks
                ], axis=-1)
                
                val_y_price_deviations = np.array(validation_data['price_deviations'])
                val_y_news_targets = np.array(validation_data['news_targets'])
                
                validation_data_prepared = (
                    [val_X_keywords, val_X_baseline_correlation],
                    [val_y_correlation_augmented, val_y_price_deviations, val_y_news_targets]
                )
            
            # ---- Train ----
            history = self.model.fit(
                [X_keywords, X_baseline_correlation],
                [y_correlation_augmented, y_price_deviations, y_news_targets],
                validation_data=validation_data_prepared,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_correlation_changes_probabilistic_calibration_metric' if validation_data else 'correlation_changes_probabilistic_calibration_metric',
                        mode='min',
                        patience=40,  # Increased patience for calibration
                        min_delta=1e-3,
                        restore_best_weights=True
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss' if validation_data else 'loss', 
                        factor=0.5, 
                        patience=8, 
                        min_lr=9e-7
                    ),
                    MLflowCallback()
                ],
                verbose=1
            )
            
            # Save model manually to avoid MLflow serialization issues
            try:
                model_path = "./saved_models/news_correlation_model.keras"
                self.model.save(model_path, save_format='keras')
                
                # Log only the path as a parameter (not the model itself)
                mlflow.log_param('model_saved_to', model_path)
                logger.info("✅ Model saved successfully to %s", model_path)
                
            except Exception as e:
                logger.warning("Could not save full model: %s", str(e))
                # Fallback: save weights only
                weights_path = "./saved_models/model_weights.h5"
                self.model.save_weights(weights_path)
                mlflow.log_param('weights_saved_to', weights_path)
                logger.info("⚠️ Only weights saved to %s", weights_path)
            
            if validation_data:
                mlflow.log_metric('best_val_loss', min(history.history['val_loss']))
            else:
                mlflow.log_metric('best_train_loss', min(history.history['loss']))     
        return history

    def predict_with_uncertainty(self, text, baseline_correlation, n_samples: int = 30):
        """
        Monte Carlo Dropout uncertainty + reconstruction-based confidence.

        Args:
            text: Input text
            baseline_correlation: Baseline correlation matrix
            n_samples: Number of MC Dropout samples (default: 30)

        Returns:
            Dictionary with MC predictions, uncertainties, price deviations, and reconstruction error
        """

        # Prepare inputs
        keywords = self.prepare_keyword_sequence(text)
        if keywords.ndim == 1:
            keywords = keywords[None, :]

        # news target embedding: [1, latent_dim]
        news_target_embedding = self.get_bert_embedding(text)
        if news_target_embedding.ndim == 1:
            news_target_embedding = news_target_embedding.reshape(1, -1)
        
        if baseline_correlation.ndim == 2:
            baseline_correlation = baseline_correlation[None, ...]

        # Monte Carlo Dropout sampling
        mc_mu, mc_logvar = [], []
        keywords_tf = tf.convert_to_tensor(keywords, dtype=tf.int32)
        baseline_tf = tf.convert_to_tensor(baseline_correlation, dtype=tf.float32)
        
        for _ in range(n_samples):
            pred = self.model([keywords_tf, baseline_tf], training=True)
            mu_sample = pred[0][..., 0].numpy()
            logvar_sample = pred[0][..., 1].numpy()
            mc_mu.append(mu_sample)
            mc_logvar.append(logvar_sample)
        
        mc_mu = np.stack(mc_mu, axis=0)
        mc_logvar = np.stack(mc_logvar, axis=0)

        # Reconstruction error
        final_pred = self.model.predict([keywords, baseline_correlation], verbose=0)
        recon_prediction = final_pred[2]  # shape [1, latent_dim]
        min_len = min(recon_prediction.shape[-1], news_target_embedding.shape[-1])
        recon_error = np.mean(np.abs(news_target_embedding[..., :min_len] - recon_prediction[..., :min_len]))

        return {
            "mc_mu": mc_mu,
            "mc_logvar": mc_logvar,
            "price_deviations": final_pred[1][0],
            "recon_error": recon_error
        }
    
    def prepare_keyword_sequence(self, text: str) -> np.ndarray:
        """
        Tokenize text using the model's tokenizer

        Args:
            text (str): The text to be tokenized.
    
        Returns:
            A numpy array containing the tokenized input_ids.
        """
        return self.tokenizer(
            text,
            truncation=True, 
            padding='max_length',
            max_length=self.max_keywords, 
            return_tensors="np"
        )["input_ids"][0]
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding with simple hash-based caching
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector [latent_dim]
        """
        # Simple cache check
        text_key = hash(text.strip().lower())
        if text_key in self._bert_cache:
            return self._bert_cache[text_key].copy()

        # Tokenize and get embedding
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_keywords,
            return_tensors="tf"
        )

        # Forward pass (classifier token embedding)
        embedding_full = self.bert_model(inputs).last_hidden_state[:, 0, :].numpy()[0]  # shape (768,)
        embedding_sliced = embedding_full[:self.latent_dim] # shape (256,)

        # Cache the result
        self._bert_cache[text_key] = embedding_sliced.copy()
        return embedding_sliced