# Responsibility: Frozen SecBERT encoder for generating state embeddings (Phase 2)
#
# This module does NOT train anything.
# It loads the already-trained SecBERT checkpoint (EXP_20260727_001),
# freezes the weights, and provides a clean interface to convert
# incident text into 768-dimensional embeddings for the PPO agent.

import hashlib
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SecBERTStateEncoder:
    """
    Frozen SecBERT encoder that converts incident alert text
    into 768-dimensional state embeddings for the PPO agent.

    This is NOT a new model. It is a read-only wrapper around
    the SecBERT transformer that was already trained in Phase 1.
    """

    # Default path to our locked final model (EXP_002G / FP32 Ablation)
    DEFAULT_CHECKPOINT = os.path.join("models", "secbert_finetuned")

    def __init__(self, checkpoint_path=None, normalize=False):
        """
        Step 1: Load the tokenizer.

        The tokenizer converts English text into token IDs that
        SecBERT can understand. It is the exact same tokenizer
        that was used during training — saved alongside the model
        inside the checkpoint directory.

        Args:
            checkpoint_path: Path to the best_model directory.
                             Defaults to models/secbert_finetuned.
            normalize: If True, L2-normalize embeddings before
                       returning. This maps all vectors onto the
                       unit sphere, which can help PPO by making
                       embedding magnitudes consistent. Optional,
                       not forced.

        Example:
            Input:  "PowerShell executed"
            Output: [101, 2035, 4568, ...]
        """
        self.checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT
        self.normalize = normalize

        if not os.path.isdir(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at: {self.checkpoint_path}\n"
                f"Expected the best_model directory at models/secbert_finetuned."
            )

        # ── Step 1: Load tokenizer ───────────────────────────────
        # Same tokenizer used during Phase 1 training.
        # max_length=512 to match the training configuration.
        self.max_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)

        print("=" * 50)
        print("SecBERT State Encoder — Initialization")
        print("=" * 50)
        print(f"  Checkpoint       : {self.checkpoint_path}")
        print(f"  Tokenizer class  : {self.tokenizer.__class__.__name__}")
        print(f"  Vocab size       : {self.tokenizer.vocab_size}")
        print(f"  Max length       : {self.max_length}")
        print(f"  [PASS] Tokenizer loaded successfully")

        # ── Step 2: Load the trained model ───────────────────────
        # We load using AutoModel, NOT AutoModelForSequenceClassification.
        #
        # Why?
        #   AutoModelForSequenceClassification = BERT + Classifier Head
        #   AutoModel                         = BERT only
        #
        # The checkpoint was saved as BertForSequenceClassification,
        # but AutoModel knows how to load just the base transformer
        # from it. The classifier weights are simply ignored.
        #
        # This IS our trained model (EXP_20260727_001).
        # It already learned cybersecurity semantics.
        # We are NOT loading the generic jackaduma/SecBERT.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)

        # Confirm the embedding dimension is 768
        self.embedding_dim = self.model.config.hidden_size

        print(f"  Model class      : {self.model.__class__.__name__}")
        print(f"  Hidden layers    : {self.model.config.num_hidden_layers}")
        print(f"  Attention heads  : {self.model.config.num_attention_heads}")
        print(f"  Embedding dim    : {self.embedding_dim}")
        print(f"  Device           : {self.device}")
        print(f"  [PASS] Trained model loaded successfully")

        # ── Step 3: Freeze the model ────────────────────────────
        # PPO should never change SecBERT's weights.
        #
        # requires_grad = False
        #   → PyTorch will not compute gradients for these parameters.
        #   → No optimizer can update them.
        #
        # model.eval()
        #   → Disables dropout and batch normalization updates.
        #   → The model behaves identically on every forward pass.
        #
        # After this, SecBERT is read-only. Like a PDF.
        # You can read it. You cannot edit it.
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        frozen_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"  Frozen params    : {frozen_params:,}")
        print(f"  Trainable params : {trainable_params}  (must be 0)")
        assert trainable_params == 0, "FREEZE FAILED: Some parameters are still trainable!"
        print(f"  [PASS] Model frozen successfully")

        # ── Step 4: Verify classifier head is NOT present ───────
        # The most confusing part, so let's be very explicit.
        #
        # The checkpoint on disk (model.safetensors) contains:
        #   - BERT transformer weights     (what we want)
        #   - Classifier head weights       (what we ignore)
        #
        # Because we used AutoModel in Step 2 (not
        # AutoModelForSequenceClassification), the classifier
        # was never loaded into memory. It still exists in the
        # checkpoint file on disk — we simply don't use it.
        #
        # Nothing is deleted. Nothing is retrained.
        # We just stop at the 768-dimensional vector.
        #
        #   Text → Transformer → 768 vector  ← we stop here
        #                              ↓
        #                         Classifier  ← never loaded
        #                              ↓
        #                         20 actions   ← not our job
        has_classifier = hasattr(self.model, "classifier")
        has_out_proj = hasattr(self.model, "out_proj")

        print(f"  Has 'classifier' : {has_classifier}  (must be False)")
        print(f"  Has 'out_proj'   : {has_out_proj}  (must be False)")
        assert not has_classifier, (
            "Classifier head found! Did you use AutoModelForSequenceClassification? "
            "Use AutoModel instead."
        )
        assert not has_out_proj, (
            "Output projection found! The classifier head was loaded by mistake."
        )
        print(f"  [PASS] No classifier head — encoder only")

        # ── Embedding cache ─────────────────────────────────────
        # If Wazuh sends the same alert 100 times:
        #   Without cache → run SecBERT 100 times
        #   With cache    → run once, reuse 99 times
        #
        # Key: SHA-256 hash of the alert text
        # Value: the verified, detached CPU embedding
        self._cache = {}

        print(f"  L2 normalize     : {self.normalize}")
        print(f"  Cache            : enabled")
        print(f"  [PASS] Encoder ready")
        print("=" * 50)

    # ── Step 5: Return the embedding ────────────────────────
    #
    # This is the core purpose of this entire module.
    #
    # Input:  "PowerShell spawned"
    # Output: [0.45, -0.27, 1.82, ..., 768 values]
    #
    # NOT "Kill Process"
    # NOT "Block Port"
    # Just 768 numbers.
    #
    # That's what PPO needs.

    def encode_incident(self, text):
        """
        Convert a single incident alert into a 768-dimensional embedding.

        Args:
            text: A single alert string.
                  e.g. "PowerShell spawned encoded command on workstation-04"

        Returns:
            torch.Tensor of shape [768] on CPU — the [CLS] token embedding.
        """
        # ── Cache lookup ────────────────────────────────────────
        # Same alert text always produces the same embedding
        # (because SecBERT is frozen). No need to recompute.
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tokenize: convert English text into token IDs
        # Same settings as training: truncation at 512 tokens
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        # Forward pass through frozen SecBERT
        # torch.no_grad() because we never compute gradients here
        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.last_hidden_state shape: [1, seq_len, 768]
        # We take the [CLS] token (position 0) — this is the
        # sentence-level representation that SecBERT learned
        # during Phase 1 training.
        #
        #   [CLS] embedding = outputs.last_hidden_state[0, 0, :]
        #
        # Result: a flat tensor of 768 numbers.
        embedding = outputs.last_hidden_state[0, 0, :]

        # Optional L2 normalization — maps vector onto unit sphere
        # so all embeddings have magnitude 1.0. Not forced.
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=0)

        # Step 6: Verify before returning
        self.verify_embedding(embedding)

        # Detach from computation graph and move to CPU
        # so PPO can manage its own device placement cleanly.
        embedding = embedding.detach().cpu()

        # Store in cache for reuse
        self._cache[cache_key] = embedding

        return embedding

    def encode_batch(self, texts):
        """
        Convert a list of incident alerts into embeddings.

        Args:
            texts: A list of alert strings.

        Returns:
            torch.Tensor of shape [N, 768] on CPU — one embedding per alert.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Take [CLS] token (position 0) for every sample in the batch
        # outputs.last_hidden_state shape: [N, seq_len, 768]
        # Result: [N, 768]
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Optional L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 6: Verify every embedding in the batch
        for i in range(embeddings.shape[0]):
            self.verify_embedding(embeddings[i])

        # Detach and move to CPU
        embeddings = embeddings.detach().cpu()

        return embeddings

    # ── Step 6: Verify the embedding ────────────────────────
    #
    # Every single embedding is checked before it leaves
    # this module. Every time. No exceptions.
    #
    # If any check fails, we crash immediately.
    # A bad embedding going into PPO would silently corrupt
    # the entire training run.

    def verify_embedding(self, embedding):
        """
        Sanity-check a single embedding vector.

        Checks:
            1. Length == 768
            2. No NaN values
            3. No Infinity values
            4. Correct datatype (float32)

        Raises:
            AssertionError if any check fails.
        """
        # Check 1: Length must be exactly 768
        assert embedding.shape == (self.embedding_dim,), (
            f"Wrong embedding dimension: expected ({self.embedding_dim},), "
            f"got {embedding.shape}"
        )

        # Check 2: No NaN values
        assert not torch.isnan(embedding).any(), (
            "Embedding contains NaN values! Model output is corrupted."
        )

        # Check 3: No Infinity values
        assert not torch.isinf(embedding).any(), (
            "Embedding contains Inf values! Model output is corrupted."
        )

        # Check 4: Correct datatype (float32, matching our FP32 model)
        assert embedding.dtype == torch.float32, (
            f"Wrong dtype: expected torch.float32, got {embedding.dtype}"
        )

    def clear_cache(self):
        """
        Clear the embedding cache.

        Useful when switching to a new episode or scenario
        where old cached embeddings are no longer needed.
        """
        count = len(self._cache)
        self._cache.clear()
        print(f"  [INFO] Cache cleared ({count} entries removed)")
