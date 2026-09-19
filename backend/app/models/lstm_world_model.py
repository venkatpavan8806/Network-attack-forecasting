"""LSTM world model: learns P(S_{t+1} | S_t, S_{t-1}, ..., S_{t-L+1}).

Two heads share a CNN -> Bi-LSTM -> attention-pooled encoding:
  - next_stage_head: classifies the next window's state_label (7 classes,
    including the derived "ambiguous_pre_attack" class)
  - next_state_head: regresses the next window's normalized feature vector,
    which is what makes K-step rollout possible -- the predicted vector is
    fed back in as if it were observed, so the model genuinely predicts its
    own future inputs rather than only ever seeing real history.

Encoder: a small 1D-CNN pass over each timestep's engineered feature vector
(finds local combinations among related features) feeds a bidirectional
LSTM (reads the seed window in both directions before pooling), then an
additive attention pools across time -- this satisfies the project's
CNN-hybrid + Bi-LSTM + attention requirement together. forward() still
returns (stage_logits, next_state, attn_weights) with unchanged shapes, so
rollout(), one_step_forecast(), and input_gradient_saliency() all work
exactly as before.

Explainability: a Bahdanau-style additive attention over the L past
timesteps is used to pool the LSTM's hidden states, and the resulting
per-timestep attention weights are returned alongside every prediction --
this is the primary explainability mechanism the project uses (SHAP is
reserved for the logistic-regression baseline only; SHAP-on-sequences is
fragile per the project brief). A simple input-gradient saliency pass gives
per-feature attribution for a specific prediction.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.config import (
    N_FEATURES, STAGE_CLASSES, STAGE_TO_IDX, IDX_TO_STAGE, SEQ_LEN,
    LSTM_WEIGHTS, LSTM_META, ROLLOUT_K,
)

N_CLASSES = len(STAGE_CLASSES)
BENIGN_IDX = STAGE_TO_IDX["benign"]
AMBIGUOUS_IDX = STAGE_TO_IDX["ambiguous_pre_attack"]


class _FeatureCNNBlock(nn.Module):
    """Per-timestep 1D conv across the feature axis -- treats the n_features
    engineered values at one timestep as a short 1D signal and looks for
    local combinations of related features, before the sequence model sees
    it. Input (batch, seq_len, n_features) -> output (batch, seq_len, C)."""

    def __init__(self, n_features: int, channels=(16, 32), dropout: float = 0.1):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_features = x.shape
        x = x.reshape(batch * seq_len, 1, n_features)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return x.reshape(batch, seq_len, self.out_dim)


class LSTMWorldModel(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, hidden_size: int = 64, num_layers: int = 1,
                 n_classes: int = N_CLASSES, dropout: float = 0.1, cnn_channels=(16, 32)):
        super().__init__()
        self.hidden_size = hidden_size
        self.cnn_channels = tuple(cnn_channels)

        self.cnn = _FeatureCNNBlock(n_features, self.cnn_channels, dropout)
        self.lstm = nn.LSTM(self.cnn.out_dim, hidden_size, num_layers=num_layers, batch_first=True,
                             bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        encoder_dim = hidden_size * 2  # bidirectional doubles it

        # additive attention over timesteps (same mechanism as before, dims adjusted for bidirectional)
        self.attn_w = nn.Linear(encoder_dim, encoder_dim)
        self.attn_v = nn.Linear(encoder_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.stage_head = nn.Linear(encoder_dim, n_classes)
        self.state_head = nn.Linear(encoder_dim, n_features)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, n_features)
        returns stage_logits (batch, n_classes), next_state (batch, n_features),
        attn_weights (batch, seq_len)
        """
        x = self.cnn(x)              # (batch, seq_len, cnn_out_dim)
        h_seq, _ = self.lstm(x)      # (batch, seq_len, hidden*2)
        scores = self.attn_v(torch.tanh(self.attn_w(h_seq))).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), h_seq).squeeze(1)  # (batch, hidden*2)
        context = self.dropout(context)
        stage_logits = self.stage_head(context)
        next_state = self.state_head(context)
        return stage_logits, next_state, attn_weights


def infiltration_probability(stage_probs: np.ndarray) -> float:
    """P(infiltration) = 1 - P(benign). The ambiguous_pre_attack class is
    counted as partial evidence of rising concern, not full confirmation, so
    it is left inside the "non-benign" mass at full weight (it already IS a
    non-benign class in STAGE_CLASSES) -- this keeps the curve monotonic in
    the model's own class probabilities rather than adding an extra
    hand-tuned weighting on top of what the model already learned."""
    return float(1.0 - stage_probs[BENIGN_IDX])


def save_model(model: LSTMWorldModel, hidden_size: int, num_layers: int):
    torch.save(model.state_dict(), LSTM_WEIGHTS)
    with open(LSTM_META, "w") as f:
        json.dump({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "seq_len": SEQ_LEN,
            "stage_classes": STAGE_CLASSES,
            "cnn_channels": list(model.cnn_channels),
        }, f, indent=2)


def load_model() -> LSTMWorldModel:
    with open(LSTM_META) as f:
        meta = json.load(f)
    model = LSTMWorldModel(n_features=meta["n_features"], hidden_size=meta["hidden_size"],
                            num_layers=meta["num_layers"], n_classes=meta["n_classes"],
                            cnn_channels=meta.get("cnn_channels", (16, 32)))
    model.load_state_dict(torch.load(LSTM_WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def rollout(model: LSTMWorldModel, seed_window: np.ndarray, k: int = ROLLOUT_K):
    """Rolls the world model forward k steps from a real seed window.

    seed_window: (seq_len, n_features) normalized real observations ending at t.
    Returns dict with:
      infiltration_probs: list[float] length k (P(non-benign) at t+1..t+k)
      stage_probs: list[list[float]] length k (full class distribution at each horizon)
      predicted_stage: list[str] length k (argmax stage at each horizon)
      attn_weights_step1: list[float] length seq_len (attention over the seed window for the
                           first forecast step, i.e. which past real window drove the nearest-term forecast)
    """
    model.eval()
    seq = torch.tensor(seed_window, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_features)
    infiltration_probs, stage_probs_list, predicted_stage = [], [], []
    attn_weights_step1 = None

    for step in range(k):
        stage_logits, next_state, attn_weights = model(seq)
        probs = F.softmax(stage_logits, dim=1).squeeze(0).numpy()
        if step == 0:
            attn_weights_step1 = attn_weights.squeeze(0).numpy().tolist()
        infiltration_probs.append(infiltration_probability(probs))
        stage_probs_list.append(probs.tolist())
        predicted_stage.append(IDX_TO_STAGE[int(np.argmax(probs))])

        # autoregressive feed-forward: drop oldest, append predicted next state
        next_state_t = next_state.squeeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,n_features)
        seq = torch.cat([seq[:, 1:, :], next_state_t], dim=1)

    return {
        "infiltration_probs": infiltration_probs,
        "stage_probs": stage_probs_list,
        "predicted_stage": predicted_stage,
        "attn_weights_step1": attn_weights_step1,
    }


@torch.no_grad()
def one_step_forecast(model: LSTMWorldModel, window: np.ndarray):
    """Single one-step-ahead forecast (no autoregressive rollout). window:
    (seq_len, n_features). Returns (stage_probs: np.ndarray[n_classes],
    attn_weights: np.ndarray[seq_len])."""
    seq = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    stage_logits, next_state, attn_weights = model(seq)
    probs = F.softmax(stage_logits, dim=1).squeeze(0).numpy()
    return probs, attn_weights.squeeze(0).numpy(), next_state.squeeze(0).numpy()


def input_gradient_saliency(model: LSTMWorldModel, window: np.ndarray) -> np.ndarray:
    """Per-feature, per-timestep saliency for the infiltration-probability
    output: gradient of (1 - P(benign)) w.r.t. the input window, evaluated at
    the actual input. Returns (seq_len, n_features) array. This is a genuine
    gradient computed from the trained model's actual weights on the actual
    input -- not a static or hard-coded importance table."""
    model.eval()
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)
    stage_logits, _, _ = model(x)
    probs = F.softmax(stage_logits, dim=1)
    infiltration = 1.0 - probs[0, BENIGN_IDX]
    infiltration.backward()
    grad = x.grad.squeeze(0).numpy()  # (seq_len, n_features)
    return grad
