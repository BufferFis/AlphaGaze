"""
rl_agent.py
-----------
PPO agent wrapper for stock trading signal generation.

Provides:
    - train_agent(): Train or fine-tune the PPO agent on historical data
    - predict_signal(): Get BUY/HOLD/SELL signal + probabilities for one observation
    - compute_feature_importance(): Perturbation-based feature importance (replaces SHAP)
    - save_agent() / load_agent(): Model persistence
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env import StockTradingEnv, FEATURE_COLS, ACTION_MAP

# PPO hyperparameters tuned for small financial datasets
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 64,           # Small: dataset is ~200 days. Default 2048 is too large.
    "batch_size": 32,        # Must be <= n_steps.
    "n_epochs": 10,          # Gradient updates per rollout.
    "gamma": 0.99,           # High discount factor — future rewards matter in trading.
    "gae_lambda": 0.95,      # Standard GAE lambda.
    "clip_range": 0.2,       # PPO clipping. Prevents catastrophic policy updates.
    "ent_coef": 0.01,        # Entropy bonus. Prevents collapse to always-HOLD.
    "vf_coef": 0.5,          # Value function coefficient.
    "max_grad_norm": 0.5,    # Gradient clipping for stability.
    "verbose": 0,            # Silent (runs inside API requests).
    "device": "cpu",         # CPU device for fast MLP rollouts without GPU transfer overhead
}

TOTAL_TIMESTEPS = 50_000


def _make_env(df: pd.DataFrame):
    """Create a vectorized environment (required by stable-baselines3)."""
    def _init():
        return StockTradingEnv(df)
    return DummyVecEnv([_init])


def train_agent(df: pd.DataFrame, total_timesteps: int = TOTAL_TIMESTEPS,
                seed: int = 42, pretrained_model: PPO = None) -> PPO:
    """
    Train (or fine-tune) a PPO agent.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FEATURE_COLS + 'next_return' columns. Sorted by date.
    total_timesteps : int
        Total environment interactions for training.
    seed : int
        Random seed for reproducibility.
    pretrained_model : PPO or None
        If provided, continue training from this model's weights (transfer learning).
        Used by rl_data.py for chaining training across multiple stocks, and by
        predictor.py for fine-tuning the pre-trained model on recent data.

    Returns
    -------
    PPO
        Trained PPO model.
    """
    env = _make_env(df)

    if pretrained_model is not None:
        # Transfer learning: keep learned weights, swap to new environment
        model = pretrained_model
        model.set_env(env)
    else:
        model = PPO("MlpPolicy", env, seed=seed, **PPO_CONFIG)

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=(pretrained_model is None),
    )
    env.close()
    return model


def predict_signal(model: PPO, obs: np.ndarray) -> dict:
    """
    Get trading signal and action probabilities from the trained agent.

    This extracts the actor network's categorical distribution, giving
    true action probabilities that sum to 1.0 — directly compatible with
    the existing dashboard probability bar chart.

    Parameters
    ----------
    model : PPO
        Trained PPO model.
    obs : np.ndarray
        Feature vector of shape (7,).

    Returns
    -------
    dict with keys:
        signal: str — "BUY" / "HOLD" / "SELL"
        probabilities: dict — {"BUY": float, "HOLD": float, "SELL": float}
        action_idx: int — 0=SELL, 1=HOLD, 2=BUY
    """
    import torch

    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Get action probabilities from the policy network
    obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.policy.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(
            model.policy.extract_features(obs_tensor, model.policy.features_extractor)
        )
        action_probs = dist.distribution.probs.cpu().numpy().flatten()

    # Deterministic prediction (argmax)
    action_idx = int(np.argmax(action_probs))
    signal = ACTION_MAP[action_idx]

    # Build probability dict matching existing API format: {"BUY": ..., "HOLD": ..., "SELL": ...}
    probabilities = {
        "BUY": round(float(action_probs[2]), 4),   # action 2 = BUY
        "HOLD": round(float(action_probs[1]), 4),   # action 1 = HOLD
        "SELL": round(float(action_probs[0]), 4),    # action 0 = SELL
    }

    return {
        "signal": signal,
        "probabilities": probabilities,
        "action_idx": action_idx,
    }


def compute_feature_importance(model: PPO, obs: np.ndarray,
                                n_perturbations: int = 50) -> dict:
    """
    Perturbation-based feature importance (replaces SHAP TreeExplainer).

    For each of the 7 features, randomly perturb it N times and measure how much
    the predicted action probability changes. Higher change = more important feature.

    This produces a dict in the SAME FORMAT as the existing shap_dict:
        {feature_name: importance_score}
    The frontend's "Feature Impact" bar chart renders it without any changes.

    Parameters
    ----------
    model : PPO
        Trained PPO model.
    obs : np.ndarray
        Feature vector of shape (7,).
    n_perturbations : int
        Number of random perturbations per feature.

    Returns
    -------
    dict
        {feature_name: float} — same schema as existing shap_dict.
    """
    import torch

    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Get baseline prediction
    obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.policy.device)
    with torch.no_grad():
        dist = model.policy.get_distribution(
            model.policy.extract_features(obs_tensor, model.policy.features_extractor)
        )
        baseline_probs = dist.distribution.probs.cpu().numpy().flatten()

    baseline_action = int(np.argmax(baseline_probs))
    baseline_conf = float(baseline_probs[baseline_action])

    importance = {}
    np.random.seed(42)

    for i, feature_name in enumerate(FEATURE_COLS):
        deltas = []
        for _ in range(n_perturbations):
            perturbed_obs = obs.copy()
            noise_scale = max(abs(obs[i]) * 0.5, 0.01)
            perturbed_obs[i] += np.random.normal(0, noise_scale)

            p_tensor = torch.as_tensor(perturbed_obs).unsqueeze(0).to(model.policy.device)
            with torch.no_grad():
                p_dist = model.policy.get_distribution(
                    model.policy.extract_features(p_tensor, model.policy.features_extractor)
                )
                p_probs = p_dist.distribution.probs.cpu().numpy().flatten()

            p_conf = float(p_probs[baseline_action])
            deltas.append(baseline_conf - p_conf)

        mean_delta = float(np.mean(deltas))
        importance[feature_name] = round(mean_delta, 6)

    return importance


def save_agent(model: PPO, path: str):
    """Save trained PPO model to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    print(f"[rl_agent] Model saved to {path}")


def load_agent(path: str) -> PPO:
    """Load a trained PPO model from disk."""
    model = PPO.load(path)
    print(f"[rl_agent] Model loaded from {path}")
    return model
