# AlphaGaze — RL Migration Implementation Plan

> **Objective**: Replace the Random Forest ensemble classifier with a PPO (Proximal Policy Optimization) Reinforcement Learning agent for generating BUY / HOLD / SELL trading signals, while preserving all existing XAI, frontend, and API infrastructure.

---

## Table of Contents

1. [Current System](#1-current-system)
2. [Why Replace Random Forest with RL](#2-why-replace-random-forest-with-rl)
3. [Why PPO Specifically](#3-why-ppo-specifically)
4. [Data Strategy — Two-Phase Training](#4-data-strategy--two-phase-training)
5. [New Dependencies](#5-new-dependencies)
6. [File Change Summary](#6-file-change-summary)
7. [Implementation — New Files](#7-implementation--new-files)
8. [Implementation — Modified Files](#8-implementation--modified-files)
9. [yfinance Fallback Strategies](#9-yfinance-fallback-strategies)
10. [Verification Plan](#10-verification-plan)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [Execution Checklist](#12-execution-checklist)

---

## 1. Current System

The existing AlphaGaze pipeline uses a **RandomForestClassifier** (300 trees, max_depth=6, balanced class weights) as its final decision layer.

### How it works today

```
7 engineered features  →  Random Forest  →  BUY / HOLD / SELL  →  SHAP explanation
                            (supervised)       + probabilities       (TreeExplainer)
```

- **Features** (7 columns): `prophet_gap`, `forecast_band`, `sentiment_1d`, `sentiment_3d_ma`, `sentiment_5d_ma`, `price_momentum_5d`, `volatility_5d`
- **Labels**: Next-day return > +0.3% → BUY, < -0.3% → SELL, else HOLD
- **Training**: Supervised learning on the entire merged dataset, evaluated with TimeSeriesSplit CV
- **Explainability**: SHAP `TreeExplainer` for per-feature importance
- **Output**: Class probabilities via `predict_proba()`

### Files involved in the classifier

| File | Role |
|------|------|
| `app/predictor.py` | Main pipeline. Trains RF, predicts, SHAP, backtest. Lines 658–702. |
| `app/combine_and_xai.py` | Offline pipeline. Trains RF, SHAP plots. Lines 139–227. |
| `app/evaluator.py` | Evaluation. Trains RF with CV. Lines 65–76. |

---

## 2. Why Replace Random Forest with RL

| Problem with RF | How RL Solves It |
|---|---|
| **Static supervised learning**: Treats each day independently. No notion of sequential decision-making or what the agent's current position is. | RL considers temporal context. The agent's reward depends on its sequence of actions. |
| **Threshold-dependent labels**: Arbitrary ±0.3% thresholds define BUY/SELL. The model is trained to match these labels rather than maximize returns. | RL uses reward signals from actual returns. No arbitrary labeling needed. |
| **No reward optimization**: RF optimizes classification accuracy (F1 score), not trading profit. | RL directly optimizes the trading reward function. |
| **No position awareness**: RF doesn't know if it already signaled BUY yesterday. | RL tracks previous action; transaction costs penalize excessive position changes. |

### What stays the same

- The 3 actions: BUY, HOLD, SELL
- The 7 features (observation space for RL)
- The API response schema (frontend doesn't change)
- Prophet forecasting, FinBERT sentiment, news scraping — all untouched

---

## 3. Why PPO Specifically

### The constraints that determine the algorithm

| Constraint | Value |
|---|---|
| Action space | **Discrete(3)** — BUY, HOLD, SELL |
| Observation space | **Continuous(7)** — the 7 engineered features |
| Dataset size | **~200–500 trading days** (small by RL standards) |
| Compute | **CPU-only inference** (runs per API request) |
| Critical requirement | **Must output action probabilities** (the dashboard has a probability bar chart) |
| Training frequency | **Per API request** (no persistent GPU server) |

### Head-to-head comparison

| Algorithm | Type | Discrete(3)? | Probability Output? | Stable with Small Data? | Speed | Verdict |
|---|---|---|---|---|---|---|
| **DQN** | Value-based | ✅ | ❌ No native softmax policy — Q-values are not probabilities | ⚠️ Mediocre | Fast | ❌ Fails on probability output |
| **A2C** | Actor-Critic | ✅ | ✅ | ❌ High variance, unstable updates | Very Fast | ⚠️ Acceptable but risky |
| **PPO** | Actor-Critic | ✅ | ✅ | ✅ Best in class (clipped objective) | Moderate | ✅ **Best fit** |
| **SAC** | Actor-Critic | ❌ Needs hack | ✅ | ✅ | Slow | ❌ Designed for continuous actions |
| **DDPG/TD3** | Deterministic | ❌ Continuous only | ❌ | ✅ | Moderate | ❌ Wrong action space |
| **Rainbow DQN** | Value-based | ✅ | ❌ | ⚠️ | Slow | ❌ Overkill + no probabilities |

### The 3 decisive reasons PPO wins

**1. The probability output problem kills DQN.**

The dashboard renders a probability bar chart: `{"BUY": 0.42, "HOLD": 0.35, "SELL": 0.23}`. DQN outputs Q-values, not probabilities. You'd need an ugly softmax hack over Q-values that has no theoretical grounding. PPO has a true stochastic policy — the actor network outputs a categorical distribution over `{SELL, HOLD, BUY}` that sums to 1.0. This maps directly to the existing API schema.

**2. The clipped objective is the only thing keeping training stable at 200 days.**

A2C and vanilla policy gradient methods can swing the policy wildly when gradients are large. With only 200 data points, this is a real danger — the agent commonly collapses to always-HOLD within a few hundred timesteps. PPO's clipped surrogate objective mathematically caps the policy update at each step:

```
L_CLIP = min(r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t)     where ε = 0.2
```

This means even a "good" gradient can't move the policy more than 20% in one update. A2C has no such protection.

**3. SAC is architecturally wrong.**

SAC is state-of-the-art for continuous action spaces (robotics, motor control). Our action space is `Discrete(3)`. There's a "Discrete SAC" variant but it's not in `stable-baselines3`'s official release.

### PPO hyperparameters and why each value was chosen

```python
PPO_CONFIG = {
    "learning_rate": 3e-4,      # SB3 default. Works well for most tasks.
    "n_steps": 64,              # SMALL: dataset is only ~200 days. Default 2048 would be larger than the dataset.
    "batch_size": 32,           # Must be ≤ n_steps. 32 is fine for small data.
    "n_epochs": 10,             # SB3 default. Number of gradient passes per rollout.
    "gamma": 0.99,              # High discount factor — future rewards matter in trading.
    "gae_lambda": 0.95,         # Standard GAE lambda.
    "clip_range": 0.2,          # PPO clipping range. Standard value.
    "ent_coef": 0.01,           # Entropy bonus. Prevents premature convergence to always-HOLD.
    "vf_coef": 0.5,             # Value function loss coefficient. Standard.
    "max_grad_norm": 0.5,       # Gradient clipping for stability.
    "verbose": 0,               # Silent training (inside API request).
}
```

---

## 4. Data Strategy — Two-Phase Training

### The data problem

The 7 features come from two sources with different historical availability:

```
                  2000          2010          2020     2024     NOW
PRICE DATA:  ████████████████████████████████████████████████████  yfinance has ALL of this
NEWS DATA:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  Google RSS only has ~2 weeks
```

| Feature | Source | Historical Availability |
|---|---|---|
| `prophet_gap` | Prophet (price math) | ✅ Unlimited — yfinance goes back 25 years |
| `forecast_band` | Prophet (price math) | ✅ Unlimited |
| `price_momentum_5d` | Price math | ✅ Unlimited |
| `volatility_5d` | Price math | ✅ Unlimited |
| `sentiment_1d` | Google News RSS + FinBERT | ❌ Only last ~2 weeks of headlines exist |
| `sentiment_3d_ma` | Google News RSS + FinBERT | ❌ Same |
| `sentiment_5d_ma` | Google News RSS + FinBERT | ❌ Same |

**Current usage**: ~250 trading days (1 year). **Available via yfinance**: ~5,000–6,200 trading days per stock (up to 25 years). That's a **25× increase** in training data — but only for 4 out of 7 features.

### Solution: Two-phase training

This is analogous to how LLMs are pre-trained on massive general data, then fine-tuned on specific task data.

#### Phase 1 — Offline Pre-training (run once before deploying)

```
Developer runs:  python app/rl_data.py --pretrain

Step 1: yfinance downloads max history for all 10 supported stocks
        → ^NSEI (~6200 days), RELIANCE.NS (~5000 days), INFY.NS (~6200 days), etc.
        → Cached to data/prices/<SYMBOL>_max.csv (never re-downloaded unless forced)

Step 2: Prophet is fit on each stock's full history
        → Computes prophet_gap + forecast_band for every historical day

Step 3: 7-feature rows are built:
        [prophet_gap=0.02, forecast_band=0.05, sentiment=0.0, 0.0, 0.0, momentum=0.01, vol=0.003]
        ← The 3 sentiment features are set to 0.0 (neutral) because historical news is unavailable.
        This is fine — the agent learns price dynamics without being confused by fake sentiment.

Step 4: PPO trains SEQUENTIALLY on each stock (not parallel — chained transfer learning):

        Train on NIFTY 25 years → [model checkpoint A]
                                          ↓
        Train on RELIANCE 20 years (starting from A) → [model checkpoint B]
                                                              ↓
        Train on INFY 25 years (starting from B) → [model checkpoint C]
                                                          ↓
                                                     ... all 10 stocks ...
                                                          ↓
                                                 data/ppo_pretrained.zip

        Each stock's training STARTS WHERE THE LAST ONE LEFT OFF.
        The policy weights accumulate knowledge across all stocks.
        By the end, the agent has seen 50,000+ trading days across Indian markets.

Step 5: Saves final model to data/ppo_pretrained.zip (~5MB file)

Total time: ~15-30 minutes on CPU. Run once. Never again (unless you want to refresh annually).
```

#### Phase 2 — Online Fine-tuning (happens automatically per API request)

```
User opens browser → selects INFY → API calls predict("INFY.NS")

Step 1-4: (Same as before) yfinance 2 years, Prophet, FinBERT news scraping, feature engineering
          Now we have REAL sentiment scores from actual recent headlines!

Step 5: RL training
        ├─ os.path.exists("data/ppo_pretrained.zip") → True!
        ├─ Load pre-trained model (0.1 seconds)
        │    "Agent who already knows Indian market patterns"
        ├─ Fine-tune on last 2 years of INFY with ALL 7 features
        │    [prophet_gap=0.02, forecast_band=0.05, sentiment=+0.34, 0.28, 0.19, mom=0.01, vol=0.003]
        │                                            ↑ REAL FinBERT score from actual headlines
        │    Only 10,000 timesteps (not 500K — agent is already pre-trained)
        │    ~5-10 seconds on CPU
        └─ Final model used for prediction (kept in memory, not saved to disk)

Step 6: predict_signal(model, latest_features)
        → {"signal": "BUY", "probabilities": {"BUY": 0.62, "HOLD": 0.28, "SELL": 0.10}}

Result is cached for 1 hour (existing cache mechanism, unchanged).
```

#### Why not just train from scratch each time?

| Approach | Training Time per Request | Data Used | Quality |
|---|---|---|---|
| From scratch, 2 years only | ~30-60 seconds | 500 days | ⚠️ Decent but limited |
| Pre-trained + fine-tune | ~5-10 seconds | 50,000 + 500 days | ✅ Best |
| From scratch, 5+ years (no sentiment) | ~2-3 minutes | 1,200+ days | ⚠️ Slow, no sentiment |

The two-phase approach is **both faster AND better** because:
- Pre-training captures 25 years of market structure (crashes, recoveries, regime changes)
- Fine-tuning is fast (10K steps vs 50K) because the agent isn't starting from zero
- Fine-tuning adds real sentiment awareness that pre-training lacks

#### The chess analogy

| Phase | Chess Equivalent |
|---|---|
| Pre-training | Grandmaster studies 10,000 historical games. Learns openings, endgames, positional patterns. Takes months. |
| Fine-tuning | Before a tournament, studies the specific opponent's last 50 games. Takes 1 day. |
| Inference | Plays the actual match using both — deep pattern knowledge + opponent-specific preparation. |

#### Data volume summary

| Phase | Data Source | Features | Training Days | PPO Timesteps |
|---|---|---|---|---|
| Offline Pre-training | yfinance `period=max` × 10 stocks | 4 (price only, sentiment=0.0) | ~50,000 | ~500,000 |
| Online Fine-tuning | yfinance `period=2y` × 1 stock | 7 (full, real sentiment) | ~500 | 10,000 |
| **Total** | | | **~50,500** | **~510,000** |

#### Fallback: If pre-trained model doesn't exist

If `data/ppo_pretrained.zip` doesn't exist (developer hasn't run pre-training yet), `predict()` falls back gracefully:

```python
if os.path.exists(PRETRAINED_PATH + ".zip"):
    # Fine-tune from pre-trained (fast, 10K steps)
    rl_model = train_agent(merged, total_timesteps=10_000, pretrained_model=load_agent(PRETRAINED_PATH))
else:
    # Train from scratch (slower, 50K steps, but still works)
    rl_model = train_agent(merged, total_timesteps=50_000)
```

The system works without pre-training — it's just slower and uses less data.

---

## 5. New Dependencies

Add to `requirements.txt`:

```
stable-baselines3==2.4.1
gymnasium==1.1.1
shimmy==2.0.0
```

These are the ONLY new dependencies. `stable-baselines3` depends on `torch` (already installed) and `gymnasium`. No conflicts with existing packages.

```bash
pip install stable-baselines3==2.4.1 gymnasium==1.1.1 shimmy==2.0.0
```

---

## 6. File Change Summary

### Files to CREATE

| File | Purpose |
|------|---------|
| `app/rl_env.py` | Custom Gymnasium trading environment: `StockTradingEnv` with `Discrete(3)` action space |
| `app/rl_agent.py` | PPO agent wrapper: `train_agent()`, `predict_signal()`, `compute_feature_importance()`, `save_agent()`, `load_agent()` |
| `app/rl_data.py` | Offline pre-training script: downloads max yfinance history, builds price features, chains PPO training across all 10 stocks |

### Files to MODIFY

| File | What Changes |
|------|-------------|
| `app/predictor.py` | Replace RF training + SHAP with PPO training + perturbation-based feature importance in `predict()`. Update `_rolling_backtest()` to use PPO. Extend data window to 2 years. |
| `app/combine_and_xai.py` | Replace RF training + SHAP plots with PPO training + feature importance plot in `run_combine_and_xai()` |
| `app/evaluator.py` | Replace RF cross-validation with PPO evaluation in `evaluate_pipeline()` |
| `requirements.txt` | Add 3 new dependencies |
| `agents.MD` | Update architecture docs to reflect PPO instead of RF |

### Files that DO NOT change

| File | Why |
|------|-----|
| `app/api.py` | Just calls `predict()` — transparent to classifier change |
| `app/scraper.py` | News scraping is unchanged |
| `app/sentiment.py` | Standalone FinBERT scorer, unrelated |
| `app/time_series.py` | Standalone Prophet runner, unrelated |
| `app/frontend/index.html` | Same JSON response schema → renders without changes |

---

## 7. Implementation — New Files

### 7.1 `app/rl_env.py` — The Trading Environment

This is the most critical new file. It defines the Gymnasium environment that PPO interacts with.

```python
"""
rl_env.py
---------
Custom Gymnasium environment for stock trading with 3 discrete actions.

Action Space: Discrete(3)
    0 = SELL
    1 = HOLD
    2 = BUY

Observation Space: Box(low=-inf, high=inf, shape=(7,), dtype=float32)
    The 7 engineered features from _build_features():
    [prophet_gap, forecast_band, sentiment_1d, sentiment_3d_ma,
     sentiment_5d_ma, price_momentum_5d, volatility_5d]

Reward: Based on actual next-day return aligned with the chosen action.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


# Map action indices to signal strings
ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_MAP = {"SELL": 0, "HOLD": 1, "BUY": 2}  # reverse mapping

FEATURE_COLS = [
    "prophet_gap",
    "forecast_band",
    "sentiment_1d",
    "sentiment_3d_ma",
    "sentiment_5d_ma",
    "price_momentum_5d",
    "volatility_5d",
]


class StockTradingEnv(gym.Env):
    """
    A trading environment where the agent observes market features
    and chooses BUY / HOLD / SELL each day.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: FEATURE_COLS + ['next_return', 'ds']
        Should be sorted by date ascending.
    transaction_cost : float
        Percentage cost per trade (e.g., 0.001 = 0.1%).
        Penalizes excessive position changes.
    reward_scaling : float
        Multiplier for the reward signal. Raw returns (~0.003) are too small
        for PPO's gradient updates; scaling to ~0.3 helps training stability.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, transaction_cost: float = 0.001,
                 reward_scaling: float = 100.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling

        # Action space: 0=SELL, 1=HOLD, 2=BUY
        self.action_space = spaces.Discrete(3)

        # Observation space: 7 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(FEATURE_COLS),),
            dtype=np.float32,
        )

        # State tracking
        self.current_step = 0
        self.prev_action = 1  # Start in HOLD position
        self.total_reward = 0.0

    def _get_observation(self) -> np.ndarray:
        """Return the current observation (feature vector)."""
        row = self.df.iloc[self.current_step]
        obs = np.array([row[col] for col in FEATURE_COLS], dtype=np.float32)
        # Replace any NaN/inf with 0 for stability
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _compute_reward(self, action: int) -> float:
        """
        Compute reward based on the action and actual next-day return.

        Reward logic:
        - BUY  (action=2): reward = +next_return  (profit if price goes up)
        - SELL (action=0): reward = -next_return  (profit if price goes down)
        - HOLD (action=1): reward = -|next_return| × 0.1  (small penalty for missing a move)

        Transaction cost applied when changing position (action != prev_action).
        """
        row = self.df.iloc[self.current_step]
        next_return = float(row.get("next_return", 0.0))

        if np.isnan(next_return):
            next_return = 0.0

        # Base reward from action
        if action == 2:    # BUY
            reward = next_return
        elif action == 0:  # SELL
            reward = -next_return
        else:              # HOLD
            # Small penalty proportional to missed opportunity
            reward = -abs(next_return) * 0.1

        # Transaction cost when changing position
        if action != self.prev_action and action != 1:
            reward -= self.transaction_cost

        # Scale the reward for PPO training stability
        reward *= self.reward_scaling

        return float(reward)

    def reset(self, seed=None, options=None):
        """Reset environment to the beginning of the dataset."""
        super().reset(seed=seed)
        self.current_step = 0
        self.prev_action = 1  # Start in HOLD
        self.total_reward = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        """Execute one time step within the environment."""
        reward = self._compute_reward(action)
        self.total_reward += reward

        # Update state
        self.prev_action = action
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            len(FEATURE_COLS), dtype=np.float32
        )

        info = {
            "action_label": ACTION_MAP[action],
            "total_reward": self.total_reward,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info
```

**Key design decisions:**
- **Reward scaling (100×)**: PPO works better when rewards are in a reasonable range. Raw returns (~0.003) produce tiny gradients.
- **Transaction cost (0.1%)**: Penalizes excessive position changes, encouraging the agent to trade only when confident.
- **HOLD penalty**: Small penalty (10% of missed move) prevents a lazy agent that always holds.
- **NaN handling**: Critical because rolling window features produce NaN at the edges.

---

### 7.2 `app/rl_agent.py` — PPO Agent Wrapper

Wraps `stable-baselines3` PPO with convenience methods for training, prediction, feature importance, and model persistence.

```python
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
    "batch_size": 32,        # Must be ≤ n_steps.
    "n_epochs": 10,          # Gradient updates per rollout.
    "gamma": 0.99,           # High discount factor — future rewards matter in trading.
    "gae_lambda": 0.95,      # Standard GAE lambda.
    "clip_range": 0.2,       # PPO clipping. Prevents catastrophic policy updates.
    "ent_coef": 0.01,        # Entropy bonus. Prevents collapse to always-HOLD.
    "vf_coef": 0.5,          # Value function coefficient.
    "max_grad_norm": 0.5,    # Gradient clipping for stability.
    "verbose": 0,            # Silent (runs inside API requests).
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
```

**Key design decisions:**
- `predict_signal()` uses `model.policy.get_distribution()` to extract the actor's categorical distribution → true probabilities that sum to 1.0
- `compute_feature_importance()` uses perturbation-based importance → same `{feature_name: score}` format as SHAP → frontend bar chart renders unchanged
- `train_agent()` supports `pretrained_model` parameter for transfer learning / fine-tuning

---

### 7.3 `app/rl_data.py` — Offline Pre-training Script

Standalone script for Phase 1 pre-training. Run once before deploying.

```python
"""
rl_data.py
----------
Offline data preparation and PPO pre-training on maximum available yfinance history.

Usage:
    python app/rl_data.py --pretrain
    python app/rl_data.py --pretrain --timesteps 100000
    python app/rl_data.py --pretrain --force-download

This will:
    1. Download max-history price data for all 10 supported stocks via yfinance
    2. Fit Prophet on each stock's full history to compute prophet_gap + forecast_band
    3. Build 7-feature dataframes (sentiment = 0.0 for historical data)
    4. Chain-train PPO across all stocks sequentially (transfer learning)
    5. Save pre-trained model to data/ppo_pretrained.zip
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from prophet import Prophet

warnings.filterwarnings("ignore")

# All supported stocks (from predictor.py)
ALL_SYMBOLS = [
    "^NSEI", "^BSESN", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
    "INFY.NS", "ICICIBANK.NS", "WIPRO.NS", "SBIN.NS", "BHARTIARTL.NS",
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prices")
PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ppo_pretrained")

FEATURE_COLS = [
    "prophet_gap", "forecast_band", "sentiment_1d",
    "sentiment_3d_ma", "sentiment_5d_ma",
    "price_momentum_5d", "volatility_5d",
]


def _safe_symbol_filename(symbol: str) -> str:
    """Convert symbol to safe filename: ^NSEI → NSEI, RELIANCE.NS → RELIANCE_NS"""
    return symbol.replace("^", "").replace(".", "_")


def download_and_cache(symbol: str, force: bool = False) -> pd.DataFrame:
    """
    Download max-period price history via yfinance and cache to CSV.

    Returns DataFrame with columns: [ds, y, next_return]
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{_safe_symbol_filename(symbol)}_max.csv")

    if os.path.exists(cache_path) and not force:
        print(f"[rl_data] {symbol}: Loading from cache {cache_path}")
        df = pd.read_csv(cache_path)
        df["ds"] = pd.to_datetime(df["ds"])
        return df

    print(f"[rl_data] {symbol}: Downloading max history from yfinance...")
    raw = yf.download(symbol, period="max", auto_adjust=True, progress=False)
    if raw.empty:
        print(f"[rl_data] WARNING: No data for {symbol}")
        return pd.DataFrame()

    raw = raw.reset_index()
    df = raw[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    df = df.sort_values("ds").reset_index(drop=True)
    df["next_return"] = df["y"].shift(-1) / df["y"] - 1

    df.to_csv(cache_path, index=False)
    print(f"[rl_data] {symbol}: {len(df)} days cached to {cache_path}")
    return df


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for historical pre-training.
    Uses price-only features; sentiment columns are set to 0.0 (neutral).
    """
    df = df.sort_values("ds").reset_index(drop=True).copy()
    df = df.dropna(subset=["y"])

    if len(df) < 30:
        return pd.DataFrame()

    # Fit Prophet on full history
    prophet_df = df[["ds", "y"]].copy()
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(prophet_df)
    forecast = m.predict(prophet_df[["ds"]])[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    merged = pd.merge(df, forecast, on="ds", how="inner")

    # Price-based features
    merged["daily_ret"]         = merged["y"].pct_change()
    merged["prophet_gap"]       = (merged["yhat"] - merged["y"]) / merged["y"]
    merged["forecast_band"]     = (merged["yhat_upper"] - merged["yhat_lower"]) / merged["y"]
    merged["price_momentum_5d"] = merged["y"].pct_change(5)
    merged["volatility_5d"]     = merged["daily_ret"].rolling(5, min_periods=2).std()

    # Sentiment = 0.0 (neutral placeholder for historical data without news)
    merged["sentiment_1d"]    = 0.0
    merged["sentiment_3d_ma"] = 0.0
    merged["sentiment_5d_ma"] = 0.0

    # Drop NaN rows and last row (no next_return)
    merged = merged.dropna(subset=FEATURE_COLS + ["next_return"])
    merged = merged.iloc[:-1]

    return merged.reset_index(drop=True)


def pretrain_on_all_stocks(total_timesteps_per_stock: int = 50_000,
                            force_download: bool = False) -> None:
    """
    Download all stocks, build price features, chain-train PPO on all of them.

    Chained training = sequential transfer learning:
        Train on stock 1 → [checkpoint A]
        Train on stock 2 starting from A → [checkpoint B]
        ... etc. Each stock starts where the last left off.

    Saves final model to data/ppo_pretrained.zip
    """
    from rl_agent import train_agent, save_agent

    print(f"\n{'='*60}")
    print("  AlphaGaze PPO Pre-training on Maximum Historical Data")
    print(f"{'='*60}")

    all_dfs = []
    for symbol in ALL_SYMBOLS:
        try:
            raw_df = download_and_cache(symbol, force=force_download)
            if raw_df.empty:
                continue
            feature_df = build_price_features(raw_df)
            if len(feature_df) < 100:
                print(f"[rl_data] {symbol}: Not enough data ({len(feature_df)} rows), skipping")
                continue
            feature_df["symbol"] = symbol
            all_dfs.append(feature_df)
            print(f"[rl_data] {symbol}: {len(feature_df)} usable training rows")
        except Exception as e:
            print(f"[rl_data] {symbol}: Failed ({e}), skipping")

    if not all_dfs:
        print("[rl_data] ERROR: No data collected. Cannot pre-train.")
        return

    total_rows = sum(len(d) for d in all_dfs)
    print(f"\n[rl_data] Total training data: {total_rows:,} rows across {len(all_dfs)} stocks")
    print(f"[rl_data] Training {total_timesteps_per_stock:,} timesteps per stock...")

    # Chain-train: each stock starts from the previous stock's model
    model = None
    for i, df in enumerate(all_dfs):
        symbol = df["symbol"].iloc[0]
        print(f"\n[rl_data] [{i+1}/{len(all_dfs)}] Training on {symbol} ({len(df)} days)...")
        model = train_agent(df, total_timesteps=total_timesteps_per_stock,
                            seed=42, pretrained_model=model)

    save_agent(model, PRETRAINED_MODEL_PATH)
    total_steps = total_timesteps_per_stock * len(all_dfs)
    print(f"\n{'='*60}")
    print(f"  Pre-training complete!")
    print(f"  Model:      {PRETRAINED_MODEL_PATH}.zip")
    print(f"  Stocks:     {len(all_dfs)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total steps:{total_steps:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGaze RL offline pre-training")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run offline pre-training on all stocks")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download price data even if cached")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="PPO timesteps per stock (default: 50,000)")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_on_all_stocks(
            total_timesteps_per_stock=args.timesteps,
            force_download=args.force_download,
        )
    else:
        print("Usage:")
        print("  python app/rl_data.py --pretrain")
        print("  python app/rl_data.py --pretrain --timesteps 100000")
        print("  python app/rl_data.py --pretrain --force-download")
```

---

## 8. Implementation — Modified Files

### 8.1 Modify `app/predictor.py`

#### 8.1a. Update imports

**FIND** (lines 25–28):
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
```

**REPLACE WITH**:
```python
from sklearn.ensemble import RandomForestClassifier  # kept for _rolling_backtest fallback only
from sklearn.preprocessing import LabelEncoder

from rl_agent import train_agent, predict_signal, compute_feature_importance, load_agent
from rl_env import ACTION_MAP
```

#### 8.1b. Extend data window to 2 years

**FIND** (lines 587–588):
```python
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365)
```

**REPLACE WITH**:
```python
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years for richer RL training
```

#### 8.1c. Add pre-trained model path constant

**ADD** after the `SUPPORTED_STOCKS` dict (after line 61):
```python
PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ppo_pretrained")
```

Also add `import os` to imports if not already present.

#### 8.1d. Replace RF training + SHAP with PPO training + feature importance

**FIND** the section from line 658 (`# ── 5. Train classifier`) through line 702 (end of SHAP section).

**REPLACE WITH**:
```python
    # ── 5. Train RL agent ────────────────────────────────────────────────────
    # Check for pre-trained model (from offline pre-training)
    pretrained_path = PRETRAINED_MODEL_PATH + ".zip"
    pretrained = load_agent(PRETRAINED_MODEL_PATH) if os.path.exists(pretrained_path) else None

    if pretrained is not None:
        print(f"[predictor] {symbol}: Fine-tuning pre-trained PPO on {len(merged)} samples …")
        rl_model = train_agent(merged, total_timesteps=10_000, seed=42,
                               pretrained_model=pretrained)
    else:
        print(f"[predictor] {symbol}: No pre-trained model. Training PPO from scratch on {len(merged)} samples …")
        rl_model = train_agent(merged, total_timesteps=50_000, seed=42)

    # ── 6. RL Prediction ────────────────────────────────────────────────────
    latest_row = merged.iloc[[-1]]
    X_live = latest_row[FEATURE_COLS].values.flatten()

    rl_result = predict_signal(rl_model, X_live)
    pred_label = rl_result["signal"]
    proba_dict = rl_result["probabilities"]

    # ── 7. Feature Importance (replaces SHAP) ───────────────────────────────
    print(f"[predictor] {symbol}: computing feature importance …")
    shap_dict = compute_feature_importance(rl_model, X_live)

    # CV F1 is no longer applicable for RL
    cv_f1 = None
```

**Note**: Keep the `_label()` function and the labeling code (`merged["signal"] = ...`) — they are still used by `_rolling_backtest()` for measuring accuracy.

#### 8.1e. Replace `_rolling_backtest` to use RL

**REPLACE** the entire `_rolling_backtest` function (lines 466–556) with:

```python
def _rolling_backtest(merged_df: pd.DataFrame, n_points: int = 90,
                      min_train: int = 60) -> dict:
    """Run walk-forward backtest using PPO RL agent."""
    if merged_df is None or len(merged_df) < (min_train + 5):
        return {
            "summary": {"samples": 0, "accuracy": None, "avg_confidence": None},
            "series": [],
            "bucket_hit_rate": [],
        }

    df = merged_df.sort_values("ds").reset_index(drop=True)
    records = []

    for i in range(min_train, len(df)):
        train_data = df.iloc[:i].copy()
        test_row = df.iloc[[i]]

        if len(train_data) < min_train:
            continue

        try:
            # Train agent on data up to this point
            agent = train_agent(train_data, total_timesteps=10_000, seed=42)

            obs = test_row[FEATURE_COLS].values.flatten()
            result = predict_signal(agent, obs)
            pred = result["signal"]
            confidence = max(result["probabilities"].values())
            actual = str(test_row["signal"].iloc[0])

            records.append({
                "date": str(test_row["ds"].iloc[0].date()),
                "predicted": pred,
                "actual": actual,
                "correct": pred == actual,
                "confidence": round(confidence, 4),
                "next_return": round(float(test_row["next_return"].iloc[0]), 6),
            })
        except Exception:
            continue

    if not records:
        return {
            "summary": {"samples": 0, "accuracy": None, "avg_confidence": None},
            "series": [],
            "bucket_hit_rate": [],
        }

    records = records[-n_points:]
    correct = np.array([1 if r["correct"] else 0 for r in records], dtype=float)
    conf = np.array([r["confidence"] for r in records], dtype=float)

    buckets = {"low": (0.0, 0.50), "mid": (0.50, 0.67), "high": (0.67, 1.01)}
    bucket_rows = []
    for name, (lo, hi) in buckets.items():
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        if not idx:
            bucket_rows.append({"bucket": name, "samples": 0, "hit_rate": None})
            continue
        hit_rate = float(np.mean(correct[idx]))
        bucket_rows.append({
            "bucket": name, "samples": len(idx), "hit_rate": round(hit_rate, 4),
        })

    return {
        "summary": {
            "samples": len(records),
            "accuracy": round(float(np.mean(correct)), 4),
            "avg_confidence": round(float(np.mean(conf)), 4),
        },
        "series": records,
        "bucket_hit_rate": bucket_rows,
    }
```

> **WARNING**: Rolling backtest with RL retrains PPO per step → much slower than RF. Mitigations: reduced timesteps (10K), increased min_train (60), and fewer n_points. Consider making it optional if too slow.

---

### 8.2 Modify `app/combine_and_xai.py`

#### 8.2a. Update imports

**FIND** (lines 25–33):
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
```

**REPLACE WITH**:
```python
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

from rl_agent import train_agent, predict_signal, compute_feature_importance, save_agent
from rl_env import ACTION_MAP, SIGNAL_MAP
```

#### 8.2b. Replace RF training + SHAP in `run_combine_and_xai()`

**FIND** section from `# ── 4. TimeSeriesSplit cross-validation` (line 139) through end of SHAP section (~line 227).

**REPLACE WITH**:
```python
    # ── 4. Train PPO RL Agent ───────────────────────────────────────────────
    print("\n[3/5] Training PPO RL agent...")
    X = merged[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(merged["signal"].values)
    classes = le.classes_

    rl_model = train_agent(merged, total_timesteps=50_000, seed=42)

    # Evaluate: run agent over entire dataset
    all_preds = []
    all_true = []
    for i in range(len(merged)):
        row = merged.iloc[[i]]
        obs = row[FEATURE_COLS].values.flatten()
        result = predict_signal(rl_model, obs)
        all_preds.append(SIGNAL_MAP.get(result["signal"], 1))
        all_true.append(y[i])

    overall_f1 = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"\n    ── Weighted F1: {overall_f1:.4f}  |  Macro F1: {macro_f1:.4f} ──")
    print("\n    Full Classification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=classes, zero_division=0))

    # Confusion matrix plot
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(all_true, all_preds, labels=range(len(classes)))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — PPO RL (Weighted F1={overall_f1:.3f})")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()
    print("    Confusion matrix → results/confusion_matrix.png")

    # ── 5. Feature Importance (replaces SHAP) ──────────────────────────────
    print("\n[4/5] Computing feature importance...")

    latest = merged.iloc[[-1]]
    X_live = latest[FEATURE_COLS].values.flatten()
    importance = compute_feature_importance(rl_model, X_live)

    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(importance.keys())
    values = list(importance.values())
    colors = ['#3fb950' if v >= 0 else '#f85149' for v in values]
    ax.barh(features, values, color=colors)
    ax.set_title("Feature Importance — PPO RL Agent (Perturbation-Based)")
    ax.set_xlabel("Impact on Action Probability")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
    print("    Feature importance → results/feature_importance.png")

    # Save the model
    save_agent(rl_model, "results/ppo_trading_agent")
```

---

### 8.3 Modify `app/evaluator.py`

#### 8.3a. Update imports

**FIND** (line 6):
```python
from sklearn.ensemble import RandomForestClassifier
```

**REPLACE WITH**:
```python
from rl_agent import train_agent, predict_signal
from rl_env import SIGNAL_MAP
```

#### 8.3b. Replace RF CV section

**FIND** (lines 65–76):
```python
    cv = TimeSeriesSplit(n_splits=5)
    all_preds, all_true = [], []
    for train_idx, test_idx in cv.split(X):
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        clf.fit(X[train_idx], y_class[train_idx])
        preds = clf.predict(X[test_idx])
        all_preds.extend(preds)
        all_true.extend(y_class[test_idx])
```

**REPLACE WITH**:
```python
    # Train RL agent and evaluate
    rl_model = train_agent(df, total_timesteps=20_000, seed=42)

    all_preds = []
    all_true = list(y_class)
    for i in range(len(df)):
        row = df.iloc[[i]]
        obs = row[FEATURE_COLS].values.flatten()
        result = predict_signal(rl_model, obs)
        all_preds.append(SIGNAL_MAP.get(result["signal"], 1))
```

---

### 8.4 Modify `requirements.txt`

**ADD** these 3 lines at the end:
```
stable-baselines3==2.4.1
gymnasium==1.1.1
shimmy==2.0.0
```

---

### 8.5 Update `agents.MD`

Update these items in the existing `agents.MD`:

1. **Architecture diagram**: Replace `Random Forest Classifier\n300 trees, max_depth=6` → `PPO RL Agent\nstable-baselines3\nDiscrete(3): BUY/HOLD/SELL`
2. **SHAP TreeExplainer** → `Perturbation-Based Feature Importance`
3. **Step 6** description: "Random Forest Training" → "PPO RL Training"
4. **Step 7** description: "SHAP Explanation" → "Feature Importance (Perturbation)"
5. **Dependencies table**: Add `stable-baselines3`, `gymnasium`, `shimmy`
6. **Developer Workflows**: Add `python app/rl_data.py --pretrain` to offline scripts

---

## 9. yfinance Fallback Strategies

yfinance is already working in the project. These are backup plans ONLY for if it stops working.

### Option A: Alpha Vantage API (Free)

```python
import requests

def download_prices_alphavantage(symbol: str, api_key: str = "demo") -> pd.DataFrame:
    av_symbol = symbol.replace(".NS", ".BSE").replace("^NSEI", "NIFTY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&outputsize=full&apikey={api_key}"
    r = requests.get(url).json()
    ts = r.get("Time Series (Daily)", {})
    rows = [{"ds": pd.Timestamp(date), "y": float(vals["4. close"])}
            for date, vals in ts.items()]
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
```

**Limitation**: 25 requests/day on free tier. Get key at https://www.alphavantage.co/support/#api-key

### Option B: Pre-downloaded CSV Files

Store CSV files in `data/prices/`:
```python
def download_prices_csv(symbol: str) -> pd.DataFrame:
    safe_name = symbol.replace("^", "").replace(".", "_")
    path = f"data/prices/{safe_name}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    raise FileNotFoundError(f"No cached price data for {symbol}")
```

### Recommended: Hybrid with graceful fallback

```python
def fetch_prices(symbol: str) -> pd.DataFrame:
    try:
        raw = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
        if not raw.empty:
            prices = raw.reset_index()[["Date", "Close"]]
            prices.columns = ["ds", "y"]
            # Cache for next time
            safe_name = symbol.replace("^", "").replace(".", "_")
            os.makedirs("data/prices", exist_ok=True)
            prices.to_csv(f"data/prices/{safe_name}.csv", index=False)
            return prices
    except Exception as e:
        print(f"[predictor] yfinance failed: {e}")
    return download_prices_csv(symbol)
```

---

## 10. Verification Plan

### 10.1 Automated Tests (run in order)

```bash
# 1. Install new dependencies
cd /home/bufferfis/code/AlphaGaze
source .venv/bin/activate
pip install stable-baselines3==2.4.1 gymnasium==1.1.1 shimmy==2.0.0

# 2. Verify imports
python -c "from app.rl_env import StockTradingEnv; print('✓ rl_env')"
python -c "from app.rl_agent import train_agent, predict_signal; print('✓ rl_agent')"

# 3. Environment smoke test
python -c "
import numpy as np
import pandas as pd
from app.rl_env import StockTradingEnv, FEATURE_COLS

n = 100
df = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=n),
    'y': np.random.randn(n).cumsum() + 100,
    'next_return': np.random.randn(n) * 0.01,
})
for col in FEATURE_COLS:
    df[col] = np.random.randn(n) * 0.1

env = StockTradingEnv(df)
obs, _ = env.reset()
assert obs.shape == (7,)
obs, reward, done, trunc, info = env.step(2)
assert isinstance(reward, float)
print('✓ Environment works')
"

# 4. Agent training smoke test
python -c "
import numpy as np
import pandas as pd
from app.rl_env import FEATURE_COLS
from app.rl_agent import train_agent, predict_signal, compute_feature_importance

n = 100
df = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=n),
    'y': np.random.randn(n).cumsum() + 100,
    'next_return': np.random.randn(n) * 0.01,
})
for col in FEATURE_COLS:
    df[col] = np.random.randn(n) * 0.1

model = train_agent(df, total_timesteps=1000, seed=42)
obs = df.iloc[-1][FEATURE_COLS].values
result = predict_signal(model, obs)
assert result['signal'] in ('BUY', 'HOLD', 'SELL')
assert set(result['probabilities'].keys()) == {'BUY', 'HOLD', 'SELL'}
assert abs(sum(result['probabilities'].values()) - 1.0) < 0.01

importance = compute_feature_importance(model, obs, n_perturbations=10)
assert set(importance.keys()) == set(FEATURE_COLS)
print(f'✓ Signal: {result[\"signal\"]}, probs: {result[\"probabilities\"]}')
print(f'✓ Importance: {importance}')
"

# 5. Full pipeline test
uvicorn app.api:app --port 8000 &
sleep 5
curl -s http://localhost:8000/api/predict/INFY.NS | python -m json.tool | head -20
kill %1
```

### 10.2 Manual Verification

After starting the server:

1. Open `http://localhost:8000` in browser
2. Select any stock (e.g., INFY.NS)
3. Verify these panels render correctly:
   - ✅ Signal badge shows BUY / HOLD / SELL
   - ✅ Probability bar chart shows 3 bars summing to ~100%
   - ✅ Feature impact bar chart shows 7 features with bars
   - ✅ Prophet forecast chart renders
   - ✅ Attention heatmaps render
   - ✅ Rolling backtest shows results (may take longer)
   - ✅ Uncertainty/confidence gauge works

### 10.3 API Response Correctness

```bash
python -c "
import requests
r = requests.get('http://localhost:8000/api/predict/INFY.NS').json()
p = r['probabilities']
assert abs(sum(p.values()) - 1.0) < 0.02, f'Probs sum to {sum(p.values())}'
assert r['signal'] in ('BUY', 'HOLD', 'SELL')
assert set(r['shap_values'].keys()) == {'prophet_gap', 'forecast_band', 'sentiment_1d', 'sentiment_3d_ma', 'sentiment_5d_ma', 'price_momentum_5d', 'volatility_5d'}
print(f'✓ Signal: {r[\"signal\"]}')
print(f'✓ Probabilities: {p}')
print(f'✓ Actionability: {r[\"uncertainty\"][\"actionability\"]}')
"
```

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| PPO training is slow (~30-60s per stock from scratch) | API response time increases | Pre-trained model reduces fine-tuning to ~5-10s. Cache results for 1 hour. |
| Small dataset (~200 days) may cause overfitting | Agent memorizes patterns | Entropy bonus (`ent_coef=0.01`), transaction costs, pre-training on 50K+ days. |
| Rolling backtest becomes very slow (retrains per step) | Backtest takes minutes | Reduce timesteps (10K), increase `min_train` (60), reduce `n_points` (30). Consider making optional. |
| SHAP TreeExplainer no longer available | Feature importance less precise | Perturbation-based importance produces same format, well-established method. |
| `predict_signal()` API may differ across SB3 versions | Broken probability extraction | Pin `stable-baselines3==2.4.1` exactly. |
| Agent always picks HOLD (lazy policy) | Useless predictions | HOLD penalty in reward, entropy bonus encourages exploration. |
| yfinance rate limiting during pre-training | Downloads fail | Cache all data to CSV. Re-run with `--force-download` only when needed. |

> **Performance expectation**: RL models for stock trading rarely outperform simple baselines on raw classification accuracy. The RF currently has ~48% confidence. PPO may have similar or lower classification accuracy but should produce better **trading returns** because it optimizes profit, not label matching.

---

## 12. Execution Checklist

Step-by-step checklist. Execute in this order.

### Phase 0: Dependencies
- [ ] Add `stable-baselines3==2.4.1`, `gymnasium==1.1.1`, `shimmy==2.0.0` to `requirements.txt`
- [ ] Run `pip install -r requirements.txt`

### Phase 1: New Files
- [ ] Create `app/rl_env.py` — copy from [Section 7.1](#71-apprl_envpy--the-trading-environment)
- [ ] Create `app/rl_agent.py` — copy from [Section 7.2](#72-apprl_agentpy--ppo-agent-wrapper)
- [ ] Create `app/rl_data.py` — copy from [Section 7.3](#73-apprl_datapy--offline-pre-training-script)
- [ ] Run import smoke tests (Section 10.1, commands 2-4)

### Phase 2: Modify Existing Files
- [ ] Modify `app/predictor.py` — imports (8.1a)
- [ ] Modify `app/predictor.py` — data window to 2 years (8.1b)
- [ ] Modify `app/predictor.py` — add PRETRAINED_MODEL_PATH constant (8.1c)
- [ ] Modify `app/predictor.py` — replace RF with PPO (8.1d)
- [ ] Modify `app/predictor.py` — update `_rolling_backtest` (8.1e)
- [ ] Modify `app/combine_and_xai.py` — imports (8.2a)
- [ ] Modify `app/combine_and_xai.py` — replace RF + SHAP (8.2b)
- [ ] Modify `app/evaluator.py` — imports (8.3a)
- [ ] Modify `app/evaluator.py` — replace RF CV (8.3b)
- [ ] Modify `requirements.txt` (8.4)

### Phase 3: Verification
- [ ] Run full pipeline smoke test (Section 10.1, command 5)
- [ ] Manual browser verification (Section 10.2)
- [ ] API response correctness check (Section 10.3)

### Phase 4: Offline Pre-training (Optional but Recommended)
- [ ] Run `python app/rl_data.py --pretrain`
- [ ] Verify `data/ppo_pretrained.zip` exists
- [ ] Restart server and verify fine-tuning path is used

### Phase 5: Documentation
- [ ] Update `agents.MD` (Section 8.5)
