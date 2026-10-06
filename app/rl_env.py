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
        - HOLD (action=1): reward = -|next_return| * 0.1  (small penalty for missing a move)

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
