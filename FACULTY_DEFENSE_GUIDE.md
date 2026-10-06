# AlphaGaze: Faculty Defense Guide

This document is a complete, implementation-grounded explanation of AlphaGaze. It is written for an oral review in which the examiner may begin with “What is a state?” and gradually move to PPO mathematics, data leakage, financial assumptions, and whether the system is really reinforcement learning.

This is the second repository audit of the guide. It was checked against the current Python modules, frontend JavaScript, stored artifacts, requirements, README, implementation plan, and local agent context. “Current” below means the checked-in implementation, not an aspirational plan or an older generated result.

The safest way to present the project is:

> AlphaGaze is a multimodal stock-signal dashboard. Prophet extracts price trend and uncertainty features, FinBERT converts financial news into a sentiment signal, and a PPO policy chooses one of three directional actions—SELL, HOLD, or BUY. The policy is trained in a custom Gymnasium environment whose reward is based on the next observed return and a transaction-cost penalty. The current implementation is a compact, single-asset, daily directional simulator; it is not a complete brokerage-grade portfolio simulator.

That last sentence is important. It is accurate, defensible, and prevents overclaiming.

## How to use this guide in a viva

Start with the “30-second answer” and the pipeline diagram. If the examiner asks for a basic definition, use the plain-language answer first and then give the project-specific answer. If the examiner asks for a limitation, state it directly and explain the planned fix. Do not call policy probabilities “confidence” without qualification: they are probabilities from the PPO policy distribution and have not been statistically calibrated.

The references below point to the current code. The older README and parts of `IMPLEMENTATION_PLAN.md` describe the earlier Random Forest design; where they disagree with the implementation, the implementation is the source of truth.

## 30-second answer

AlphaGaze receives daily price history and recent financial headlines for a supported Indian stock. Prophet produces a fitted value and an uncertainty band. FinBERT scores each headline by positive probability minus negative probability, and those scores are aggregated by day. Seven values are fed to a PPO agent: Prophet gap, Prophet band width, one-day sentiment, three-day sentiment average, five-day sentiment average, five-day price momentum, and five-day volatility. The custom environment maps action `0/1/2` to SELL/HOLD/BUY and rewards the action using the return to the next retained row, subtracting a 0.1% cost when a non-HOLD action changes. PPO learns a categorical policy over the three actions. At inference, the action with the largest policy probability is returned, along with the three probabilities and local perturbation-based feature impact. The FastAPI backend serves this result to the dashboard.

## 1. What the project contains

| Layer | Current implementation | Purpose |
|---|---|---|
| Price data | `yfinance` in `app/predictor.py` and `app/rl_data.py` | Historical daily close prices and online refresh |
| News data | Google News RSS in `app/scraper.py` | Recent headlines, dates, and URLs |
| Financial NLP | FinBERT (`yiyanghkust/finbert-tone`) in `app/predictor.py` and `app/sentiment.py` | Financially contextual sentiment scores |
| Time series | Prophet in `app/predictor.py`, `app/time_series.py`, and `app/rl_data.py` | Trend, weekly/yearly seasonality, fitted value, uncertainty band, changepoints |
| Feature fusion | `_build_features()` in `app/predictor.py` | Seven-dimensional observation vector |
| RL environment | `StockTradingEnv` in `app/rl_env.py` | Defines actions, observations, transitions, and rewards |
| RL algorithm | PPO from Stable-Baselines3 in `app/rl_agent.py` | Learns the policy and value function |
| Pre-training | `app/rl_data.py` | Caches maximum price history and chains training across ten instruments |
| Online prediction | `predict()` in `app/predictor.py` | Rebuilds data, fine-tunes PPO, predicts one current action |
| Explainability | Policy perturbation, FinBERT attention display, Prophet sensitivity | Gives local diagnostic views |
| API | FastAPI in `app/api.py` | Serves stocks, predictions, news, and evaluation |
| UI | `app/frontend/index.html` | Renders the signal, probabilities, charts, news, and explanations |
| Experiments | `experiments/*.ipynb` | Earlier Prophet, FinBERT, SHAP, and baseline demonstrations |

The supported instruments are NIFTY 50, Sensex, Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Wipro, SBI, and Bharti Airtel. The repository also contains cached price CSVs and `data/ppo_pretrained.zip`.

## 2. End-to-end architecture

```text
                           ┌────────────────────────┐
                           │ FastAPI /api/predict   │
                           └────────────┬───────────┘
                                        │ symbol
                    ┌───────────────────┴───────────────────┐
                    │                                       │
          yfinance price history                    Google News RSS
                    │                                       │
                    ▼                                       ▼
             Prophet fit/forecast                         headlines
                    │                                       │
                    │                                  FinBERT
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    │ daily merge
                                    ▼
                         seven engineered features
                                    │
                                    ▼
                         StockTradingEnv (Gymnasium)
                                    │
                                    ▼
                          PPO actor + critic (MLP)
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
             categorical policy                 value estimate
                    │                                │
                    ▼                                │
             SELL/HOLD/BUY + probabilities          │
                    │                                │
                    └───────────────┬────────────────┘
                                    ▼
                       FastAPI JSON → dashboard
```

At startup the API exposes the frontend. A prediction request is cached for one hour by symbol. A cache miss downloads roughly two years of prices, fits Prophet, scrapes current headlines, scores them with FinBERT, merges rows that have both price and news, trains or fine-tunes PPO, computes explanations, and returns JSON.

## 3. Reinforcement learning from first principles

### What is learning?

Learning means changing internal parameters after seeing examples so that future decisions improve according to a chosen objective. A supervised model is shown an input and a target answer. An RL agent is shown a situation, chooses an action, observes a consequence, receives a reward, and improves a policy that maps situations to actions.

### What is an agent?

An agent is the decision maker. In AlphaGaze, the PPO neural network is the agent. It receives the seven-number observation and produces a distribution over SELL, HOLD, and BUY.

### What is an environment?

An environment is the world that responds to the agent. In AlphaGaze, `StockTradingEnv` is a historical replay simulator. It presents one row at a time, accepts an action, looks up that row’s `next_return`, computes a reward, advances to the next row, and eventually ends the episode. In a complete daily-price table this is the next trading-day return; after the online news inner join it can span multiple trading days, a caveat covered later.

### What is a state?

In plain language, a state is everything relevant about the situation at a particular time. In a trading simulator, a complete state might include the date, price, features, current position, cash, portfolio value, and any other information needed to predict what happens next.

For this code, the environment’s internal state is approximately:

```text
(current_step, previous_action, dataframe, cumulative_reward)
```

The agent does **not** receive all of that. The observation returned to the neural network is only this seven-element vector:

```text
[prophet_gap,
 forecast_band,
 sentiment_1d,
 sentiment_3d_ma,
 sentiment_5d_ma,
 price_momentum_5d,
 volatility_5d]
```

The precise statement is therefore: “The environment has an internal state; the policy receives a seven-feature observation.” Calling the seven numbers the full Markov state would be too strong because `previous_action` affects transaction cost but is not included in the observation. The current observation is a partial view of the internal state.

### What is an observation?

An observation is the information made available to the agent. It is often equal to the state in a simple environment, but it can be incomplete. AlphaGaze’s observation is a `float32` vector of shape `(7,)`, defined by `spaces.Box(low=-inf, high=inf, shape=(7,))` in `app/rl_env.py`.

### What is an action?

An action is the decision taken by the agent. AlphaGaze has a discrete action space:

| Integer | Label | Directional meaning |
|---:|---|---|
| 0 | SELL | Benefit if the next return is negative |
| 1 | HOLD | Avoid a directional bet, but receive a small missed-move penalty |
| 2 | BUY | Benefit if the next return is positive |

The code calls these “positions,” but there is no explicit share quantity, cash balance, margin, or inventory. The safest interpretation is that an action is a one-day directional exposure or signal, not an executable order ticket.

### What is an action space?

The action space is the set of legal actions. `spaces.Discrete(3)` means exactly one of three integer actions is legal at each step. This is why a categorical policy is appropriate.

### What is a transition?

A transition is one move from the current situation to the next one:

```text
observation_t → action_t → reward_t, observation_(t+1)
```

The environment computes the reward using `next_return` from row `current_step`, then increments `current_step`. It ends when `current_step >= len(df) - 1` and returns a zero observation at termination.

### What is a reward?

A reward is the numeric score used to teach the agent. Positive reward means the action was useful under the chosen objective; negative reward means it was costly or wrong. It is not automatically the same thing as raw profit unless the reward equation faithfully represents the trading account.

For the row’s target return `r` (normally a next-day return, but potentially the next retained-news-date return online), transaction cost `c = 0.001`, and scaling `k = 100`, the implemented reward is:

```text
BUY:  ( r - cost_if_action_changed ) × 100
SELL: (-r - cost_if_action_changed ) × 100
HOLD: (-0.1 × |r|) × 100
```

The cost is subtracted only when `action != previous_action` and the new action is not HOLD. A change from BUY to HOLD therefore has no cost in this implementation. The reward is scaled after the cost is applied.

### Worked reward example

Suppose the target return is `+0.8% = 0.008`, and the previous action was HOLD.

| Action | Unscaled reward | Scaled reward |
|---|---:|---:|
| BUY | `0.008 - 0.001 = 0.007` | `0.70` |
| SELL | `-0.008 - 0.001 = -0.009` | `-0.90` |
| HOLD | `-0.1 × 0.008 = -0.0008` | `-0.08` |

If the target return is `-0.5%`, SELL receives a positive directional reward, BUY receives a negative directional reward, and HOLD receives a small negative reward. This explains exactly what “the agent learns from returns” means in this project.

### What is a policy?

A policy is the agent’s decision rule. A deterministic policy maps each observation to one action. A stochastic policy maps an observation to a probability distribution over actions. PPO learns a stochastic categorical policy and `predict_signal()` takes the argmax of that distribution for the displayed signal.

For example, if the actor produces:

```text
P(SELL | observation) = 0.10
P(HOLD | observation) = 0.25
P(BUY  | observation) = 0.65
```

the API returns BUY, while the probability chart shows 65%, 25%, and 10% in its BUY/HOLD/SELL order. The policy distribution sums to one. These are model probabilities, not guaranteed frequencies of correctness.

### What is a value function?

The value function estimates expected future discounted reward from a situation:

```text
V(s) = expected future return when starting from s and following the policy
```

PPO uses a critic network to estimate this quantity. The critic is not the displayed stock price forecast; Prophet handles price forecasting. The critic estimates the quality of states for the RL reward objective.

### What is a return?

The return is the sum of future rewards, usually discounted:

```text
G_t = r_t + γr_(t+1) + γ²r_(t+2) + ...
```

With `gamma = 0.99`, a reward one day later is multiplied by 0.99, two days later by 0.99², and so on. This makes the agent care about a sequence of decisions rather than only the immediate reward.

### What is an episode?

An episode is one complete run from `reset()` to termination. Here, an episode starts at the first usable historical row and walks through the rows in date order until the penultimate row. PPO repeatedly resets and replays the same historical episode during training.

### What is exploration?

Exploration means trying actions whose value is uncertain. Exploitation means choosing the action currently believed to be best. PPO explores through its stochastic policy and the entropy bonus `ent_coef = 0.01`. In inference, AlphaGaze exploits by choosing the highest-probability action.

### What is an MDP?

An MDP, or Markov Decision Process, is commonly described by `(S, A, P, R, γ)`:

| Symbol | Meaning | AlphaGaze interpretation |
|---|---|---|
| `S` | States | Internal simulator state; only seven features are observed |
| `A` | Actions | SELL, HOLD, BUY |
| `P` | Transition dynamics | Move to the next chronological row; deterministic historical replay |
| `R` | Reward | Directional next return, HOLD penalty, transaction cost, ×100 |
| `γ` | Discount | 0.99 |

The environment is not a learned market simulator. It is a fixed historical replay. The future return is known to the environment during training only so it can calculate the reward after the action.

## 4. The seven features, explained at two levels

| Feature | Baby-level explanation | Code-level definition and interpretation |
|---|---|---|
| `prophet_gap` | How far Prophet’s estimate is from today’s price | `(yhat - y) / y`. Positive means Prophet’s fitted value is above the actual price; negative means below. |
| `forecast_band` | How wide Prophet’s uncertainty interval is | `(yhat_upper - yhat_lower) / y`. Larger values mean a wider model interval relative to price. |
| `sentiment_1d` | The mood of the news for that day | Mean FinBERT score for that date, where score is positive-class probability minus negative-class probability. |
| `sentiment_3d_ma` | A short smoothing of recent news mood | Three-day rolling mean of the daily sentiment. |
| `sentiment_5d_ma` | A slightly slower smoothing of news mood | Five-day rolling mean of the daily sentiment. |
| `price_momentum_5d` | Whether price has risen or fallen over five trading days | `y.pct_change(5)`. Positive means the current price is above the price five rows earlier. |
| `volatility_5d` | How much recent daily returns have moved around | Standard deviation of daily percentage returns over five rows. |

`_build_features()` in `app/predictor.py` and `build_price_features()` in `app/rl_data.py` implement these formulas. The Prophet gap and band are normalized by actual price so the feature is scale-independent across instruments. Rolling features at the beginning of a series can be missing; rows with missing required features are dropped before training. At the final observation, NaN and infinite values are defensively converted to zero by the environment and prediction helper.

### Why include both raw and moving-average sentiment?

The one-day value reacts quickly to a new headline. The three-day and five-day means capture persistence and reduce sensitivity to one unusual article. They are correlated by design, so their separate importance values must not be interpreted as independent causal contributions.

### Why include both momentum and Prophet gap?

They describe different ideas. Momentum is a purely price-based recent direction. Prophet gap compares the observed price with Prophet’s fitted trend/seasonality estimate. A policy may use agreement between them, disagreement, or the size of the disagreement.

### Why include forecast-band width?

The predicted level says where the model points; the band says how uncertain that forecast is. A high band can support HOLD or lower conviction, although PPO is free to learn any relationship from the reward data.

## 5. What PPO is and what happens during training

PPO means Proximal Policy Optimization. It is an actor-critic, policy-gradient method.

### Actor and critic

The actor receives the seven features and outputs three logits. A categorical distribution converts the logits into three action probabilities. The critic receives the same observation and outputs one scalar value estimate. In Stable-Baselines3, `PPO("MlpPolicy", ...)` creates the feed-forward policy/value model.

```text
seven features
      │
      ▼
 MLP feature layers
      ├──────────────► actor logits → categorical SELL/HOLD/BUY probabilities
      └──────────────► critic scalar → V(observation)
```

### Why not update the policy without protection?

A plain policy-gradient update can change the policy too aggressively after a noisy return. PPO stores the old policy probability of the chosen action and compares it with the new policy probability:

```text
r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
```

It estimates an advantage `A_t`, which asks whether the chosen action performed better or worse than the critic expected. The clipped surrogate objective is:

```text
L_CLIP = E[min(r_t A_t,
               clip(r_t, 1 - ε, 1 + ε) A_t)]
```

The code uses `epsilon = clip_range = 0.2`. If a new policy moves too far from the old policy in a direction that would improve the objective, the clipped term stops rewarding further movement for that sample.

Important precision: clipping does not guarantee that every action probability changes by at most exactly 20%. It clips the probability ratio inside the surrogate objective. That is why the defensible statement is “PPO discourages destructive policy updates,” not “PPO mathematically caps every policy probability at 20% change.”

### Generalized Advantage Estimation

The configuration uses `gae_lambda = 0.95`. GAE combines short-term temporal-difference errors over multiple horizons to produce a lower-variance, reasonably responsive advantage estimate. `gamma = 0.99` controls economic discounting; `gae_lambda` controls the bias/variance trade-off of the estimator. They are different parameters.

### Bellman view of the critic

The critic is trained toward a bootstrapped target. In the simplest one-step form:

```text
target_t = r_t + γ V(s_(t+1))
TD error δ_t = target_t − V(s_t)
```

At a terminal transition there is no future value term. GAE combines these TD errors:

```text
A_t ≈ δ_t + (γλ)δ_(t+1) + (γλ)²δ_(t+2) + ...
```

The actor wants to increase the probability of actions with positive advantage and decrease the probability of actions with negative advantage. The critic minimizes a value prediction error, commonly a squared error or clipped value loss. Entropy is added as a regularizer. Stable-Baselines3 combines the policy, value, and entropy terms internally; the project sets their relative coefficients through `vf_coef` and `ent_coef`.

### Policy gradient versus supervised gradient

In supervised learning, a target class is available for each input and the gradient directly reduces classification loss. In PPO, the action sampled by the current policy is scored after the environment returns a reward. The gradient is weighted by the estimated advantage, so the algorithm learns from “how much better or worse was this chosen action than expected?” This is why PPO can optimize a reward that includes transaction cost instead of merely matching a threshold label.

### Value function versus Prophet forecast

Both produce numbers, but they answer different questions. Prophet estimates a price-series quantity such as a fitted level or forecast interval. The PPO critic estimates expected future **RL reward** from an observation under the current policy. The critic is not an alternative stock-price forecaster, and replacing one with the other would change the meaning of the model.

### PPO update loop in this project

1. Reset the environment and observe the first feature vector.
2. Sample or select an action from the current categorical policy.
3. Step the environment; receive reward and the next observation.
4. Store observation, action, reward, old log probability, and value estimate.
5. Repeat for `n_steps = 64` rollout steps.
6. Compute returns and GAE advantages.
7. Split the rollout into batches of 32.
8. Perform `n_epochs = 10` optimization passes using the clipped actor loss, critic loss, and entropy term.
9. Repeat until `total_timesteps` is reached.

`DummyVecEnv([_init])` supplies the vectorized environment interface expected by Stable-Baselines3, even though only one environment is used.

### PPO configuration and defensible rationale

| Parameter | Current value | Explanation |
|---|---:|---|
| `learning_rate` | `3e-4` | Standard PPO starting value; controls optimizer step size. |
| `n_steps` | `64` | Rollout length chosen to be compatible with small daily datasets; 2048 would be larger than a typical online window. |
| `batch_size` | `32` | Two minibatches per 64-step rollout; must not exceed `n_steps`. |
| `n_epochs` | `10` | Reuses each collected rollout several times; increases data efficiency but can overfit repeated historical data. |
| `gamma` | `0.99` | Retains long-horizon reward information. |
| `gae_lambda` | `0.95` | Standard advantage-estimation compromise. |
| `clip_range` | `0.2` | Limits incentive for large policy-ratio changes. |
| `ent_coef` | `0.01` | Adds entropy reward to discourage premature deterministic collapse. |
| `vf_coef` | `0.5` | Balances value-function loss against policy loss. |
| `max_grad_norm` | `0.5` | Clips gradient norm for numerical stability. |
| `device` | `cpu` | Fits the API’s local inference/training target and the small MLP. |

These are reasonable engineering defaults, not results of a full hyperparameter search. A strong answer is: “The values were selected for a small, CPU-only proof-of-concept and should be tuned with a time-ordered validation protocol.”

### Why scale rewards by 100?

Daily returns are often fractions such as `0.003`. Multiplying all rewards by a positive constant makes optimization signals numerically larger without changing which action is better at a single step. It can change optimization dynamics and value magnitudes, so it is not mathematically irrelevant to training, but it does not reverse reward ordering. A claim that scaling alone improves financial performance would require an experiment.

## 6. Why RL rather than a standard classifier?

### The intended motivation

A Random Forest classifier learns to reproduce a label such as “next-day return exceeded +0.3%.” That objective is not the same as maximizing a trading reward. RL permits the objective to include direction, transaction costs, and sequential consequences.

The project’s original labels remain in the data pipeline because the current diagnostics and rolling backtest use them. PPO training itself uses `next_return` in `_compute_reward`; it does not use the `signal` label to calculate its gradient.

### The important qualification

Because the current environment has no cash, holdings, position size, slippage model, or multi-day portfolio accounting, it is closer to a sequential directional-action simulator or contextual bandit than a full portfolio-management MDP. The sequential episode and previous-action cost make it RL-shaped, but the economic realism is limited. This is a limitation to acknowledge, not hide.

### Why not keep Random Forest?

Random Forest is still a valuable baseline and may outperform PPO on classification metrics. It is easier to train, naturally handles tabular features, and offers strong feature importance tools. The reason to use PPO is objective alignment: the project wants a policy that chooses actions under a reward with transaction costs and exposes a stochastic categorical policy. The claim should be “PPO is a better fit for the intended decision objective,” not “PPO is universally more accurate.”

## 7. Why PPO and not other RL methods?

| Method | What it does | Why it was not selected here | Fair qualification |
|---|---|---|---|
| REINFORCE | Direct policy gradient using full returns | Very high variance; no critic or clipping | Simple educational baseline, but likely unstable on repeated noisy market episodes |
| A2C | Actor-critic with synchronous updates | Has a critic but no PPO-style ratio clipping; can make larger updates | Could work and is faster, but needs stronger stability evidence |
| DQN | Learns action values with a replay buffer | Discrete actions fit, but output is Q-values rather than a native policy distribution; the dashboard specifically needs policy probabilities | DQN is not invalid. A calibrated softmax over Q-values is possible, but it is temperature-dependent and not the same as a learned stochastic policy |
| Double/Dueling/Rainbow DQN | Improved value-based DQN families | More machinery than justified by the current dataset and no native policy probabilities | Worth benchmarking if the project changes its output contract to value scores |
| SAC | Entropy-regularized actor-critic | Standard Stable-Baselines3 SAC targets continuous actions; discrete SAC needs a different implementation | Discrete SAC exists in the literature, so “impossible” would be incorrect |
| DDPG | Deterministic continuous-control actor-critic | Wrong native action space; designed for continuous actions | Could be used only after redefining the action as a continuous position size |
| TD3 | More stable DDPG variant | Same continuous-action mismatch | Strong for continuous control, irrelevant to the current three-action interface |
| TRPO | Trust-region policy optimization | More expensive and complex than PPO for this small deployment | PPO approximates a trust-region idea with a simple clip objective |
| Recurrent PPO | PPO with memory | Current observation is a small tabular vector and the implementation uses feed-forward `MlpPolicy` | Recurrent memory may help if history is not fully summarized by rolling features |
| Contextual bandit | One-step action/reward mapping | Would match the one-day reward more honestly, but would not model longer sequences or changing action costs | A useful baseline and possibly the more faithful formulation for the current simulator |
| Transformer policy | Attention over long sequences | Data volume and CPU/API constraints do not justify it | More expressive is not automatically better on a small, noisy market dataset |

### The three strongest PPO arguments

1. **Discrete categorical policy:** the three actions map directly to a `Categorical` distribution and the dashboard’s probability chart.
2. **Conservative updates:** PPO’s clipped surrogate objective is a practical guard against large policy changes after noisy returns.
3. **Actor-critic efficiency:** the critic reduces the variance of policy-gradient learning compared with plain REINFORCE.

The honest counterargument is that PPO is on-policy and therefore data-inefficient. The system partly addresses this with offline pre-training and fine-tuning, but it still replays a small historical dataset many times and can overfit.

## 8. Data and training strategy

### Online pipeline

`predict(symbol)` does the following:

1. Checks the in-memory one-hour cache.
2. Downloads about 730 calendar days of adjusted daily prices.
3. Fits Prophet with weekly and yearly seasonality and creates a 30-day forecast.
4. Fetches up to 150 current Google News RSS articles.
5. Scores headlines with FinBERT in batches.
6. Groups sentiment by date and performs an **inner** merge with prices and Prophet output.
7. Builds the seven features and the return to the next retained row.
8. Creates the ±0.3% BUY/HOLD/SELL label for diagnostics.
9. Loads `data/ppo_pretrained.zip` if present and fine-tunes for 10,000 timesteps; otherwise trains from scratch for 50,000 timesteps.
10. Predicts the latest action, computes perturbation impact, attention maps, Prophet sensitivity, and a rolling backtest.
11. Returns the result and caches it.

If the news feed produces too few overlapping dates, the function raises a “not enough overlapping data” error. Because the merge is inner, a missing-news date is removed rather than treated as neutral.

### Offline pre-training

`python app/rl_data.py --pretrain`:

1. Downloads maximum available price history for ten symbols and caches it under `data/prices/`.
2. Fits Prophet to each stock’s full cached history.
3. Builds price features.
4. Sets all three sentiment features to zero because historical news is not available from the RSS source.
5. Trains one PPO model for 50,000 timesteps per stock, sequentially passing the model into the next stock’s training call.
6. Saves the final model to `data/ppo_pretrained.zip`.

This is transfer learning across instruments, not an ensemble. The last stock’s training continues from the weights learned on the previous stocks. It may transfer general directional patterns, but the sequential order can bias the final policy toward later instruments. A stronger future design would shuffle training environments or use a jointly sampled multi-stock environment.

The current cached raw-price snapshot contains approximately 68,396 rows across the ten CSV files (before rolling-feature warm-up and final-return drops): NSEI 4,659; BSESN 7,194; HDFC Bank 7,709; ICICI Bank 6,011; Infosys 7,709; Reliance 7,706; SBI 7,707; TCS 5,982; Wipro 7,709; and Bharti Airtel 6,011. The older plan’s “about 50,000 days” is a rounded design estimate, not the measured count for this snapshot. The number of usable feature rows is smaller and should be read from the pre-training log.

### Why sentiment is zero during pre-training

Zero is used as a neutral placeholder rather than inventing historical headlines. This gives the policy more price-only exposure, then online fine-tuning introduces real recent sentiment. The drawback is distribution shift: the policy sees three exact zero features for most pre-training rows and nonzero news features during online use. This should be measured, not assumed harmless.

## 9. Leakage and evaluation issues an examiner may find

These are the most important technical caveats in the current repository.

### Prophet is fitted on the full historical window

For each row used to train the policy, Prophet is fitted using the entire available window and then asked to predict dates inside that same window. A historical row can therefore receive a feature influenced by later prices. In a strict forecasting experiment, Prophet must be refit using only data available before each decision date, or features must be generated with a proper walk-forward procedure.

### The current PPO classification diagnostics are in-sample

`combine_and_xai.py` trains PPO on all merged rows and then predicts those same rows. Its weighted and macro F1 values are descriptive in-sample scores, not a generalization estimate. `evaluator.py` also trains one model and evaluates it on the same dataset. The live `predict()` response deliberately sets `cv_weighted_f1` to `None` because PPO is no longer evaluated with the old cross-validation field.

### The evaluator uses synthetic sentiment and a mock band

`app/evaluator.py` creates random sentiment with a fixed seed and sets `forecast_band = 0.05`. Its sentiment-agreement rate compares FinBERT with a simplistic word-set proxy, not human labels. Those metrics should never be presented as real trading performance.

### Label/action encoding deserves care

`LabelEncoder` orders strings alphabetically (`BUY`, `HOLD`, `SELL`), while `SIGNAL_MAP` uses (`SELL: 0`, `HOLD: 1`, `BUY: 2`). The code compares those integer arrays in the offline/evaluator diagnostics. A defensible presentation should report this as a known evaluation bug to fix by comparing string labels or using one explicit mapping everywhere.

### Rolling backtest is not a fully independent research backtest

The rolling function fine-tunes a model for each historical test row and catches exceptions silently. It uses the pre-trained model object as a continuing base, and it reports classification accuracy and policy confidence rather than cumulative net return, turnover, drawdown, and Sharpe ratio. It is useful as an application diagnostic, not sufficient evidence of investment performance.

### The reward is not a complete account simulator

The environment does not model cash, holdings, leverage, short-sale constraints, slippage, market impact, position sizing, overnight gaps beyond the single return, or portfolio-level risk. “Profit” in the code means directional reward under the stated formula.

### Policy probability is not calibrated confidence

The probability chart is the actor’s categorical distribution. It does not prove that actions with probability 0.70 are correct 70% of the time. Calibration requires held-out predictions, reliability diagrams, Brier score, or expected calibration error.

## 10. Explainability: what is shown and what it means

### PPO feature impact

`compute_feature_importance()` keeps the current observation fixed, perturbs one feature at a time with Gaussian noise, reruns the policy, and measures:

```text
baseline_probability_of_baseline_action
    − perturbed_probability_of_baseline_action
```

It averages 50 perturbations per feature and returns the same dictionary shape that the frontend expects under `shap_values`.

Interpretation: a positive score means the perturbations tended to reduce the baseline action’s probability; a negative score means they tended to increase it. This is a local sensitivity diagnostic. It is not SHAP, not a causal effect, and not a global feature ranking. Correlated features and perturbations outside the realistic data distribution can distort it.

### FinBERT attention maps

The code averages the last-layer attention heads and uses the CLS row to display token weights. This is useful as a visualization of where attention was allocated, but attention weights alone are not a faithful proof of model causality. The counterfactual test—removing a high-attention token and rescoring the headline—is a stronger sanity check, although it still changes the text unnaturally.

### Prophet sensitivity and changepoints

The code perturbs segments of the price history with noise, refits Prophet, and measures the change in the average future forecast. It also extracts Prophet changepoints and their fitted delta magnitudes. This is a model sensitivity analysis, not proof that a market event caused a changepoint.

### Uncertainty summary

`_compute_uncertainty_summary()` combines normalized policy entropy, top-two probability margin, forecast-band width, and sentiment polarity with fixed weights `0.45/0.25/0.20/0.10`. The result is a human-facing heuristic labeled TRADE, CAUTION, or NO_TRADE. It is not a statistically learned risk model.

## 11. How to evaluate the project properly

If asked how the work should be validated, give this protocol:

1. Sort all data chronologically and define a final untouched test interval.
2. For every decision date, build Prophet features using only prices available up to that date.
3. Include only headlines published by the decision time; never use future article timestamps.
4. Fit feature scaling or normalization only on the training portion if scaling is added.
5. Train PPO on the historical training window and evaluate on the next chronological window without updating on it.
6. Repeat this walk-forward process across several windows and several random seeds.
7. Compare PPO with meaningful baselines: buy-and-hold, always-HOLD, random actions, momentum rule, Random Forest, and a price-only policy.
8. Report economic metrics: cumulative net return, annualized return, volatility, Sharpe ratio, maximum drawdown, turnover, transaction costs, hit rate, and calibration.
9. Report class metrics only as secondary diagnostics: macro F1, balanced accuracy, per-class recall, and confusion matrix.
10. Perform ablations: price-only, news-only, Prophet-only, no transaction cost, different thresholds, and no pre-training.

The central evaluation question is not “Does it classify the next day perfectly?” It is “Does the policy produce better risk-adjusted net outcomes than simple, leakage-free baselines after realistic costs?”

## 12. Whole-project component explanations

### Prophet

Prophet models a time series as trend plus seasonality and noise, with automatically selected trend changepoints. AlphaGaze enables weekly and yearly seasonality and disables daily seasonality because the data is daily market data with weekends absent. It provides `yhat`, lower and upper intervals, and changepoints. Prophet is used as a feature generator and visualization model; PPO remains the action selector.

Why Prophet instead of only SMA/EMA: it can represent a forecast horizon, trend changes, and recurring seasonality. Why not claim it is always superior: financial prices are noisy, nonstationary, and often violate stable seasonal assumptions. It must be validated against a naive forecast and moving-average baselines.

### FinBERT

FinBERT is a BERT classifier adapted to financial language. The code computes `P(positive) − P(negative)` and leaves neutrality implicitly near zero. This is more context-aware than a generic word lexicon. It still measures language sentiment, not the actual market impact of a headline; a positive headline can already be priced in, and a neutral headline can coincide with a large move.

### News scraper

`scraper.py` queries Google News RSS, cleans titles, unwraps redirect URLs, removes duplicate titles, and sorts by date. RSS recency and article coverage are external dependencies. The feed is not a historical news archive, which is why online training has a short news-overlap window.

### API and frontend

FastAPI exposes `/`, `/api/stocks`, `/api/predict/{symbol}`, `/api/news/{symbol}`, and `/api/evaluate/{symbol}`. The frontend consumes the existing JSON shape, including `signal`, `probabilities`, `shap_values`, attention data, Prophet charts, and rolling-backtest fields. The field is still called `shap_values` for schema compatibility even though the current values come from perturbation analysis.

The live prediction response is conceptually:

```text
symbol, name, date, price, prophet_yhat, sentiment, signal
probabilities: BUY/HOLD/SELL floats from the PPO categorical policy
uncertainty: heuristic score, confidence, level, actionability, components
shap_values: legacy field name containing local perturbation scores
cv_weighted_f1: None in live prediction (old compatibility field)
news: at most 20 scored articles
attention_maps: up to 8 headlines with filtered token weights
attention_keywords: up to 12 aggregated tokens
counterfactual_tests: up to 3 headline token-removal tests
prophet_sensitivity: 10 temporal segments, top changepoints, forecast chart
rolling_backtest: up to 90 retained classification diagnostics
```

The frontend sorts probability keys alphabetically before drawing its chart, so the visual order is not the action-index order. The backend’s action-index mapping remains SELL=0, HOLD=1, BUY=2. The frontend escapes article URLs and titles before inserting them into the DOM, but it assumes numeric fields and nested objects are present; malformed partial JSON can still break rendering. Browser loading errors are shown in the error box, while the server returns 404 for unsupported symbols, 422 for expected data/overlap failures, and 500 for other pipeline exceptions.

### Request edge-case matrix

| Trigger | Current behavior | What a hardened system should add |
|---|---|---|
| Unknown symbol | HTTP 404 from prediction/news/evaluation routes | Keep an allowlist and return a structured error |
| Empty price response | `predict()` raises `ValueError`; route returns 422 | Include provider/status reason and retry guidance |
| Empty RSS feed | `predict()` raises `ValueError` after scraper returns `[]` | Distinguish no news from provider outage and optionally use a price-only fallback |
| Too few overlap rows | `predict()` raises `ValueError` below 15 rows | Report exact date ranges and missingness; do not silently lower the threshold |
| Pretrained ZIP absent | Train PPO from scratch for 50,000 timesteps | Expose initialization mode and expected latency |
| Pretrained ZIP incompatible | Outer API exception becomes HTTP 500 | Validate model version and fall back safely or fail with a specific model error |
| Explanation failure | Current outer exception can fail the whole response | Return the policy signal with partial XAI fields marked unavailable |
| Concurrent request | Duplicate expensive work is possible | Add lock/coalescing and shared cache |
| Provider data changes | Repeated runs can differ | Archive raw inputs and timestamp each result |

### Experiments and artifacts

The notebooks demonstrate earlier Prophet forecasts, FinBERT scoring, SHAP explanations, and a baseline-versus-AlphaGaze story. They are useful for explaining project evolution, but notebook numbers and simulated event-week examples should be labeled as experiments or demonstrations unless reproduced with a fixed, leakage-free test set.

## 13. Basic-to-advanced viva questions and answers

### Level 0: very basic questions

**Q: What problem does your project solve?**  
**A:** It combines price history and recent financial news to produce an interpretable daily SELL, HOLD, or BUY signal for supported Indian instruments.

**Q: What is machine learning?**  
**A:** It is a way of fitting parameters from data so a system can make predictions or decisions on new inputs.

**Q: What is reinforcement learning in one sentence?**  
**A:** An agent learns which actions are useful by trying actions in an environment and receiving rewards or penalties.

**Q: Who is the agent here?**  
**A:** The PPO policy and value networks.

**Q: What is the environment here?**  
**A:** `StockTradingEnv`, which replays chronological rows and calculates the reward for each action.

**Q: What is a state?**  
**A:** The information that describes the current decision situation. Internally this includes the current row and previous action; the network sees only the seven-feature observation.

**Q: What is an observation?**  
**A:** The portion of the state exposed to the agent. Here it is a seven-number float vector.

**Q: What is an action?**  
**A:** A decision. Here it is SELL, HOLD, or BUY.

**Q: What is a reward?**  
**A:** A numeric score assigned after an action. Here it is directional target-return reward, plus a HOLD penalty and transaction-cost penalty, scaled by 100. The target is normally next trading day on complete price data but can span a gap after news filtering.

**Q: What is an episode?**  
**A:** One pass from the first usable historical row to the last usable transition.

**Q: What is a policy?**  
**A:** A rule that maps an observation to action probabilities.

**Q: What is the difference between BUY and a prediction that price rises?**  
**A:** BUY is the agent’s chosen action. A price forecast is an intermediate model output. PPO uses all features and the reward to choose the action; it is not simply copying Prophet.

**Q: What is the difference between HOLD and doing nothing?**  
**A:** In this simulator HOLD receives `-0.1 × |next_return|`, so it is a deliberate action with a small missed-move penalty. It is not a fully modeled portfolio hold because no holdings are tracked.

**Q: What does a 0.62 BUY probability mean?**  
**A:** The learned categorical policy assigns 0.62 probability to BUY for that observation. It is not proof that BUY will be correct 62% of the time.

**Q: Why are there three actions?**  
**A:** The product interface is designed around three intuitive directional decisions, and `Discrete(3)` matches that contract.

**Q: Why is the action index order important?**  
**A:** The code maps 0→SELL, 1→HOLD, 2→BUY. If a metric or UI uses a different order, results can be silently mislabeled.

**Q: What is a feature?**  
**A:** A measurable input supplied to the model. The seven features summarize price trend, uncertainty, news mood, momentum, and volatility.

**Q: Why do you need seven numbers rather than raw news and raw prices?**  
**A:** Prophet and FinBERT compress complex inputs into consistent numerical signals that are practical for a small CPU policy. Raw sequences would require a larger sequence model and more data.

**Q: What is volatility?**  
**A:** A measure of how spread out recent returns are. The implementation uses the five-row standard deviation of daily percentage returns.

**Q: What is momentum?**  
**A:** The percentage change over the previous five trading rows.

**Q: What is sentiment?**  
**A:** The daily average of FinBERT’s positive probability minus negative probability for the available headlines.

**Q: How do you know which FinBERT logit is positive or negative?**  
**A:** The current code assumes the `yiyanghkust/finbert-tone` label order documented in its comments: neutral index 0, positive index 1, negative index 2. A robust implementation should read and validate the model’s `id2label` configuration instead of relying only on a comment, especially because the notebooks also use a different FinBERT model name and older index assumptions.

### Level 1: understanding the pipeline

**Q: Explain one prediction request from start to finish.**  
**A:** Fetch two years of adjusted prices, fit Prophet, scrape recent RSS headlines, score them with FinBERT, aggregate sentiment by date, inner-merge price/forecast/news rows, compute seven features and next-day returns, fine-tune or train PPO, extract the latest policy distribution, compute explanations, build the JSON response, and cache it for one hour.

**Q: Why do you use Prophet?**  
**A:** It provides a structured trend/seasonality forecast, an uncertainty band, and changepoints that become useful explanatory features. It is not claimed to solve the entire trading problem.

**Q: Why do you use FinBERT?**  
**A:** Financial words are context-dependent. FinBERT is trained for financial language and is more suitable than a generic sentiment lexicon for headlines.

**Q: Why is the merge an inner join?**  
**A:** The current implementation keeps dates for which both price/forecast data and daily news sentiment exist. The trade-off is that dates without news disappear instead of being explicitly neutral.

**Q: Why does the project use rolling sentiment averages?**  
**A:** To represent both immediate news response and persistence over three and five days.

**Q: Can the sentiment attached to a displayed article be misaligned?**  
**A:** The main predictor scores the filtered `news_df` but attaches those scores back by zipping them with the original `articles` list. Because the filtered rows are normally the newest rows this often aligns by accident, but a missing/out-of-range or reordered article can cause an incorrect attachment. The safe fix is to join scores by a stable article identifier or title/URL, not list position.

**Q: Why include a forecast uncertainty band as an input?**  
**A:** A forecast direction without uncertainty can cause overconfident actions. The band width gives the policy a measure of how uncertain Prophet is.

**Q: What happens when a feature is NaN?**  
**A:** Training rows with missing required features are dropped; the environment and inference helper convert any remaining NaN or infinite observation values to zero.

**Q: Why is the latest result cached?**  
**A:** The pipeline performs network calls, FinBERT inference, Prophet fitting, and PPO training. A one-hour cache avoids repeating expensive work for identical requests.

**Q: What happens if there is no pre-trained model?**  
**A:** The system trains PPO from scratch for 50,000 timesteps on the available merged rows. With a pre-trained model, it loads it and fine-tunes for 10,000 timesteps.

**Q: Is the pretrained model an ensemble?**  
**A:** No. It is one PPO model whose weights are updated sequentially across stocks.

**Q: Why set sentiment to zero during pre-training?**  
**A:** Historical RSS news is unavailable, so zero is used as a neutral placeholder rather than fabricated news. Online fine-tuning adds recent real sentiment.

### Level 2: RL mechanics

**Q: Why is this reinforcement learning and not just classification?**  
**A:** The policy is trained through an environment interaction loop and a reward function, not by directly minimizing a label classification loss. However, because the current simulator lacks full portfolio state, it is a simplified directional RL formulation and should not be sold as a complete portfolio MDP.

**Q: Does PPO train on the BUY/HOLD/SELL labels?**  
**A:** No. PPO’s reward uses `next_return` and the action. The threshold labels remain for diagnostics and backtest comparison.

**Q: Why is the label threshold ±0.3% still present?**  
**A:** It is legacy diagnostic labeling. It makes classification-style reports possible, but it is not the objective used by PPO. A cleaner future evaluation would emphasize returns and remove label dependence.

**Q: What is the transition probability in this environment?**  
**A:** In historical replay, the next row is deterministic given the current index. The market itself is stochastic, but the offline simulator does not sample a learned transition model.

**Q: Is the environment Markov?**  
**A:** The internal environment can be described with enough variables to make it Markov, but the seven-value observation omits `previous_action`, which affects cost. Therefore the policy sees a partial observation of the internal state.

**Q: What does `prev_action` do?**  
**A:** It is initialized to HOLD, updated after every step, and used to decide whether a non-HOLD action change pays the transaction cost. It is not included in the observation.

**Q: Is a SELL allowed when the agent owns nothing?**  
**A:** Yes, because the simulator has no inventory constraint. SELL should therefore be described as a bearish directional exposure, not necessarily a literal short sale.

**Q: Why penalize HOLD?**  
**A:** A small penalty prevents a trivial always-HOLD policy from looking attractive when the environment offers no other explicit objective. It represents a missed directional opportunity, but its coefficient is a design choice and needs sensitivity testing.

**Q: Does the reward penalize risk, drawdown, or volatility?**  
**A:** Not directly. Volatility is an observation, but the reward has no variance penalty, drawdown term, leverage limit, or portfolio-risk budget. Two policies with the same cumulative directional reward can have very different risk, which is why Sharpe, drawdown, turnover, and tail-loss evaluation are essential.

**Q: Why include a transaction cost?**  
**A:** Without a cost, the agent can change actions too freely and produce unrealistic churn. The cost encourages fewer unnecessary directional switches.

**Q: Why exactly 0.1%?**  
**A:** It is a simple prototype assumption, not a measured all-in Indian-market execution cost. It is lower than the ±0.3% diagnostic label threshold, so a small predicted move can still be economically unattractive after costs. The value should be replaced or stress-tested using brokerage, spread, taxes, slippage, and execution timing appropriate to the instrument.

**Q: Why is the cost only applied when the new action is not HOLD?**  
**A:** That is how the current code is written. It means entering or changing a directional action is charged, while moving to HOLD is free. A realistic broker simulator would model entry, exit, and spread costs explicitly.

**Q: What does reward scaling change?**  
**A:** It multiplies rewards by 100 so gradients and value targets are numerically easier to optimize. It does not change the sign or ordering of a one-step reward, but it can affect optimizer dynamics.

**Q: Why is PPO on-policy?**  
**A:** Each update uses data collected by the current or immediately previous policy. This makes the objective controlled and stable but less data-efficient than replay-buffer methods such as DQN.

**Q: What is the critic’s job?**  
**A:** It estimates expected discounted future reward and provides a baseline for computing lower-variance advantages.

**Q: What is an advantage?**  
**A:** `A(s,a)` estimates how much better or worse action `a` was than the policy’s average expected action at state `s`.

**Q: What does PPO clipping actually clip?**  
**A:** It clips the action-probability ratio inside the surrogate objective, not every network weight and not every final probability change.

**Q: Why use entropy?**  
**A:** The entropy bonus rewards a less-certain distribution early in learning, helping prevent premature collapse to one action. It does not guarantee balanced actions.

**Q: Why use an MLP?**  
**A:** The policy receives a fixed seven-value tabular vector, so a small feed-forward network is simpler and cheaper than a sequence model. Temporal context is summarized by momentum, volatility, sentiment averages, and the environment’s chronological order.

**Q: Why is a recurrent network not used?**  
**A:** The current observation engineering already includes short windows, and CPU/API latency and limited data favor a small MLP. Recurrent PPO is a valid future experiment if longer memory improves walk-forward results.

### Level 3: alternatives and model selection

**Q: Why not Random Forest?**  
**A:** Random Forest is a strong tabular baseline, but it optimizes label agreement. PPO directly optimizes the defined action reward and supplies a native action distribution. We should still benchmark Random Forest rather than assume PPO wins.

**Q: Why not DQN if there are exactly three discrete actions?**  
**A:** DQN is a legitimate candidate. PPO was chosen because the dashboard needs policy probabilities and PPO’s categorical actor maps directly to that requirement. DQN would produce Q-values; applying softmax would add a temperature-dependent interpretation that is not a native probability policy.

**Q: Is DQN unable to output probabilities?**  
**A:** It has no native calibrated policy distribution, but one can derive a softmax-like preference distribution from Q-values. Therefore “DQN cannot output probabilities” is too absolute; “DQN does not provide the same principled stochastic policy output” is accurate.

**Q: Why not A2C?**  
**A:** A2C has actor and critic and supports discrete actions, but PPO adds a conservative clipped update. With noisy, repeatedly replayed data, that stability was the practical reason to prefer PPO.

**Q: Why not SAC?**  
**A:** Standard SAC in Stable-Baselines3 is for continuous action spaces. A discrete SAC implementation exists conceptually, but it would add a different dependency or custom implementation for this three-action interface.

**Q: Why not DDPG or TD3?**  
**A:** They are designed for continuous actions such as a position fraction. The current product contract is a discrete three-action signal.

**Q: Why not a Transformer?**  
**A:** The dataset and observation are small, the API targets CPU execution, and a large sequence model would increase complexity and overfitting risk without evidence of benefit.

**Q: Why Prophet instead of an LSTM, GRU, or time-series Transformer?**  
**A:** Prophet is fast, transparent, supports trend changepoints and seasonality, and fits the small CPU-oriented prototype. Deep sequence models are valid alternatives, but they require careful sequence construction, more data, normalization, tuning, and leakage-safe validation. “Prophet is always better” would be incorrect; the choice is a complexity/data/interpretability trade-off.

**Q: Why FinBERT instead of a word-count lexicon or generic BERT?**  
**A:** Financial language changes word meaning and context, so a finance-domain model is a better prior for headlines. A lexicon is cheaper and easier to audit, while a generic BERT may miss financial usage. FinBERT is still imperfect: it scores headline language, not causal price impact.

**Q: Why use a headline rather than the full article?**  
**A:** The RSS feed reliably supplies titles and URLs, while full articles may be paywalled, unavailable, or legally restricted. Headlines provide low-latency text, but they lose context and can be sensational or ambiguous.

**Q: Why concatenate features instead of using separate policy branches?**  
**A:** Concatenation makes a seven-dimensional interface simple and lets one MLP learn cross-modal interactions. Separate branches could preserve modality structure, but would add parameters and require more data and ablations.

**Q: Why daily data rather than intraday data?**  
**A:** Daily data reduces microstructure noise, network load, and execution complexity and matches the available historical price/news alignment. It cannot support claims about intraday response or precise news-time trading.

**Q: Why use adjusted close and not OHLCV?**  
**A:** Adjusted close makes long-horizon returns more comparable across corporate actions and keeps the prototype small. Omitting open/high/low/volume loses liquidity, gap, range, and volume information; adding them is a natural future feature study.

**Q: Why not use a paid historical news API?**  
**A:** The prototype uses an accessible RSS source without credentials, but that sacrifices historical coverage, timestamp precision, and stability. A serious backtest needs an archived, licensed news source or an explicit price-only pre-training strategy.

**Q: Is PPO guaranteed to outperform Random Forest?**  
**A:** No. Algorithm selection is a fit-to-objective argument, not a guarantee. The correct comparison requires leakage-free walk-forward financial metrics and baselines.

### Level 4: adversarial technical questions

**Q: Your “state” omits previous action, yet previous action changes reward. Is your observation Markov?**  
**A:** No, not fully. The internal environment state tracks previous action, but the policy receives only the seven features. This is a partial-observability limitation. I would fix it by appending previous action or current position to the observation, and by adding actual cash/holdings if the goal is portfolio simulation.

**Q: You call it trading profit. Where are holdings and cash?**  
**A:** They are not modeled. The current reward is directional return reward with a switching cost. I would call it a signal simulator, not a complete account-level profit calculation, until portfolio accounting is added.

**Q: You say RL removes arbitrary labels, but your code still creates ±0.3% labels.**  
**A:** Correct: the labels remain for diagnostics and rolling classification metrics. PPO training uses the return-based reward, so the learning objective does not depend on the threshold. I should not claim that every evaluation path is label-free.

**Q: You evaluate on the same data used for training. Is that a valid F1?**  
**A:** It is an in-sample diagnostic, not a valid generalization estimate. A proper report needs chronological walk-forward holdouts.

**Q: Does fitting Prophet on all dates leak future information?**  
**A:** Yes, for historical feature rows it can. Prophet must be fitted only through each cutoff date for a leakage-free backtest. This is a known methodological improvement.

**Q: Why are your policy probabilities called confidence?**  
**A:** They are policy probabilities, and the UI’s uncertainty summary is a heuristic using entropy, margin, forecast band, and sentiment. Calibration is still required before interpreting them as empirical confidence.

**Q: Your perturbation explanation is called SHAP. Is it SHAP?**  
**A:** No. The response key retains the name `shap_values` for frontend compatibility, but current values are local Gaussian perturbation sensitivities. They are not Shapley values and should be labeled accordingly.

**Q: Can attention weights prove that a word caused the sentiment?**  
**A:** No. They show where attention was allocated. The counterfactual removal test is a more direct diagnostic, but neither proves causality.

**Q: Why train 50,000 timesteps on a few hundred rows?**  
**A:** PPO timesteps count environment interactions, so the same historical episode is replayed many times. This makes optimization possible but creates overfitting risk. The fix is more independent historical episodes, stronger holdouts, regularization, and early stopping based on validation performance.

**Q: Does pre-training across stocks solve the data problem?**  
**A:** It increases price-only exposure and transfers weights, but it does not create historical sentiment. It also introduces cross-asset distribution shift and order dependence. It is a practical initialization strategy, not proof of universal market knowledge.

**Q: Why not fill missing news dates with neutral zero instead of dropping them?**  
**A:** The current code uses an inner merge, so it drops them. Filling them would preserve price continuity but would make “no article found” indistinguishable from measured neutral sentiment unless an explicit news-availability feature were added.

**Q: Is the next-day return available when making today’s decision?**  
**A:** No. It is only used by the historical environment after the action to calculate the training reward. The live observation contains features known at decision time; the final row is dropped because it has no future return.

**Q: How would you prove the agent learned rather than memorized?**  
**A:** Use chronological holdouts, multiple seeds, ablation tests, baseline comparisons, calibration, and net-return metrics after costs. A high in-sample F1 is not enough.

**Q: What is the strongest current limitation?**  
**A:** The environment is not a full portfolio simulator and the current evaluation is not yet a clean, leakage-free out-of-sample financial backtest. The architecture is a useful research prototype, not an investment guarantee.

### Level 5: implementation edge cases and production questions

**Q: Is `next_return` always literally the next trading day in the online pipeline?**  
**A:** Not necessarily. The online data is first inner-joined with dates that have news. Only then does the code apply `shift(-1)`. If news is present on Monday and Wednesday but not Tuesday, the “next” return is Wednesday’s close divided by Monday’s close, not a one-day return. A robust fix is to calculate returns on the complete price table before joining sentiment, then attach the features to the decision date.

**Q: Do the three-day and five-day sentiment averages always represent three and five trading days?**  
**A:** In the current merged table they are rolling over rows that survived the news inner join. Missing-news dates are absent, so a three-row average can cover more than three calendar or trading days. The production fix is to reindex sentiment to the full trading calendar, add an explicit missing-news indicator, and choose a documented missing-value policy.

**Q: Does “daily news sentiment” mean sentiment known before the market close?**  
**A:** No. RSS provides a publication date but the current code does not use an intraday publication timestamp or market-session cutoff. A headline published after the close can be attached to that date while the same day’s close is used as if the information were available. A strict experiment needs a timestamp-aware information set and a clearly defined execution time.

**Q: Is `prophet_gap` a forecast error?**  
**A:** In production code it is `(yhat - y) / y`, where `yhat` is Prophet’s fitted value at the same historical date. It is an in-sample fitted residual-like feature, not an out-of-sample forecast error. A causal forecasting feature would be generated from a model fit only through the preceding cutoff.

**Q: Are Prophet’s lower and upper values guaranteed confidence limits?**  
**A:** They are Prophet’s model-based uncertainty interval under its assumptions. They are not a guarantee that the market price will remain inside the band, and the in-sample bands used here should not be interpreted as calibrated coverage probabilities.

**Q: Why can the current chart show weekends if the market is closed?**  
**A:** `make_future_dataframe(periods=30)` uses Prophet’s default daily frequency, so the forecast chart can include calendar days. Price decisions and return calculations use available trading rows. A cleaner chart would request business/trading dates or clearly label non-trading forecast dates.

**Q: What happens if yfinance returns a MultiIndex?**  
**A:** `rl_data.py` explicitly flattens MultiIndex columns, but the main `predictor.py` path does not. Depending on the yfinance response version, selecting `Date` and `Close` can therefore produce a shape/type problem. A production fix is to normalize yfinance columns in one shared helper and test both single- and multi-index responses.

**Q: What happens if timestamps have different timezones?**  
**A:** `rl_data.py` and `evaluator.py` remove timezone information, while the main predictor relies on the downloaded and RSS dates lining up after conversion. A mismatch can silently reduce the inner merge to zero rows. All sources should be converted to one explicit exchange timezone and normalized to a date key after applying the session cutoff.

**Q: What if there is no news, an empty feed, or a malformed RSS item?**  
**A:** `scrape_news()` returns an empty list on a feed-parser exception; `predict()` then raises a `ValueError` that becomes HTTP 422. Missing titles are skipped, dates fall back to the current UTC date, and duplicate titles are removed. There is no request timeout, retry policy, stale-feed indicator, or explicit rate-limit handling.

**Q: What if a price value or return is infinite rather than NaN?**  
**A:** The observation helper converts NaN and infinities to zero, but `_compute_reward()` explicitly checks only `np.isnan(next_return)`. An infinite target could therefore contaminate the reward. Input validation should reject non-finite prices/returns before constructing the environment.

**Q: Does Google News RSS really have no rate limits?**  
**A:** That is an overly strong README claim. No API key is required, but the service can throttle, change result coverage, or fail. The pipeline should treat RSS as an external, non-deterministic dependency and log source freshness and article counts.

**Q: Could the news query return market-wide articles rather than stock-specific information?**  
**A:** Yes. Queries such as “NIFTY 50 India stock market” are intentionally broad, and an article can mention a company without describing a causal price event. Entity filtering, source diversity checks, and a relevance score would reduce this noise.

**Q: Does deduplicating by title remove valid evidence?**  
**A:** It can. Two outlets can publish the same headline with different context or timing, but the current set-based deduplication keeps only one copy. Deduplication should ideally use canonical URL, publisher, and a time window, then decide whether repeated coverage is signal or duplication.

**Q: What happens if the same symbol is requested twice at the same time?**  
**A:** The cache is an in-memory dictionary without a per-symbol lock. Two requests can both execute Prophet, FinBERT, and PPO training before either stores a result. A production service should use request coalescing or a lock and a shared cache.

**Q: What happens if the offline script is launched from a different working directory?**  
**A:** `rl_data.py` and the main predictor construct several paths relative to their files, but `time_series.py` and `combine_and_xai.py` use relative `results/...` paths. Their behavior therefore depends on the current working directory. A robust CLI should resolve all input/output paths from the repository root or explicit arguments.

**Q: Is the one-hour cache shared across processes?**  
**A:** No. `_cache` lives inside one Python process. Multiple workers, restarts, or reloads have separate caches, so they can repeat work and return results generated at different times.

**Q: Can online fine-tuning make the cached result stale?**  
**A:** Yes. The result can be up to one hour old, while prices and news change during that period. The cache policy should be stated as a freshness trade-off rather than silently treated as real-time.

**Q: Is the API ready for public deployment?**  
**A:** It is a local demo API. CORS allows every origin, there is no authentication, rate limiting, request tracing, or user-level resource quota, and prediction requests trigger expensive computation. Public deployment needs those controls plus validation of external URLs and network failures.

**Q: Is a symbol in the URL automatically safe?**  
**A:** The prediction and news handlers check membership in `SUPPORTED_STOCKS`, which prevents arbitrary symbols through those endpoints. The scraper still constructs external URLs from query text, and the returned article links are rendered as external anchors. URL validation, content-security policy, and safe outbound request policies belong in a hardened deployment.

**Q: Why does the UI still call the feature-impact field `shap_values`?**  
**A:** The field name was retained so the existing frontend schema would continue to work. The current values are perturbation scores, not SHAP values. The field should be renamed or accompanied by an explicit method field in a future API version.

**Q: What happens if an explanation fails after the policy has been trained?**  
**A:** In the current `predict()` path, an exception in attention, counterfactual, Prophet sensitivity, or rolling backtest can fail the whole API request because the route catches it only at the outer level. A more resilient API would return the signal with per-component diagnostics marked unavailable.

**Q: What if attention has no tokens or contains unknown tokens?**  
**A:** The code filters special/punctuation tokens, computes an unknown-token ratio, and the frontend displays a “no attention map” message if necessary. Attention extraction uses max length 128, while sentiment scoring uses max length 512, so the displayed map may not cover all text used for the sentiment score.

**Q: What if two actions have exactly equal policy probability?**  
**A:** `np.argmax` selects the first index, which means a tie resolves to SELL because SELL is index 0. This is a deterministic implementation detail and should be documented or replaced with an explicit tie policy.

**Q: Do rounded probabilities always sum exactly to one?**  
**A:** The underlying categorical probabilities sum to one, but each value is rounded to four decimal places before it is placed in the response. The displayed values can therefore sum to 0.9999 or 1.0001. The UI should treat them as rounded display values.

**Q: Is the perturbation importance stable?**  
**A:** It is deterministic for a given call because the function resets NumPy’s seed to 42, but it is not necessarily statistically stable. The noise scale is `max(abs(feature) × 0.5, 0.01)`, not an empirical feature standard deviation; perturbations are not clipped to realistic ranges; and correlated features are perturbed independently. Repeating seeds and using distribution-aware counterfactuals would be stronger.

**Q: Can a negative perturbation score mean a feature is “bad”?**  
**A:** No. It means the chosen perturbations tended to increase the baseline action’s probability. The score is local and action-specific; it is not a global good/bad feature judgment.

**Q: Does feature importance explain why the model chose BUY over both alternatives?**  
**A:** It measures the baseline action’s probability, not a complete pairwise comparison against BUY, HOLD, and SELL. A better explanation would report per-action probability deltas or a local surrogate for all three actions.

**Q: Is the random seed enough for reproducibility?**  
**A:** No. PPO receives `seed=42`, but reproducibility also depends on NumPy/PyTorch state, library versions, deterministic kernels, remote data snapshots, Prophet’s fitting behavior, RSS results, and model-file compatibility. Reproducibility requires version locking, cached inputs, saved configuration, several seeds, and deterministic settings where supported.

**Q: What is the difference between `n_steps` and `total_timesteps`?**  
**A:** `n_steps=64` is the number of transitions collected before an update from one environment. `total_timesteps=10,000` or `50,000` is the requested total number of environment interactions. Stable-Baselines3 collects complete rollouts, so the actual count can be rounded to rollout boundaries.

**Q: Why can the same historical row be seen many times?**  
**A:** The environment always resets to the beginning and replays a deterministic dataset. PPO therefore obtains many optimization samples from the same finite trajectory. That is computational reuse, not new independent market evidence, and it raises overfitting risk.

**Q: Is there more than one environment running?**  
**A:** No. `DummyVecEnv([_init])` is a vectorized wrapper around one environment. It satisfies the library interface but does not provide parallel trajectories or independent market scenarios.

**Q: What happens if `step()` is called after termination?**  
**A:** The normal Gymnasium contract says the caller must reset first. The custom class does not add a protective guard, so a second call can index beyond the dataframe. A production environment should validate the episode state and reject invalid calls clearly.

**Q: What happens for an empty or one-row dataframe?**  
**A:** `reset()` expects a valid current row, and there is no explicit constructor validation for a minimum number of transitions. The application-level pipeline checks for at least 15 merged rows, but direct environment use should validate length and required columns.

**Q: Why is `truncated` always false?**  
**A:** The environment has only a data-exhaustion termination condition and no independent time limit. If a future version adds a maximum episode length, that should be reported as truncation rather than natural termination.

**Q: Is `total_reward` part of what the policy learns?**  
**A:** No. It is stored for the `info` dictionary and logging. It is not in the observation and does not directly affect the next reward.

**Q: Does the current policy carry a BUY position across days?**  
**A:** No explicit position is carried. The previous action only affects a possible switching cost. The reward is recalculated as a fresh directional bet on each row. This is why the current system should be described as action/signal RL rather than inventory-aware portfolio RL.

**Q: Does `gamma=0.99` create genuine long-horizon portfolio planning here?**  
**A:** Only partially. The episode spans many rows, so discounted future rewards are in the objective, but there is no position or portfolio value that carries economic exposure across days. The main reward remains one-step directional return plus a local switching cost.

**Q: Is the environment stochastic?**  
**A:** The historical transition is deterministic, but the PPO policy samples actions during training and the perturbation/explanation procedures use noise. The market itself is not simulated stochastically.

**Q: Why are observations not normalized?**  
**A:** The current code declares an unbounded Box and passes raw engineered values to the MLP; it does not use `VecNormalize` or a train-only scaler. The features are mostly normalized ratios, but different distributions can still affect optimization. Adding train-only normalization and preserving its statistics at inference is a clear improvement.

**Q: Does pre-training across ten stocks avoid survivorship bias?**  
**A:** No. The list contains currently supported, mostly large Indian instruments and no delisted securities. The resulting sample may not represent the full investable universe and can be affected by survivorship and selection bias.

**Q: Is NIFTY 50 itself a tradable stock?**  
**A:** No. It is an index symbol used as a market benchmark. A BUY/SELL signal for it is a directional index signal, not a direct equity order.

**Q: Are corporate actions handled?**  
**A:** `auto_adjust=True` requests adjusted price series, which helps account for splits and dividends in historical prices. The system does not separately model dividends, splits, taxes, liquidity, or execution prices, so adjusted close should not be confused with an executable total-return account.

**Q: What is missing from the observation?**  
**A:** There is no volume, order book, spread, market index regime, macroeconomic variable, earnings calendar, cash, position size, drawdown, or portfolio exposure. This limits what the agent can learn and what the signal means.

**Q: Does the current evaluation measure money made?**  
**A:** No. The main reported diagnostics are classification accuracy/F1, confidence buckets, Prophet MAE/RMSE, and a sentiment proxy agreement. A net portfolio return evaluation is a required next step.

**Q: Is 33% always the correct random baseline for three classes?**  
**A:** Only for balanced classes and a uniform random classifier. If HOLD is much more common, a majority-class baseline can exceed 33%, and a stratified random baseline is more informative. The README’s “48% versus 33%” statement is therefore not sufficient evidence of skill.

**Q: Why can accuracy be misleading here?**  
**A:** A policy that predicts the dominant HOLD class can achieve high accuracy while failing to identify profitable BUY/SELL opportunities. Macro F1, per-class recall, balanced accuracy, action frequency, and net-return metrics are needed.

**Q: Why is a fixed ±0.3% threshold questionable?**  
**A:** It is simple and interpretable, but it is arbitrary and not volatility-adjusted. A 0.3% move means different things in quiet and turbulent regimes, and the threshold does not include spread, taxes, or slippage. Threshold sensitivity and volatility-scaled labels should be tested.

**Q: How do you know PPO’s improvement is statistically significant?**  
**A:** The current repository does not establish that. A defensible study needs repeated chronological windows, multiple seeds, paired comparisons against baselines, confidence intervals or bootstrap intervals, and a correction for repeated model/feature choices.

**Q: What is the null hypothesis for a serious experiment?**  
**A:** A simple null is that PPO’s net risk-adjusted return is no better than buy-and-hold, always-HOLD, random, or a price-only baseline after costs. The exact null and test window must be fixed before looking at results.

**Q: What is an ablation study in this project?**  
**A:** Remove one information branch or feature group while keeping the evaluation protocol fixed: price-only, news-only, Prophet-only, no sentiment averages, no forecast band, no pre-training, and no transaction cost. The performance change estimates whether that component adds evidence.

**Q: Why is the market’s non-stationarity important?**  
**A:** Relationships that worked in one regime can fail after a policy change, crisis, interest-rate shift, or technology change. Randomly mixing all dates can hide regime failure, so evaluation should preserve time and report performance by regime.

**Q: What is survivorship bias in this project?**  
**A:** The pre-training list contains current well-known instruments and excludes companies that disappeared, were delisted, or were not selected. Studying only survivors can make historical performance look better than a real historical universe.

**Q: What is look-ahead bias besides Prophet leakage?**  
**A:** Using a headline before its publication cutoff, using revised/adjusted data without recreating the information available then, calculating a return after a filtered merge, fitting scaling on future rows, or tuning thresholds after seeing the test set are all forms of future information entering the decision.

**Q: Why is Prophet MAE in the evaluator not enough?**  
**A:** The evaluator predicts Prophet on dates it was fitted on, so the MAE/RMSE are in-sample fit errors. They do not measure 30-day out-of-sample forecast quality.

**Q: Are the evaluator’s features identical to production?**  
**A:** No. The evaluator reverses the production `prophet_gap` sign/denominator, uses a mock fixed band, random sentiment, and computes volatility from rolling price levels rather than rolling daily returns. Its output is therefore a diagnostic scaffold, not a faithful production evaluation.

**Q: Is the saved pretrained model guaranteed to load forever?**  
**A:** No. Stable-Baselines3, PyTorch, Gymnasium, Python, and policy serialization versions can affect compatibility. The model must be versioned with its training configuration and tested after dependency changes.

**Q: What exactly is transferred when fine-tuning a pretrained model?**  
**A:** `train_agent()` reuses the PPO object, calls `set_env()`, and continues `learn()`. This preserves policy/value weights and the optimizer state rather than resetting a fresh optimizer. That can be useful for continuity but can also make adaptation depend on the previous training history; an experiment should compare warm-start and fresh-start training.

**Q: What if one of the ten pre-training stocks fails?**  
**A:** `rl_data.py` catches the exception, skips that stock, and continues. If many fail, the saved model may have seen far fewer instruments than the log’s intended design. The pre-training manifest should record which symbols succeeded, their row counts, and their data ranges.

**Q: Does pre-training resume after a crash?**  
**A:** No checkpoint is saved after each stock; the final model is saved only after the loop. A crash can lose all progress. Per-stock checkpoints and a manifest would make the process recoverable.

**Q: Does the environment validate its dataframe schema?**  
**A:** No explicit constructor validation checks required columns, sorted dates, finite values, or a minimum length. Missing columns fail later with a less helpful exception. A production environment should validate schema and chronology at construction.

**Q: What happens if `max_articles` is negative or extremely large?**  
**A:** The endpoint passes it into the scraper without a strong range validator. Query limits, network timeouts, and a safe upper bound should be enforced at the API boundary.

**Q: Is there a test suite?**  
**A:** There is no dedicated tests directory in the repository. Syntax compilation and frontend JavaScript syntax checks pass, but environment invariants, reward examples, mapping consistency, API error cases, and walk-forward leakage need automated tests.

**Q: Are the cached price files sufficient for exact reproduction?**  
**A:** They help reproduce the price inputs, but current news RSS, yfinance revisions, model downloads, timestamps, and library versions can still differ. The exact input snapshot and model version must be archived for a repeatable experiment.

**Q: Why should the old notebooks not be treated as the final RL evidence?**  
**A:** They document the project’s Prophet/FinBERT/SHAP development and contain simulated or earlier baseline experiments. They predate or do not fully represent the current PPO code path, so their numbers need explicit labels and independent reproduction.

**Q: What documentation is stale?**  
**A:** `README.md`, `agents.MD`, parts of `IMPLEMENTATION_PLAN.md`, and some module/API docstrings still mention Random Forest, SHAP, one-year data, or old dependency versions. The current implementation uses PPO and perturbation impact. A final release should update all documentation from one source of truth.

**Q: Why do some current Python files still import SHAP, Random Forest, or `TimeSeriesSplit`?**  
**A:** They are migration leftovers and are not the active decision path in the current PPO functions. Unused imports increase confusion and dependency surface; they should be removed after confirming that the old offline artifacts are no longer needed.

**Q: How many API endpoints are actually present?**  
**A:** There are five route groups: `/`, `/api/stocks`, `/api/predict/{symbol}`, `/api/news/{symbol}`, and `/api/evaluate/{symbol}`. The “four endpoints” wording in the old agent context excludes the root page.

**Q: Does the API evaluation endpoint use the same data as prediction?**  
**A:** No. Prediction uses about two years plus current RSS news; evaluation uses one year of `Ticker.history`, synthetic sentiment, and a mock band. Results from the two endpoints are not directly comparable.

**Q: Can a network failure be distinguished from a bad model prediction?**  
**A:** Not cleanly in the current API. Many exceptions become a generic HTTP 500, while a `ValueError` becomes 422. A production response should distinguish data unavailable, model unavailable, insufficient overlap, and computation failure.

**Q: Are article URLs trusted?**  
**A:** They come from RSS and are placed in external links with `target="_blank"` and `rel="noopener"`. The UI escapes URL and title text, but a production system should still validate schemes, handle link availability, and avoid assuming external content is safe or permanent.

**Q: Is there any financial advice or execution layer?**  
**A:** No. The system displays research signals. It has no broker integration, order execution, suitability assessment, tax calculation, or guarantee of returns. The correct user-facing disclaimer is that the signal is informational and not financial advice.

**Q: What ethical or legal questions should be asked about the data?**  
**A:** Verify the permitted use and attribution for yfinance data, Google RSS content, the FinBERT model, and the IN-FINews dataset. Avoid redistributing article content beyond allowed metadata, protect any user data, and document that a demo signal is not a promise of performance.

## 14. A short oral defense sequence

If given five minutes, follow this order:

1. “The product combines price and news into an action policy.”
2. Define agent, environment, observation, action, reward in one sentence each.
3. Show the seven features and explain one example of each.
4. Show the reward equation and the worked `+0.8%` example.
5. Explain actor, critic, categorical policy, advantage, and PPO clipping.
6. Explain why PPO fits the discrete action/probability interface.
7. State the no-inventory limitation before the examiner discovers it.
8. Explain pre-training versus online fine-tuning.
9. Explain that current F1 is diagnostic/in-sample and propose walk-forward net-return evaluation.
10. End with the practical next fix: add position/cash to state and perform leakage-free evaluation.

## 15. Improvements that make the defense stronger

These are concrete next steps, ordered by methodological importance:

1. Add `position`, cash or portfolio value, and previous action to the observation.
2. Define explicit position semantics: flat/long/short or a continuous position fraction.
3. Charge realistic entry/exit costs, spread, slippage, and turnover.
4. Generate all Prophet features walk-forward, using only information available at each timestamp.
5. Replace in-sample evaluator scores with chronological train/validation/test splits.
6. Compare action strings directly rather than mixing `LabelEncoder` indices with `SIGNAL_MAP` indices.
7. Report net-return, Sharpe, drawdown, turnover, and calibration metrics.
8. Add always-HOLD, buy-and-hold, random, momentum, and Random Forest baselines.
9. Use multiple random seeds and confidence intervals over evaluation windows.
10. Normalize features using training-only statistics and test whether the policy is sensitive to scale.
11. Replace zero sentiment pre-training with an explicit missing-news indicator or an archived news source.
12. Rename `shap_values` to `feature_impact` in a future API version while preserving backward compatibility during migration.
13. Update stale docstrings in `predictor.py`, `api.py`, `combine_and_xai.py`, and `README.md` that still say Random Forest/SHAP.
14. Avoid silently swallowing every rolling-backtest exception; log the failure reason and count skipped windows.

## 16. Second repository audit: facts that should not be blurred together

This table is a compact “do not get trapped” checklist. It separates production behavior, offline demonstrations, and documentation that has not yet been synchronized.

| Topic | Current fact | Safe defense wording |
|---|---|---|
| Production learner | PPO with `MlpPolicy`, 7 features, one `DummyVecEnv` environment | “The deployed decision layer is PPO.” |
| Production training window | About 730 calendar days of prices, but only dates surviving the current news merge reach the online environment | “The request fetches two years of prices; effective RL rows depend on news overlap.” |
| Production update | Load pre-trained PPO if the file exists; otherwise train from scratch; fine-tuning is done in memory and is not saved by `predict()` | “Pre-training is persistent; request-specific adaptation is temporary and cached only as a result.” |
| Offline combined run | Trains on every merged row, then evaluates those same rows | “Its F1 is an in-sample diagnostic.” |
| Offline evaluator | Uses synthetic sentiment, fixed `forecast_band`, and feature formulas that do not exactly match production | “It is a scaffold and sanity check, not the final benchmark.” |
| Rolling diagnostic | Re-fits/fine-tunes repeatedly, catches all exceptions silently, reports classification hit rate and policy probability | “It is an application diagnostic, not a complete net-return backtest.” |
| Labels | ±0.3% labels remain for diagnostics | “Labels are not PPO’s reward objective, but they still influence reported classification metrics.” |
| Action semantics | Directional reward, no holdings/cash/position size | “SELL/HOLD/BUY are signals or directional exposures in the current simulator.” |
| Return horizon | Full price rows have next-row returns; online news-filtered rows can create multi-day gaps | “The horizon must be fixed explicitly before claiming next-day performance.” |
| Feature chronology | Prophet is fitted on the full window before historical features are generated | “Strict walk-forward feature generation is still required to remove look-ahead.” |
| Probability meaning | Actor categorical probabilities, rounded to four decimals in the API | “They are policy preferences, not calibrated correctness probabilities.” |
| Feature explanation | Gaussian local perturbation score under the legacy `shap_values` key | “It is perturbation sensitivity, not SHAP or causality.” |
| Sentiment explanation | Last-layer CLS attention visualization plus token-removal tests | “Attention is a diagnostic; counterfactual rescoring is a stronger but imperfect check.” |
| Price data | `auto_adjust=True` close series from a mutable remote provider | “Prices are adjusted historical inputs, not an immutable exchange archive.” |
| News data | Google RSS headline dates, no reliable historical archive or intraday cutoff | “News coverage and information timing are limitations.” |
| Reproducibility | Remote data/model downloads, library versions, and random seeds all matter | “A reproducible experiment must archive inputs, versions, configuration, and seeds.” |
| Dependencies | The checked-in requirements currently use `stable-baselines3==2.9.0`, `gymnasium==1.0.0`, and `shimmy==2.0.1`; the plan mentions older versions | “The installed requirements, not the migration-plan example, define the current environment.” |
| Documentation | README, `agents.MD`, module docstrings, and plan contain stale RF/SHAP language | “Documentation synchronization is a release task.” |
| Artifacts | CSVs, plots, cached prices, and the pretrained ZIP are snapshots; they are not proof of current performance | “Artifact timestamps and provenance must accompany any reported number.” |
| Deployment | No authentication, rate limiting, shared cache, request lock, or broker execution | “This is a local research dashboard, not a production trading service.” |

### Exact evaluator mismatches

The evaluator is particularly easy to misdescribe. In production, `prophet_gap` is `(yhat - y) / y`; the evaluator uses `(y - yhat) / yhat`. Production volatility is the standard deviation of five daily percentage returns; the evaluator uses the rolling standard deviation of price levels. Production forecast-band width comes from Prophet’s interval; the evaluator hard-codes `0.05`. Production sentiment comes from RSS and FinBERT; the evaluator samples random sentiment with seed 42. These differences mean an evaluator score cannot be presented as a direct measurement of the production request path.

### Exact documentation/artifact mismatches

The repository includes an ignored `agents.MD` that still describes a Random Forest architecture in parts of its module reference, while its diagram mentions PPO. `README.md` still explains SHAP and reports an old approximate “48% confidence” claim. `IMPLEMENTATION_PLAN.md` contains the intended migration and older dependency examples, not a guarantee that every plan statement matches the checked-in code. Existing `results/` and `app/results/` files are generated snapshots, and the root `data/` artifacts reflect the date they were created rather than live performance.

The stored snapshots themselves should be treated carefully. The checked-in sentiment CSVs contain 173 daily rows spanning approximately February–August 2025, while the stored Prophet forecast extends into September 2025. The two `latest_prediction.csv` copies contain different policy probabilities and F1 values for the same displayed date, which is evidence that they came from different runs. They are useful examples of output format, not a single authoritative benchmark.

## 17. Code map for quick revision

| File | Lines/concepts to know |
|---|---|
| [`app/rl_env.py`](app/rl_env.py) | Action mapping, seven-feature observation, reward equation, reset/step, termination |
| [`app/rl_agent.py`](app/rl_agent.py) | PPO hyperparameters, environment wrapper, transfer learning, policy probabilities, perturbation impact, save/load |
| [`app/rl_data.py`](app/rl_data.py) | Max-history downloads, cached CSVs, Prophet price features, zero sentiment, sequential pre-training |
| [`app/predictor.py`](app/predictor.py) | FinBERT loading/scoring, features, uncertainty, online pipeline, rolling backtest |
| [`app/scraper.py`](app/scraper.py) | RSS query, title cleaning, URL unwrapping, deduplication |
| [`app/sentiment.py`](app/sentiment.py) | Offline FinBERT scoring for the JSON news dataset |
| [`app/time_series.py`](app/time_series.py) | Standalone Prophet forecast artifact |
| [`app/combine_and_xai.py`](app/combine_and_xai.py) | Offline combined run and in-sample PPO diagnostics |
| [`app/evaluator.py`](app/evaluator.py) | Current evaluator, including synthetic sentiment and mapping caveat |
| [`app/api.py`](app/api.py) | FastAPI routes and stale legacy descriptions |
| [`app/frontend/index.html`](app/frontend/index.html) | Signal/probability display and feature-impact UI |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Design rationale and migration history; not always identical to current code |
| [`README.md`](README.md) | User-facing overview; contains older RF/SHAP language that should be updated |

## 18. Glossary

**Action:** A decision made by the agent.  
**Actor:** The policy network that produces action probabilities.  
**Advantage:** Estimated benefit of an action compared with the policy’s expected baseline.  
**Agent:** The learner making decisions.  
**Batch:** A group of rollout samples used for one optimizer update.  
**Categorical distribution:** A probability distribution over discrete choices.  
**Critic:** The value network estimating expected future reward.  
**Entropy:** A measure of uncertainty/randomness in a probability distribution.  
**Episode:** One complete environment run from reset to termination.  
**Feature:** A numeric input to the policy.  
**GAE:** Generalized Advantage Estimation, used to reduce advantage variance.  
**Gymnasium:** The environment interface used by the custom trading simulator.  
**On-policy:** Training from data generated by the current or recent policy.  
**Observation:** Information exposed to the agent.  
**PPO:** Proximal Policy Optimization.  
**Policy:** A mapping from observations to action probabilities.  
**Reward:** The scalar training signal after an action.  
**Return:** Discounted sum of future rewards.  
**Rollout:** A sequence of transitions collected before a PPO update.  
**State:** The complete environment situation; it can contain more information than the observation.  
**Transaction cost:** A penalty for changing directional exposure.  
**Walk-forward evaluation:** Training on past data and testing on the next chronological block, repeatedly.

## Final statement to memorize

“PPO was chosen because AlphaGaze is intended to learn a discrete trading policy from a reward that includes directional return and switching cost, while the dashboard requires a categorical action distribution. The environment is a chronological historical replay with seven engineered observations and three actions. The current implementation is a research prototype: it does not yet model a full portfolio, its policy probabilities are not calibrated confidence, and its strongest evaluation should be replaced by leakage-free walk-forward net-return testing. Those limitations define the next engineering steps rather than invalidating the architecture.”
