# Poker Bot Training Workflow (ML)

This document explains how training is executed in this repository, what actually improves policy quality over time, and which changes are most likely to improve speed and sample efficiency.

This complements `MODEL.md`:
- `MODEL.md` = model architecture and observation/action encoding.
- `ML.md` = execution pipeline, optimization dynamics, bottlenecks, and roadmap.

## 1. Source-of-Truth Files

- Training entrypoint: `main.py`
- Environment simulation + model + rollout logic: `poker_ml.py`
- Game rules and transitions: `game_logic/`
  - `game_logic/cards.py`
  - `game_logic/draw.py`
  - `game_logic/hand_eval.py`
  - `game_logic/scoring.py`
  - `game_logic/trick.py`

## 2. End-to-End Training Execution

### 2.1 Process setup

`main.py` does the following once:
1. Parse hyperparameters and guards (`parse_args`).
2. Choose device (`cpu` or `cuda`).
3. Seed Python RNG and PyTorch RNG.
4. Create one `PolicyNet`.
5. Create one Adam optimizer.
6. Build a cyclic schedule for number of opponents:
   - `n_bots` cycles through `[min_bots, ..., max_bots]` each episode.

### 2.2 One training episode

Per episode (`main.py:train`), execution is:
1. Pick player count from cycle.
2. Generate one self-play game trajectory with `run_episode(...)`.
3. Compute losses over all players' trajectories.
4. One optimizer update.
5. Periodic logging / evaluation / checkpointing.

## 3. Rollout Generation (`run_episode`) in Detail

`run_episode` in `poker_ml.py` creates trajectories for all players and returns:
- `trajs`: per-player list of `StepRecord`s
- `winner`: game winner
- `scores`: final score vector
- `rounds`: number of rounds played

### 3.1 Internal game loop

While game is not won:
1. Initialize round state:
   - fresh shuffled `Deck`
   - dealt hands
   - observation context containers (`table_history`, `discard_history`, `shown_history`, announcement context)
2. Draw phase 1 decisions (clockwise):
   - build draw observation
   - model samples draw action under legal mask
   - apply action with `apply_draw`
   - store `StepRecord`
3. Straight flush early termination check.
4. Announcement scoring resolution:
   - `resolve_first_scoring_announcements`
   - update scores/rewards for phase-1 winner
   - update tie-reveal context
5. Draw phase 2 decisions (same pattern as phase 1).
6. Straight flush early termination check.
7. Scoring phase 2:
   - `best_hand_index`
   - award score + reward
8. Trick-taking showdown (5 tricks):
   - build trick observation each action
   - sample legal action
   - resolve each trick winner
   - final trick winner gains +5 and reward
9. Check circle-win condition and terminate if satisfied.

### 3.2 What gets stored per decision

Each `StepRecord` contains:
- `obs`: full 4,087-dim observation
- `action`: sampled action id
- `phase`: 0 (draw1), 1 (draw2), 2 (trick)
- `legal_mask`: action legality mask used at sampling
- `reward`: initialized to 0, later incremented when scoring events happen

## 4. Reward and Credit Assignment

Rewards are sparse, delayed, and event-driven.

### 4.1 Immediate reward events

A player's most recent step gets reward increments when events occur:
- Scoring phase 1 winner: `+winning_points / target_score`
- Scoring phase 2 winner: `+hand_points / target_score`
- Final trick winner: `+5 / target_score`
- Game winner bonus: `+winner_bonus`

Straight flush gives only winner bonus and immediate episode end.

### 4.2 Returns

For each player's trajectory:
- discounted return `G_t` is computed backward via `compute_returns` with `gamma`.

So policy credit for early decisions is propagated only through discounted future rewards.

## 5. Optimization Objective (Current)

For each step record:
- Run model forward again on stored observation.
- Rebuild masked action distribution for the relevant head.
- Compute:
  - `logp = log pi(a_t | s_t)`
  - `entropy = H(pi(.|s_t))`
  - `adv = G_t - V(s_t)`

Loss terms:
- Policy term: `L_policy = -(adv * logp)`
- Value term: `L_value = (V(s_t) - G_t)^2`
- Entropy term: `L_entropy = H(pi(.|s_t))`

Final scalar loss:
- `L = mean(L_policy) + value_coef * mean(L_value) - entropy_coef * mean(L_entropy)`

Then:
- `zero_grad`
- `backward`
- gradient clipping
- `optimizer.step()`

## 6. Why the Model Improves

The model improves through on-policy self-play and shared-parameter updates:
1. All seats use the same policy parameters, so one episode provides many state-action-reward tuples.
2. Legal-action masks reduce gradient noise from impossible moves.
3. Value head reduces variance of policy gradient via baseline subtraction.
4. Entropy regularization avoids early policy collapse.
5. Opponent-count cycling exposes different strategic regimes (2..5 players by default in schedule).

## 7. Evaluation Workflow (Current)

Periodic eval (`evaluate_vs_fresh_random`):
- Creates a brand-new random `PolicyNet` every eval game.
- Plays learner vs random baseline, alternating seats.
- Reports win-rate over `eval_episodes`.

Important characteristics:
- Baseline is weak and high-variance.
- Useful for coarse sanity checks.
- Not strong for measuring strategic progress after basic competency.

## 8. Observed Runtime Bottlenecks

### 8.1 Structural bottlenecks (code-level)

1. Rollout is Python-heavy:
- repeated observation assembly
- repeated tensor allocation (`torch.zeros`, one-hots)
- repeated sampling object creation (`Categorical`)

2. Training pass is step-by-step (unbatched):
- one forward per step in Python loop
- poor hardware utilization (especially GPU)

3. Rollout action selection is done without explicit `torch.no_grad()` in `run_episode`.
- incurs autograd tracking overhead during simulation (not needed there)

4. Draw action masks are recomputed frequently though input space is tiny and cacheable.

5. Data movement overhead:
- trajectory stores CPU tensors (`obs.detach().cpu()`), then training moves each back to device.

### 8.2 Quick local profile snapshot

Profiling run (local, CPU) showed:
- large cumulative time in:
  - `run_episode`
  - `build_context_features` / observation builders
  - model forward calls
  - categorical sampling + masking
- in one end-to-end train step, backward dominated single-step wall time, with rollout still substantial.

Interpretation:
- both simulation-side Python overhead and training-side compute matter.
- speedups should target both rollout and update pipeline.

### 8.3 Concrete profiling numbers captured

From a local `cProfile` pass over several episodes:
- `run_episode` dominated total rollout wall time.
- Hot functions included:
  - `build_context_features`
  - `build_obs_trick`
  - `forward` (many small calls)
  - `sample_categorical`
  - `encode_discard_history`
  - `build_draw_action_mask`

From a local end-to-end train-step profile (single episode + backward):
- backward pass was the largest single block in wall time.
- rollout and encoding still consumed a substantial fraction.

From a quick rollout micro-benchmark on CPU:
- baseline 10 rollouts: about `1.111s`
- wrapped in outer `torch.no_grad()`: about `1.023s`
- observed speedup: about `1.09x`

These measurements are machine-dependent, but they validate that:
1. removing rollout autograd overhead gives immediate gains, and
2. observation/sampling overhead is significant enough to optimize directly.

## 9. Correctness/Information-Policy Risk

Current encoding includes exact discarded cards for all players in `encode_discard_history`.
This may leak hidden information under imperfect-information rules, because opponents should not know exact discarded identities in normal play.

Impact:
- Policy may learn to rely on privileged signals.
- Offline metrics may overstate real-game performance.

Recommendation:
- Replace opponent discard identity encoding with public-only signals (counts, timing, or coarser summaries).

## 10. Optimization Opportunities

Below is a prioritized list from low-risk small changes to larger algorithmic changes.

### 10.1 Small changes (high ROI, low risk)

1. Use `torch.no_grad()` during rollout forward passes in `run_episode`.
- Expected: immediate rollout speedup, no behavior change.
- Risk: minimal.

2. Cache draw masks for `(cards_left, phase)`.
- There are few unique combinations.
- Expected: modest speedup, less per-step Python work.

3. Reduce tensor allocation churn in observation encoders.
- Reuse preallocated buffers where practical.
- Expected: moderate speedup in simulation.

4. Batch training forwards.
- Group records by phase/head and run larger forward batches.
- Expected: significant speedup on GPU, moderate on CPU.

5. Keep trajectory tensors on device (or move in bigger batches).
- Reduce per-step host/device transfers.

### 10.2 Medium changes (moderate complexity)

1. Multi-episode rollout before update.
- Collect N episodes then do one update.
- Lowers gradient variance.
- Better hardware utilization with batching.

2. Advantage normalization.
- Normalize advantages per update batch.
- Often stabilizes and accelerates learning.

3. Add GAE-style advantage estimation.
- Better bias/variance tradeoff than pure Monte Carlo returns.

4. Parallel rollout workers (vectorized actors/processes).
- Large throughput gains if CPU-bound in environment simulation.

### 10.3 Large changes (algorithmic upgrades)

1. Move from REINFORCE-style single-pass update to PPO.
- Clipped objective + minibatch epochs per rollout.
- Typically much better sample efficiency and stability.

2. Self-play opponent pool / league training.
- Train against frozen snapshots + current policy mix.
- Reduces catastrophic forgetting and policy cycling.

3. Stronger evaluation ladder.
- Fixed heuristic bots + historical checkpoints + random.
- Better signal than fresh-random-only evaluation.

## 11. Practical Roadmap

### Phase A (fast wins)
1. Add `no_grad` to rollout forwards.
2. Cache draw masks.
3. Advantage normalization.
4. Improve eval baseline (add fixed checkpoints).

### Phase B (throughput)
1. Batch training forward/loss computation.
2. Multi-episode update batches.
3. Reduce allocation churn in encoders.

### Phase C (algorithmic)
1. Implement PPO update loop.
2. Add opponent pool.
3. Revisit reward shaping only after PPO baseline stabilizes.

## 12. Expected Impact Summary

- Fastest low-risk speedups:
  - rollout `no_grad`
  - mask caching
  - batched update computations

- Biggest sample-efficiency gains:
  - PPO
  - GAE
  - multi-episode batch updates
  - stronger self-play opponent strategy

- Biggest realism/correctness gain:
  - remove hidden-information leakage from discard identity encoding.

## 13. Notes for Future Documentation Sync

If the training pipeline changes, update both:
- `ML.md` (workflow/optimization details)
- `MODEL.md` (input semantics, especially any observation encoding changes)

These two docs should evolve together to avoid checkpoint incompatibility confusion.
