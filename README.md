# Poker Bot

Train and play a reinforcement-learning poker agent for a custom Scandinavian-style home game:

- 2 draw phases
- announcement-based scoring with partial information reveals
- trick-taking showdown where only the last trick scores (+5)
- race-to-target win condition with the "circle" rule

Game rules are documented in `RULES.md`.  
Model architecture and input/output specifications are documented in `MODEL.md`.

## Project Structure

- `game_logic.py`: core game entities and rules (cards, hand evaluation, draw logic, announcement/tie resolution, legal trick plays)
- `poker_ml.py`: model, observation encoding, self-play episode simulation, and utility functions
- `main.py`: training entrypoint (policy-gradient style loop, checkpointing, periodic eval vs fresh random actor)
- `play_cli.py`: Textual TUI for human play against the latest trained checkpoint

## Requirements

- Python 3.9+
- PyTorch
- Textual

Install dependencies:

```bash
python3 -m pip install torch textual
```

## Training

Start training:

```bash
python3 main.py --cuda
```

If no CUDA GPU:

```bash
python3 main.py
```

Useful flags:

- `--episodes`: total training episodes
- `--min-bots` / `--max-bots`: opponent-count cycle range (defaults to 1..4, rotates every episode)
- `--save-every`: checkpoint interval (episodes)
- `--eval-every`: evaluate interval (episodes)
- `--eval-episodes`: number of eval games vs newly initialized random actor
- `--out-dir`: checkpoint/model output directory (default `rl_runs`)

Example:

```bash
python3 main.py --episodes 20000 --min-bots 1 --max-bots 4 --save-every 1000 --eval-every 500 --eval-episodes 20 --out-dir rl_runs --cuda
```

### Training Logs

Main training log line:

- `ep`: current episode
- `loss`: total objective
- `policy`: policy loss term
- `value`: value loss term
- `ent`: entropy term
- `bots_this_ep`: opponents used this episode
- `last_scores`: final scores from the last simulated game

Periodic eval log line:

- `vs_fresh_random_winrate`: win rate of current model against a newly initialized random model
- `eval_games`: number of evaluation games

## Checkpoints

During training:

- `checkpoint_ep<episode>.pt` is saved every `--save-every` episodes
- `final_model.pt` is saved at the end

All artifacts are written to `--out-dir` (default `rl_runs`).

## Play in the TUI

Launch:

```bash
python3 play_cli.py
```

Behavior:

- If `--bots` is not provided, pick opponent count (1-5) in the TUI startup menu
- The TUI auto-loads the latest checkpoint from `--out-dir`
- If no checkpoint exists, it falls back to a random (untrained) model

Optional explicit model path:

```bash
python3 play_cli.py --model-path rl_runs/checkpoint_ep4000.pt
```

Optional deterministic run seed:

```bash
python3 play_cli.py --seed 123
```

## Notes

- Because observation/model input shape evolves with rule changes, older checkpoints may become incompatible with newer code.
- In that case, start a fresh training run to generate compatible checkpoints.
