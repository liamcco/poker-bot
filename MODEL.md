# Poker Bot Model Documentation

This document describes the neural network model used by the poker bot, including the exact input and output formats, model architecture, and all information available to the model during gameplay.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Information Available to the Model](#information-available-to-the-model)
- [Encoding Schemes](#encoding-schemes)

---

## Model Architecture

### Model Type
**Feed-forward neural network** (Multi-layer Perceptron) with separate output heads for different action types and value estimation.

### Network Structure

```
Input Layer:    4,087 dimensions (observation vector)
      ↓
Hidden Layer 1: 256 neurons + ReLU activation
      ↓
Hidden Layer 2: 256 neurons + ReLU activation
      ↓
      ├─→ Discard Head:  64 outputs (draw phase actions)
      ├─→ Trick Head:    5 outputs (trick-taking actions)
      └─→ Value Head:    1 output (state value estimation)
```

### Parameters

- **Total parameters**: 1,130,310
- **Trainable parameters**: 1,130,310

#### Layer-by-Layer Breakdown

| Layer | Shape | Parameters |
|-------|-------|------------|
| fc1.weight | 256 × 4,087 | 1,046,272 |
| fc1.bias | 256 | 256 |
| fc2.weight | 256 × 256 | 65,536 |
| fc2.bias | 256 | 256 |
| discard_head.weight | 64 × 256 | 16,384 |
| discard_head.bias | 64 | 64 |
| trick_head.weight | 5 × 256 | 1,280 |
| trick_head.bias | 5 | 5 |
| value_head.weight | 1 × 256 | 256 |
| value_head.bias | 1 | 1 |

### Training Algorithm

- **Method**: Policy Gradient (REINFORCE-style)
- **Optimizer**: Adam (default lr=3e-4)
- **Loss Components**:
  - Policy loss (advantage-weighted log probability)
  - Value loss (MSE between predicted value and actual returns)
  - Entropy regularization (encourages exploration)
- **Discount factor (γ)**: 0.99 (default)
- **Gradient clipping**: 1.0 (default)

---

## Input Format

The model receives a **single observation vector of 4,087 dimensions**. This vector is constructed differently depending on the game phase (draw or trick-taking).

### Input Dimension Breakdown

| Component | Dimensions | Description |
|-----------|------------|-------------|
| **Hand Encoding** | 260 | Current hand (5 cards × 52-card one-hot) |
| **Phase Indicator** | 3 | One-hot: [draw_phase_0, draw_phase_1, trick_phase] |
| **Led Suit** | 5 | One-hot: [none, clubs, diamonds, hearts, spades] |
| **Trick Number** | 5 | One-hot: [trick_0, trick_1, trick_2, trick_3, trick_4] |
| **Score Context** | 18 | Player scores, active flags, self-player indicator |
| **Last Round Scores** | 6 | Previous round's final scores (normalized) |
| **Table History** | 1,560 | Cards played in all tricks (5 tricks × 6 players × 52) |
| **Discard History** | 624 | Discarded cards (2 phases × 6 players × 52) |
| **Shown History** | 330 | Revealed single-draw cards (6 players × 55) |
| **Announcement Context** | 60 | Own announced points + public announce/pass flags (6 players × 10) |
| **Tie Reveal Events** | 1,216 | Tie-breaking reveals (64 events × 19) |
| **TOTAL** | **4,087** | |

### Phase-Specific Observations

#### Draw Phase (Phase 0 or 1)
- Hand contains exactly 5 cards
- Phase indicator shows which draw phase (0 or 1)
- Led suit is always index 0 (no suit)
- Trick number is always index 0 (no trick yet)

#### Trick-Taking Phase (Phase 2)
- Hand contains 0-5 cards (decreases as tricks are played)
- Phase indicator shows trick phase (index 2)
- Led suit indicates the suit led this trick (or index 0 if not yet led)
- Trick number indicates current trick (0-4)

---

## Output Format

The model produces **three separate outputs** from the shared hidden representation:

### 1. Discard Head (64 logits)

Used during **draw phases** (phase 0 and 1).

**Encoding**: Each action is a 6-bit integer (0-63) where:
- **Bits 0-4**: Discard mask (which of the 5 cards to discard)
  - Bit 0 = discard card at index 0
  - Bit 1 = discard card at index 1
  - Bit 2 = discard card at index 2
  - Bit 3 = discard card at index 3
  - Bit 4 = discard card at index 4
- **Bit 5**: Reject single-draw flag (only valid when exactly 1 card is discarded in phase 1)

**Legal actions** are masked based on:
- Number of cards remaining in the deck
- Current phase (bit 5 only valid in phase 1)
- Number of discards (bit 5 only valid when discarding exactly 1 card)

**Examples**:
- Action 0 = `000000` = Discard nothing (stand pat)
- Action 1 = `000001` = Discard card at index 0
- Action 3 = `000011` = Discard cards at indices 0 and 1
- Action 31 = `011111` = Discard all 5 cards
- Action 33 = `100001` = Discard card at index 0 AND reject the drawn card (phase 1 only)

### 2. Trick Head (5 logits)

Used during **trick-taking phase** (phase 2).

Each logit corresponds to playing the card at that index in the current hand (0-4).

**Legal actions** are determined by:
- Cards remaining in hand
- Must follow suit if possible (if any cards match the led suit)

### 3. Value Head (1 scalar)

Estimates the **expected cumulative discounted reward** from the current state for the current player.

Used for:
- Training the value function (baseline for advantage estimation)
- Not used for action selection during play

---

## Information Available to the Model

The model has access to **all legal information** according to the game rules. This includes:

### Perfect Information
- **Own hand**: The 5 cards currently held by the player
- **Current scores**: All players' current point totals
- **Last round scores**: All players' scores from the previous round
- **Table history**: All cards played in all tricks so far this round
- **Discard counts**: Number of cards discarded by each player in each draw phase
- **Shown cards**: Cards revealed during single-card draws in phase 2
- **Announcements**: Own announced points plus public announce/pass state for all players in phase 1
- **Tie reveals**: Incremental information revealed during announcement tie-breaking
- **Game state**: Current phase, trick number, led suit

### Imperfect Information (Hidden)
The model does **NOT** have access to:
- Other players' hands (except cards that have been played or revealed)
- Cards remaining in the deck
- Exact cards discarded by opponents (only knows they discarded, not which cards)

### Information Encoding Details

All information is encoded in the observation vector as described in the [Encoding Schemes](#encoding-schemes) section below.

---

## Encoding Schemes

This section details exactly how each component of the observation is encoded.

### Hand Encoding (260 dimensions)

**Method**: Fixed 5-slot one-hot encoding per card position

- 5 card slots × 52 possible cards = 260 dimensions
- Each card slot uses a 52-dimensional one-hot vector
- Card ID mapping: `suit_index * 13 + rank_index`
  - Suit index: Clubs=0, Diamonds=1, Hearts=2, Spades=3
  - Rank index: 2=0, 3=1, ..., Ace=12
- Empty slots (during tricks) are encoded as all zeros

**Example**: Hand with [A♠, K♥, Q♦, J♣, 10♠]
- Slot 0: one-hot at position 51 (Ace of Spades)
- Slot 1: one-hot at position 37 (King of Hearts)
- Slot 2: one-hot at position 24 (Queen of Diamonds)
- Slot 3: one-hot at position 10 (Jack of Clubs)
- Slot 4: one-hot at position 47 (10 of Spades)

### Phase Indicator (3 dimensions)

One-hot encoding:
- `[1, 0, 0]` = Draw phase 0
- `[0, 1, 0]` = Draw phase 1
- `[0, 0, 1]` = Trick-taking phase

### Led Suit (5 dimensions)

One-hot encoding:
- `[1, 0, 0, 0, 0]` = No suit led yet (or not in trick phase)
- `[0, 1, 0, 0, 0]` = Clubs
- `[0, 0, 1, 0, 0]` = Diamonds
- `[0, 0, 0, 1, 0]` = Hearts
- `[0, 0, 0, 0, 1]` = Spades

### Trick Number (5 dimensions)

One-hot encoding for current trick (0-4):
- `[1, 0, 0, 0, 0]` = Trick 0 (or not in trick phase)
- `[0, 1, 0, 0, 0]` = Trick 1
- `[0, 0, 1, 0, 0]` = Trick 2
- `[0, 0, 0, 1, 0]` = Trick 3
- `[0, 0, 0, 0, 1]` = Trick 4

### Score Context (18 dimensions)

**Structure**: 3 vectors of 6 dimensions each (MAX_PLAYERS = 6)

1. **Scores** (6 floats): Each player's current score normalized by target score
   - Value = `score / target_score`
   - Padded with zeros if fewer than 6 players

2. **Active markers** (6 floats): Indicates which player slots are active
   - `0.0` = inactive player slot
   - `1.0` = active player
   - `2.0` = active player AND showdown starter (marks the key seat for the round)

3. **Self-player** (6-dim one-hot): Indicates which player is "me"

### Last Round Scores (6 dimensions)

Normalized scores from the end of the previous round:
- Value = `last_round_score / target_score`
- Padded with zeros for inactive player slots

### Table History (1,560 dimensions)

**Structure**: 5 tricks × 6 players × 52-card one-hot

Encodes all cards played in all tricks so far:
- For each trick (0-4), for each player (0-5): 52-dim one-hot of the card played
- `None` (not played yet) encoded as all zeros
- Total: 5 × 6 × 52 = 1,560 dimensions

### Discard History (624 dimensions)

**Structure**: 2 draw phases × 6 players × 52-card one-hot

Encodes which cards were discarded:
- For each draw phase, for each player: 52-dim vector with 1.0 for each discarded card
- Multiple cards can be discarded, so this is NOT one-hot (it's multi-hot)
- Total: 2 × 6 × 52 = 624 dimensions

### Shown History (330 dimensions)

**Structure**: 6 players × 55 dimensions per player

For each player, encodes the shown card during single-card draw in phase 2:
- **Card shown** (52-dim one-hot): Which card was revealed
- **Shown flag** (1 bit): 1.0 if a card was shown, 0.0 otherwise
- **Kept flag** (1 bit): 1.0 if the shown card was kept, 0.0 if rejected
- **Rejected flag** (1 bit): 1.0 if the shown card was rejected, 0.0 if kept

Total: 6 × (52 + 3) = 330 dimensions

### Announcement Context (60 dimensions)

**Structure**: 6 players × 10 dimensions per player

For each player, encodes announcements from phase 1:
- **Announced points** (8-dim one-hot): Point value announced (0-7), only populated for the observing player
- **Announced flag** (1 bit): 1.0 if player announced, 0.0 otherwise
- **Pass flag** (1 bit): 1.0 if player passed, 0.0 otherwise

Total: 6 × (8 + 2) = 60 dimensions

### Tie Reveal Events (1,216 dimensions)

**Structure**: 64 events × 19 dimensions per event

Encodes the sequential tie-breaking reveals during announcement resolution:
- **Player** (6-dim one-hot): Which player revealed
- **Component index** (10-dim one-hot): Which tie-break component (0-9)
- **Value present flag** (1 bit): 1.0 if a value was revealed
- **Value absent flag** (1 bit): 1.0 if player passed
- **Revealed value** (1 float): Normalized value (0-14) / 14.0

Total: 64 × (6 + 10 + 2 + 1) = 1,216 dimensions

If both flags are 0, the event represents **Higher** (beats current best without revealing exact value).

**Note**: Most event slots are empty (all zeros) in typical games.

---

## Constants and Configuration

Key constants defined in `poker_ml.py`:

```python
MAX_PLAYERS = 6               # Maximum players supported
MAX_TRICKS = 5                # Tricks per round
DRAW_PHASES = 2               # Draw phases per round
ANNOUNCE_POINT_BINS = 8       # Point values (0-7)
MAX_TIE_COMPONENTS = 10       # Tie-break components
MAX_TIE_REVEAL_EVENTS = 64    # Tie-break reveal events
```

### Configurable Hyperparameters

From `main.py` training defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden` | 256 | Hidden layer size |
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `value_coef` | 0.5 | Value loss coefficient |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `grad_clip` | 1.0 | Gradient clipping threshold |
| `winner_bonus` | 0.25 | Bonus reward for winning |
| `target` | 50 | Target score to win |

---

## Action Selection

During gameplay, actions are selected using **categorical sampling** from the policy:

1. Model outputs raw logits for the appropriate head (discard or trick)
2. Illegal actions are masked by setting their logits to -1e9
3. Softmax converts logits to probabilities
4. Action is sampled from the categorical distribution

During training, this stochastic sampling enables exploration. During evaluation, the same stochastic policy is used (not greedy/deterministic).

---

## Notes

- The observation shape (4,087) is **fixed** regardless of the number of active players (2-6)
- Unused player slots are padded with zeros
- The model is **position-dependent**: it must learn to interpret the "self-player" indicator
- All scores and values are **normalized** by the target score (default 50) for stable learning
- The model does **not** use recurrent connections; each decision is based solely on the current observation
- **Compatibility**: Model checkpoints may become incompatible if the observation encoding changes
