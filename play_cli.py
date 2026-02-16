from __future__ import annotations

import argparse
import asyncio
import glob
import os
import random
import re
from typing import List, Optional, Sequence, Tuple

import torch

from game_logic import (
    Card,
    Category,
    Deck,
    Suit,
    TieRevealEvent,
    best_hand_index,
    evaluate_hand,
    legal_indices,
    resolve_first_scoring_announcements,
)
from poker_ml import PolicyNet, build_draw_action_mask, build_obs_draw, build_obs_trick, sample_categorical

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, RichLog, Static
except ImportError as exc:
    raise SystemExit(
        "Textual is required for this interface. Install it with: python3 -m pip install textual"
    ) from exc


RANK_TO_STR = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

SUIT_TO_STR = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}


def bit_count(value: int) -> int:
    return bin(value).count("1")


def card_str(card: Card) -> str:
    return f"{RANK_TO_STR[int(card.rank)]}{SUIT_TO_STR[card.suit]}"


def remove_by_discard_mask(hand: List[Card], discard_mask_5bits: int) -> List[Card]:
    discard_indices = [i for i in range(5) if (discard_mask_5bits >> i) & 1]
    discarded_cards = [hand[i] for i in discard_indices]
    for i in sorted(discard_indices, reverse=True):
        hand.pop(i)
    return discarded_cards


def draw_action_to_mask_and_reject(action: int) -> Tuple[int, bool]:
    discard_mask = action & 0b1_1111
    reject_single_draw = ((action >> 5) & 1) == 1
    return discard_mask, reject_single_draw


def choose_bot_draw_action(
    model: PolicyNet,
    hand: Sequence[Card],
    phase: int,
    deck_cards_left: int,
    device: torch.device,
    scores: Sequence[int],
    last_round_scores: Sequence[int],
    n_players: int,
    self_player: int,
    target_score: int,
    showdown_starter: Optional[int],
    table_history: Sequence[Sequence[Optional[Card]]],
    discard_history: Sequence[Sequence[Sequence[Card]]],
    shown_history: Sequence[Optional[Tuple[Card, bool]]],
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    tie_reveal_events: Sequence[TieRevealEvent],
) -> int:
    obs = build_obs_draw(
        hand,
        phase=phase,
        scores=scores,
        last_round_scores=last_round_scores,
        n_players=n_players,
        self_player=self_player,
        target_score=target_score,
        showdown_starter=showdown_starter,
        table_history=table_history,
        discard_history=discard_history,
        shown_history=shown_history,
        announced_points=announced_points,
        passed=passed,
        tie_reveal_events=tie_reveal_events,
    ).to(device)
    with torch.no_grad():
        draw_logits, _, _ = model(obs)
    mask = build_draw_action_mask(deck_cards_left, phase=phase).to(device)
    return sample_categorical(draw_logits, mask=mask)


def choose_bot_trick_action(
    model: PolicyNet,
    hand: Sequence[Card],
    led_suit: Optional[Suit],
    trick_no: int,
    legal: Sequence[int],
    device: torch.device,
    scores: Sequence[int],
    last_round_scores: Sequence[int],
    n_players: int,
    self_player: int,
    target_score: int,
    table_history: Sequence[Sequence[Optional[Card]]],
    discard_history: Sequence[Sequence[Sequence[Card]]],
    shown_history: Sequence[Optional[Tuple[Card, bool]]],
    announced_points: Sequence[Optional[int]],
    passed: Sequence[bool],
    tie_reveal_events: Sequence[TieRevealEvent],
    showdown_starter: Optional[int],
) -> int:
    obs = build_obs_trick(
        hand,
        led_suit,
        trick_no,
        scores=scores,
        last_round_scores=last_round_scores,
        n_players=n_players,
        self_player=self_player,
        target_score=target_score,
        table_history=table_history,
        discard_history=discard_history,
        shown_history=shown_history,
        announced_points=announced_points,
        passed=passed,
        tie_reveal_events=tie_reveal_events,
        showdown_starter=showdown_starter,
    ).to(device)
    with torch.no_grad():
        _, trick_logits, _ = model(obs)
    mask = torch.zeros(5, dtype=torch.float32, device=device)
    for i in legal:
        mask[i] = 1.0
    action = sample_categorical(trick_logits, mask=mask)
    if action not in legal:
        return legal[0]
    return action


def category_name(cat: Category) -> str:
    names = {
        Category.HIGH: "High Card",
        Category.PAIR: "Pair",
        Category.TWOPAIR: "Two Pair",
        Category.TRIPS: "Trips",
        Category.STRAIGHT: "Straight",
        Category.FLUSH: "Flush",
        Category.FULLHOUSE: "Full House",
        Category.QUADS: "Quads",
        Category.STRAIGHT_FLUSH: "Straight Flush",
    }
    return names[cat]


def load_or_init_model(checkpoint_path: str, hidden: int, device: torch.device) -> Tuple[PolicyNet, str]:
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
            model_hidden = int(ckpt_args.get("hidden", hidden))
            model = PolicyNet(hidden=model_hidden).to(device)
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)
            model.eval()
            return model, f"Loaded model checkpoint: {checkpoint_path}"
        except Exception as e:
            model = PolicyNet(hidden=hidden).to(device)
            model.eval()
            return (
                model,
                "Found checkpoint but could not load it cleanly "
                f"({e.__class__.__name__}: {e}). Using random untrained model.",
            )

    model = PolicyNet(hidden=hidden).to(device)
    model.eval()
    return model, f"No checkpoint found at {checkpoint_path}; using random untrained model."


def _checkpoint_episode(path: str) -> int:
    m = re.search(r"checkpoint_ep(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def resolve_model_path(model_path: Optional[str], out_dir: str) -> Optional[str]:
    if model_path:
        return model_path

    pattern = os.path.join(out_dir, "checkpoint_ep*.pt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        checkpoints.sort(key=_checkpoint_episode)
        return checkpoints[-1]

    final_path = os.path.join(out_dir, "final_model.pt")
    if os.path.exists(final_path):
        return final_path
    return None


class PokerTUI(App[None]):
    CSS = """
    Screen {
        background: #0f1b2d;
        color: #f4f4f4;
    }

    #layout {
        height: 1fr;
    }

    #left_col {
        width: 35%;
        border: solid #2b4667;
        padding: 1 1;
    }

    #center_col {
        width: 65%;
        border: solid #2b4667;
        padding: 1 1;
    }

    #played_title {
        height: 2;
    }

    #table_view {
        height: 12;
        border: round #405f84;
        padding: 1 1;
        margin-bottom: 1;
    }

    #center_spacer {
        height: 1fr;
    }

    #startup_menu {
        height: 4;
        border: round #f6b73c;
        padding: 0 1;
        margin-bottom: 1;
    }

    #startup_buttons {
        height: 3;
        margin-bottom: 1;
    }

    .startup-opt {
        width: 11;
        margin-right: 1;
    }

    .startup-selected {
        border: tall #f6b73c;
        color: #ffe1a7;
    }

    #status {
        height: 5;
        border: round #405f84;
        padding: 0 1;
        margin-bottom: 1;
    }

    #scores {
        height: 8;
        border: round #405f84;
        padding: 0 1;
        margin-bottom: 1;
    }

    #trick_state {
        height: 12;
        border: round #405f84;
        padding: 0 1;
        margin-bottom: 1;
    }

    #log {
        height: 1fr;
        border: round #405f84;
    }

    #hand_title {
        height: 2;
    }

    #hand_row {
        height: 9;
        margin-bottom: 0;
    }

    #hint_row {
        height: 2;
        margin-bottom: 1;
    }

    .card {
        width: 16;
        height: 7;
        margin-right: 1;
        border: tall #6389b7;
    }

    .selected {
        border: tall #f6b73c;
    }

    .preview {
        border: ascii #f6b73c;
    }

    .hint-slot {
        width: 16;
        margin-right: 1;
        content-align: center middle;
        color: #5b6b7d;
    }

    .hint-on {
        color: #d83a56;
    }

    .helper {
        border: tall #d83a56;
        color: #ffd9df;
    }

    .legal {
        border: tall #3ccf7d;
    }

    .illegal {
        opacity: 45%;
    }

    #controls {
        height: 3;
    }

    .control {
        width: 16;
        margin-right: 1;
    }

    .confirm-helper-no-draw {
        border-bottom: tall #d83a56;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        if args.seed is None:
            self.seed = random.SystemRandom().randrange(0, 2**63)
        else:
            self.seed = args.seed
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        resolved_model_path = resolve_model_path(args.model_path, args.out_dir)
        if resolved_model_path is None:
            resolved_model_path = os.path.join(args.out_dir, "checkpoint_ep0.pt")
        self.model, self.model_msg = load_or_init_model(resolved_model_path, args.hidden, self.device)
        self.helper_model, self.helper_msg = load_or_init_model(resolved_model_path, args.hidden, self.device)

        self.awaiting_start_menu = args.bots is None
        self.selected_bots = 1 if args.bots is None else args.bots
        self.game_started = False

        self.n_players = 2
        self.names = ["You", "Bot 1"]
        self.scores = [0, 0]
        self.last_round_scores = [0, 0]
        self.circle_round: List[Optional[int]] = [None, None]
        self.announcement_order_start = 0
        self.starting_player = 0
        self.round_no = 0

        self.deck: Optional[Deck] = None
        self.hands: List[List[Card]] = []

        self.current_phase = -1
        self.current_player = -1
        self.current_trick_no = -1
        self.current_led_suit: Optional[Suit] = None
        self.current_trick_cards: List[Tuple[int, Card]] = []
        self.trick_history: List[str] = []
        self.table_trick_rows: List[List[Optional[Card]]] = []
        self.round_discard_history: List[List[List[Card]]] = [[[] for _ in range(self.n_players)] for _ in range(2)]
        self.round_shown_history: List[Optional[Tuple[Card, bool]]] = [None for _ in range(self.n_players)]
        self.round_announced_points: List[Optional[int]] = [None for _ in range(self.n_players)]
        self.round_passed: List[bool] = [False for _ in range(self.n_players)]
        self.round_tie_reveal_events: List[TieRevealEvent] = []

        self.input_mode = "idle"
        self.selected_discards: set[int] = set()
        self.current_legal_indices: set[int] = set()
        self.revealed_card: Optional[Card] = None
        self.helper_draw_indices: set[int] = set()
        self.helper_trick_index: Optional[int] = None
        self.helper_keep_choice: Optional[bool] = None
        self._input_future: Optional[asyncio.Future] = None

        self.game_over = False
        if not self.awaiting_start_menu:
            self._configure_players(self.selected_bots)

    def _configure_players(self, bots: int) -> None:
        self.args.bots = bots
        self.n_players = bots + 1
        self.names = ["You"] + [f"Bot {i}" for i in range(1, bots + 1)]
        self.scores = [0 for _ in range(self.n_players)]
        self.last_round_scores = [0 for _ in range(self.n_players)]
        self.circle_round = [None for _ in range(self.n_players)]
        self.announcement_order_start = self.rng.randrange(self.n_players)
        self.starting_player = self.announcement_order_start
        self.round_no = 0

        self.deck = None
        self.hands = []
        self.current_phase = -1
        self.current_player = -1
        self.current_trick_no = -1
        self.current_led_suit = None
        self.current_trick_cards = []
        self.trick_history = []
        self.table_trick_rows = []
        self.round_discard_history = [[[] for _ in range(self.n_players)] for _ in range(2)]
        self.round_shown_history = [None for _ in range(self.n_players)]
        self.round_announced_points = [None for _ in range(self.n_players)]
        self.round_passed = [False for _ in range(self.n_players)]
        self.round_tie_reveal_events = []
        self.input_mode = "idle"
        self.selected_discards.clear()
        self.current_legal_indices.clear()
        self.revealed_card = None
        self.helper_draw_indices = set()
        self.helper_trick_index = None
        self.helper_keep_choice = None
        self._input_future = None
        self.game_over = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Vertical(id="left_col"):
                yield Static(id="status")
                yield Static(id="scores")
                yield Static(id="trick_state")
                yield RichLog(id="log", highlight=False, markup=False, auto_scroll=True)
            with Vertical(id="center_col"):
                yield Static("", id="startup_menu")
                with Horizontal(id="startup_buttons"):
                    for i in range(1, 6):
                        yield Button(f"{i} Opp", id=f"startup-bots-{i}", classes="startup-opt")
                    yield Button("Start", id="startup-start", classes="startup-opt")
                yield Static("Cards On Table (Round History)", id="played_title")
                yield Static("", id="table_view")
                yield Static("", id="center_spacer")
                yield Static("Your Hand", id="hand_title")
                with Horizontal(id="hand_row"):
                    for i in range(5):
                        yield Button("", id=f"card-{i}", classes="card")
                with Horizontal(id="hint_row"):
                    for i in range(5):
                        yield Static(" ", id=f"hint-{i}", classes="hint-slot")
                with Horizontal(id="controls"):
                    yield Button("Confirm", id="confirm", classes="control")
                    yield Button("Clear", id="clear", classes="control")
                    yield Button("Keep", id="keep", classes="control")
                    yield Button("Reject", id="reject", classes="control")
        yield Footer()

    def on_mount(self) -> None:
        self._set_default_controls()
        self._refresh_all()
        self._log(self.model_msg)
        self._log(self.helper_msg)
        if self.awaiting_start_menu:
            self._log("Select opponent count in the startup menu, then press Start.")
        else:
            self._start_game()

    def _start_game(self) -> None:
        if self.game_started:
            return
        self.game_started = True
        self._log(f"Run seed: {self.seed}")
        self._log(f"Initial round leader selected randomly: {self.names[self.announcement_order_start]}")
        self._log("Helper active: red marker shows what the helper model would do.")
        self._log(f"Device: {self.device}")
        self.run_worker(self._game_loop(), exclusive=True)

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def _set_default_controls(self) -> None:
        self.query_one("#confirm", Button).disabled = True
        self.query_one("#clear", Button).disabled = True
        self.query_one("#keep", Button).disabled = True
        self.query_one("#reject", Button).disabled = True

    def _set_draw_controls(self) -> None:
        self.query_one("#confirm", Button).disabled = False
        self.query_one("#clear", Button).disabled = False
        self.query_one("#keep", Button).disabled = True
        self.query_one("#reject", Button).disabled = True

    def _set_keep_reject_controls(self) -> None:
        self.query_one("#confirm", Button).disabled = True
        self.query_one("#clear", Button).disabled = True
        self.query_one("#keep", Button).disabled = False
        self.query_one("#reject", Button).disabled = False

    def _refresh_all(self) -> None:
        self._refresh_startup_menu()
        self._refresh_status()
        self._refresh_scores()
        self._refresh_trick_state()
        self._refresh_table_cards()
        self._refresh_hand_buttons()
        self._refresh_hint_bars()
        self._refresh_helper_controls()

    def _refresh_startup_menu(self) -> None:
        startup_menu = self.query_one("#startup_menu", Static)
        startup_buttons = self.query_one("#startup_buttons", Horizontal)
        if self.awaiting_start_menu:
            startup_menu.display = True
            startup_buttons.display = True
            startup_menu.update("Startup Menu\nChoose number of opponents before beginning.")
        else:
            startup_menu.display = False
            startup_buttons.display = False

        for i in range(1, 6):
            btn = self.query_one(f"#startup-bots-{i}", Button)
            btn.remove_class("startup-selected")
            if self.awaiting_start_menu and i == self.selected_bots:
                btn.add_class("startup-selected")
            btn.disabled = not self.awaiting_start_menu
        self.query_one("#startup-start", Button).disabled = not self.awaiting_start_menu

    def _refresh_hint_bars(self) -> None:
        helper_indices: set[int] = set()
        if self.input_mode == "draw_select":
            helper_indices = set(self.helper_draw_indices)
        elif self.input_mode == "trick_select" and self.helper_trick_index is not None:
            helper_indices = {self.helper_trick_index}

        hand_len = len(self.hands[0]) if self.hands else 0
        for i in range(5):
            hint = self.query_one(f"#hint-{i}", Static)
            hint.remove_class("hint-on")
            if i < hand_len and i in helper_indices:
                hint.update("━━━━")
                hint.add_class("hint-on")
            else:
                hint.update(" ")

    def _refresh_table_cards(self) -> None:
        lines: List[str] = []
        header = "Trick  " + "  ".join(f"{name[:6]:>6}" for name in self.names)
        lines.append(header)
        lines.append("-" * len(header))

        if not self.table_trick_rows:
            lines.append("No trick cards played this round yet.")
        else:
            for trick_idx, row in enumerate(self.table_trick_rows, start=1):
                row_cards = "  ".join(f"{('--' if c is None else card_str(c)):>6}" for c in row)
                lines.append(f"T{trick_idx:<2}    {row_cards}")
        table = self.query_one("#table_view", Static)
        table.update("\n".join(lines))
        table.refresh(layout=False)

    def _refresh_status(self) -> None:
        phase_txt = "-"
        if self.current_phase == 0:
            phase_txt = "Draw 1"
        elif self.current_phase == 1:
            phase_txt = "Draw 2"
        elif self.current_phase == 2:
            phase_txt = "Trick"

        current_name = "-" if self.current_player < 0 else self.names[self.current_player]
        title = f"Round {self.round_no} | Phase: {phase_txt} | Turn: {current_name}"

        if self.awaiting_start_menu:
            prompt = "Use startup menu to choose opponents and press Start."
        elif self.input_mode == "draw_select":
            prompt = "Select cards to discard, then press Confirm."
        elif self.input_mode == "draw_reveal":
            prompt = "Shown card drawn. Choose Keep or Reject."
        elif self.input_mode == "trick_select":
            prompt = "Choose a legal card to play."
        elif self.game_over:
            prompt = "Game finished. Press q to quit."
        else:
            prompt = "Game running..."

        self.query_one("#status", Static).update(f"{title}\n{prompt}")

    def _refresh_scores(self) -> None:
        lines = ["Scores"]
        for i, name in enumerate(self.names):
            circle = self.circle_round[i]
            circle_text = "" if circle is None else f" (circle@r{circle})"
            lines.append(f"{name}: {self.scores[i]}{circle_text}")
        self.query_one("#scores", Static).update("\n".join(lines))

    def _refresh_trick_state(self) -> None:
        lines = ["Trick Board"]
        if self.current_trick_no >= 0:
            led = "-" if self.current_led_suit is None else SUIT_TO_STR[self.current_led_suit]
            lines.append(f"Current trick: {self.current_trick_no + 1}")
            lines.append(f"Led suit: {led}")
            if self.current_trick_cards:
                lines.append("Cards played:")
                for p, card in self.current_trick_cards:
                    lines.append(f"{self.names[p]}: {card_str(card)}")
        else:
            lines.append("No active trick.")

        if self.trick_history:
            lines.append("")
            lines.append("Recent:")
            lines.extend(self.trick_history[-3:])

        self.query_one("#trick_state", Static).update("\n".join(lines))

    def _refresh_hand_buttons(self) -> None:
        hand = self.hands[0] if self.hands else []
        display_cards = list(hand)
        preview_index: Optional[int] = None
        if self.input_mode == "draw_reveal" and self.revealed_card is not None and len(display_cards) < 5:
            preview_index = len(display_cards)
            display_cards.append(self.revealed_card)

        for i in range(5):
            btn = self.query_one(f"#card-{i}", Button)
            btn.remove_class("selected")
            btn.remove_class("legal")
            btn.remove_class("illegal")
            btn.remove_class("preview")

            if i < len(display_cards):
                btn.label = card_str(display_cards[i])
                btn.disabled = False
                if preview_index is not None and i == preview_index:
                    btn.add_class("preview")
                if self.input_mode == "draw_select" and i in self.selected_discards:
                    btn.add_class("selected")
                if self.input_mode == "trick_select":
                    if i in self.current_legal_indices:
                        btn.add_class("legal")
                    else:
                        btn.add_class("illegal")
            else:
                btn.label = ""
                btn.disabled = True

    def _refresh_helper_controls(self) -> None:
        keep_btn = self.query_one("#keep", Button)
        reject_btn = self.query_one("#reject", Button)
        confirm_btn = self.query_one("#confirm", Button)
        keep_btn.remove_class("helper")
        reject_btn.remove_class("helper")
        confirm_btn.remove_class("confirm-helper-no-draw")
        if self.input_mode == "draw_reveal" and self.helper_keep_choice is not None:
            if self.helper_keep_choice:
                keep_btn.add_class("helper")
            else:
                reject_btn.add_class("helper")
        elif self.input_mode == "draw_select" and not self.helper_draw_indices:
            confirm_btn.add_class("confirm-helper-no-draw")

    def _compute_helper_draw_hint(self, phase: int) -> None:
        if self.deck is None or not self.hands:
            self.helper_draw_indices = set()
            self.helper_keep_choice = None
            return
        action = choose_bot_draw_action(
            self.helper_model,
            self.hands[0],
            phase,
            len(self.deck.cards),
            self.device,
            scores=self.scores,
            last_round_scores=self.last_round_scores,
            n_players=self.n_players,
            self_player=0,
            target_score=self.args.target,
            showdown_starter=(self.starting_player if phase == 1 else self.announcement_order_start),
            table_history=self.table_trick_rows,
            discard_history=self.round_discard_history,
            shown_history=self.round_shown_history,
            announced_points=self.round_announced_points,
            passed=self.round_passed,
            tie_reveal_events=self.round_tie_reveal_events,
        )
        discard_mask, reject_single = draw_action_to_mask_and_reject(action)
        self.helper_draw_indices = {i for i in range(5) if ((discard_mask >> i) & 1) == 1}
        if phase == 1:
            self.helper_keep_choice = not reject_single
        else:
            self.helper_keep_choice = None

    def _compute_helper_trick_hint(
        self,
        legal: Sequence[int],
        trick_no: int,
        led_suit: Optional[Suit],
    ) -> None:
        if not self.hands:
            self.helper_trick_index = None
            return
        self.helper_trick_index = choose_bot_trick_action(
            self.helper_model,
            self.hands[0],
            led_suit,
            trick_no,
            legal,
            self.device,
            scores=self.scores,
            last_round_scores=self.last_round_scores,
            n_players=self.n_players,
            self_player=0,
            target_score=self.args.target,
            table_history=self.table_trick_rows,
            discard_history=self.round_discard_history,
            shown_history=self.round_shown_history,
            announced_points=self.round_announced_points,
            passed=self.round_passed,
            tie_reveal_events=self.round_tie_reveal_events,
            showdown_starter=self.starting_player,
        )

    def _clear_prompt_state(self) -> None:
        self.input_mode = "idle"
        self.selected_discards.clear()
        self.current_legal_indices.clear()
        self.revealed_card = None
        self.helper_draw_indices = set()
        self.helper_trick_index = None
        self._input_future = None
        self._set_default_controls()
        self._refresh_all()

    def _resolve_input(self, value) -> None:
        mode = self.input_mode
        future = self._input_future
        if future is not None and not future.done():
            future.set_result(value)
        if mode == "draw_reveal":
            self.helper_keep_choice = None
        self._clear_prompt_state()

    async def _prompt_human_draw(self, phase: int) -> int:
        self.input_mode = "draw_select"
        self.current_phase = phase
        self.selected_discards.clear()
        self._compute_helper_draw_hint(phase)
        self._set_draw_controls()
        self._refresh_all()
        loop = asyncio.get_running_loop()
        self._input_future = loop.create_future()
        return await self._input_future

    async def _prompt_human_keep_reject(self, shown_card: Card) -> bool:
        self.input_mode = "draw_reveal"
        self.revealed_card = shown_card
        self._set_keep_reject_controls()
        self._refresh_all()
        self._log(f"Shown card: {card_str(shown_card)}")
        loop = asyncio.get_running_loop()
        self._input_future = loop.create_future()
        return await self._input_future

    async def _prompt_human_trick(self, legal: Sequence[int], trick_no: int, led_suit: Optional[Suit]) -> int:
        self.input_mode = "trick_select"
        self.current_phase = 2
        self.current_trick_no = trick_no
        self.current_led_suit = led_suit
        self.current_legal_indices = set(legal)
        self._compute_helper_trick_hint(legal, trick_no, led_suit)
        self._set_default_controls()
        self._refresh_all()
        loop = asyncio.get_running_loop()
        self._input_future = loop.create_future()
        return await self._input_future

    async def _apply_draw(self, player: int, phase: int, discard_mask: int, reject_single: bool) -> None:
        assert self.deck is not None
        hand = self.hands[player]
        discarded_cards = remove_by_discard_mask(hand, discard_mask)
        n_discards = len(discarded_cards)
        self.round_discard_history[phase][player].extend(discarded_cards)

        if phase == 1 and n_discards == 1:
            shown = self.deck.draw(1)[0]
            self._log(f"{self.names[player]} drew shown card {card_str(shown)}.")
            if player == 0:
                keep = await self._prompt_human_keep_reject(shown)
                if keep:
                    hand.append(shown)
                    self._log("You kept the shown card.")
                else:
                    replacement = self.deck.draw(1)[0]
                    hand.append(replacement)
                    self._log(f"You rejected it and took {card_str(replacement)}.")
                self.round_shown_history[player] = (shown, keep)
            else:
                if reject_single and len(self.deck.cards) > 0:
                    replacement = self.deck.draw(1)[0]
                    hand.append(replacement)
                    self._log(f"{self.names[player]} rejected shown card and took a face-down replacement.")
                    self.round_shown_history[player] = (shown, False)
                else:
                    hand.append(shown)
                    self._log(f"{self.names[player]} kept the shown card.")
                    self.round_shown_history[player] = (shown, True)
        else:
            hand.extend(self.deck.draw(n_discards))
            if player == 0:
                self._log(f"You replaced {n_discards} card(s).")

        if len(hand) != 5:
            raise RuntimeError("Hand must remain 5 cards after draw")

        self._refresh_all()

    def _log_player_hand_categories(self) -> None:
        """Log each player's hand category."""
        for p in range(self.n_players):
            hand_val = evaluate_hand(self.hands[p])
            self._log(f"{self.names[p]}: {category_name(hand_val.category)}")

    def _run_first_scoring_announcements(self) -> None:
        self._log("--- Scoring Phase 1: Announcements ---")
        self._log_player_hand_categories()
        
        result = resolve_first_scoring_announcements(self.hands, start_player=self.announcement_order_start)
        self.round_announced_points = result.announced_points
        self.round_passed = result.passed
        self.round_tie_reveal_events = result.tie_reveal_events
        self.starting_player = result.showdown_starter

        for off in range(self.n_players):
            p = (self.announcement_order_start + off) % self.n_players
            announced = result.announced_points[p]
            if announced is None:
                self._log(f"{self.names[p]}: pass")
            else:
                self._log(f"{self.names[p]}: announces {announced} point(s)")

        self._log(
            f"{self.names[self.starting_player]} is this round's leader seat "
            "(draws first, announces first, and leads showdown)."
        )

        if result.tie_reveal_events:
            self._log("Tie-break reveal sequence:")
            for ev in result.tie_reveal_events:
                if ev.revealed_value is None:
                    self._log(
                        f"  {self.names[ev.player]} passes at tie step {ev.component_idx + 1}."
                    )
                else:
                    self._log(
                        f"  {self.names[ev.player]} reveals value {ev.revealed_value} at tie step {ev.component_idx + 1}."
                    )

        if result.scoring_winner is not None and result.winning_points > 0:
            winner = result.scoring_winner
            points = result.winning_points
            self.scores[winner] += points
            self._log(f"{self.names[winner]} wins Scoring Phase 1 (+{points}).")
            if self.scores[winner] >= self.args.target and self.circle_round[winner] is None:
                self.circle_round[winner] = self.round_no
                self._log(f"{self.names[winner]} reached {self.args.target}+ and has a circle.")
        else:
            self._log("No points announced in Scoring Phase 1.")

        self._refresh_scores()

    async def _run_draw_phase(self, phase: int) -> Optional[int]:
        phase_name = "Draw 1" if phase == 0 else "Draw 2"
        self.current_phase = phase
        self._log(f"--- {phase_name} ---")
        self._refresh_all()

        for off in range(self.n_players):
            p = (self.announcement_order_start + off) % self.n_players
            self.current_player = p
            self._refresh_all()
            if p == 0:
                discard_mask = await self._prompt_human_draw(phase)
                reject_single = False
            else:
                assert self.deck is not None
                action = choose_bot_draw_action(
                    self.model,
                    self.hands[p],
                    phase,
                    len(self.deck.cards),
                    self.device,
                    scores=self.scores,
                    last_round_scores=self.last_round_scores,
                    n_players=self.n_players,
                    self_player=p,
                    target_score=self.args.target,
                    showdown_starter=(self.starting_player if phase == 1 else self.announcement_order_start),
                    table_history=self.table_trick_rows,
                    discard_history=self.round_discard_history,
                    shown_history=self.round_shown_history,
                    announced_points=self.round_announced_points,
                    passed=self.round_passed,
                    tie_reveal_events=self.round_tie_reveal_events,
                )
                discard_mask, reject_single = draw_action_to_mask_and_reject(action)
                if phase == 0 or bit_count(discard_mask) != 1:
                    reject_single = False
                self._log(f"{self.names[p]} discards {bit_count(discard_mask)} card(s).")

            await self._apply_draw(p, phase, discard_mask, reject_single)

        winner, hv = best_hand_index(self.hands)
        if hv.category == Category.STRAIGHT_FLUSH:
            self._log(f"{self.names[winner]} made a Straight Flush and wins immediately.")
            self.game_over = True
            self._refresh_all()
            return winner

        if phase == 0:
            self._run_first_scoring_announcements()
            return None

        # Scoring Phase 2 - show each player's hand
        self._log("--- Scoring Phase 2 ---")
        self._log_player_hand_categories()

        if hv.points > 0:
            self.scores[winner] += hv.points
            self._log(f"{self.names[winner]} wins {phase_name} with {category_name(hv.category)} (+{hv.points}).")
            if self.scores[winner] >= self.args.target and self.circle_round[winner] is None:
                self.circle_round[winner] = self.round_no
                self._log(f"{self.names[winner]} reached {self.args.target}+ and has a circle.")
        else:
            self._log(f"No scoring hand in {phase_name}.")

        self._refresh_scores()
        return None

    async def _run_showdown(self) -> int:
        self.current_phase = 2
        self._log("--- Trick-Taking Showdown ---")
        trick_hands = [h[:] for h in self.hands]
        leader = self.starting_player
        last_winner = leader

        for trick_no in range(5):
            self.current_trick_no = trick_no
            self.current_led_suit = None
            self.current_trick_cards = []
            self.table_trick_rows.append([None for _ in range(self.n_players)])
            self._log(f"Trick {trick_no + 1}, leader: {self.names[leader]}")
            self._refresh_all()
            await asyncio.sleep(0)

            for off in range(self.n_players):
                p = (leader + off) % self.n_players
                self.current_player = p
                hand = trick_hands[p]
                legal = legal_indices(hand, self.current_led_suit)
                self._refresh_status()

                if p == 0:
                    action = await self._prompt_human_trick(legal, trick_no, self.current_led_suit)
                else:
                    action = choose_bot_trick_action(
                        self.model,
                        hand,
                        self.current_led_suit,
                        trick_no,
                        legal,
                        self.device,
                        scores=self.scores,
                        last_round_scores=self.last_round_scores,
                        n_players=self.n_players,
                        self_player=p,
                        target_score=self.args.target,
                        table_history=self.table_trick_rows,
                        discard_history=self.round_discard_history,
                        shown_history=self.round_shown_history,
                        announced_points=self.round_announced_points,
                        passed=self.round_passed,
                        tie_reveal_events=self.round_tie_reveal_events,
                        showdown_starter=self.starting_player,
                    )

                card = hand.pop(action)
                if self.current_led_suit is None:
                    self.current_led_suit = card.suit
                self.current_trick_cards.append((p, card))
                self.table_trick_rows[trick_no][p] = card
                self._log(f"{self.names[p]} plays {card_str(card)}")

                if p == 0:
                    self.hands[0] = hand
                    self._refresh_hand_buttons()
                self._refresh_trick_state()
                self._refresh_table_cards()
                await asyncio.sleep(0.12)

            eligible = [(p, c) for (p, c) in self.current_trick_cards if c.suit == self.current_led_suit]
            winner, _ = max(eligible, key=lambda pc: (int(pc[1].rank), int(pc[1].suit)))
            leader = winner
            last_winner = winner
            summary = f"Trick {trick_no + 1} winner: {self.names[winner]}"
            self.trick_history.append(summary)
            self._log(summary)
            self._refresh_trick_state()

        self.current_trick_no = -1
        self.current_led_suit = None
        self.current_trick_cards = []
        self._refresh_trick_state()
        self._refresh_table_cards()
        return last_winner

    async def _game_loop(self) -> None:
        while not self.game_over:
            self.round_no += 1
            self._log(f"=== Round {self.round_no} ===")
            self.last_round_scores = self.scores.copy()

            self.deck = Deck(self.rng)
            self.hands = [self.deck.draw(5) for _ in range(self.n_players)]
            self.starting_player = self.announcement_order_start
            self.current_player = 0
            self.trick_history.clear()
            self.table_trick_rows = []
            self.round_discard_history = [[[] for _ in range(self.n_players)] for _ in range(2)]
            self.round_shown_history = [None for _ in range(self.n_players)]
            self.round_announced_points = [None for _ in range(self.n_players)]
            self.round_passed = [False for _ in range(self.n_players)]
            self.round_tie_reveal_events = []
            self._refresh_all()

            draw1_winner = await self._run_draw_phase(0)
            if draw1_winner is not None:
                self.game_over = True
                break

            draw2_winner = await self._run_draw_phase(1)
            if draw2_winner is not None:
                self.game_over = True
                break

            last_winner = await self._run_showdown()
            self.scores[last_winner] += 5
            self._log(f"{self.names[last_winner]} wins final trick and gains +5.")

            if self.scores[last_winner] >= self.args.target and self.circle_round[last_winner] is None:
                self.circle_round[last_winner] = self.round_no
                self._log(f"{self.names[last_winner]} reached {self.args.target}+ and has a circle.")

            self._refresh_scores()
            self.announcement_order_start = (self.announcement_order_start + 1) % self.n_players

            if self.circle_round[last_winner] is not None and self.circle_round[last_winner] < self.round_no:
                self._log(f"{self.names[last_winner]} had a circle before this round and won showdown.")
                self._log(f"{self.names[last_winner]} wins the game.")
                self.game_over = True

        self.current_player = -1
        self._clear_prompt_state()
        self.game_over = True
        self._refresh_all()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return

        if button_id.startswith("startup-bots-") and self.awaiting_start_menu:
            self.selected_bots = int(button_id.split("-")[-1])
            self._refresh_startup_menu()
            return

        if button_id == "startup-start" and self.awaiting_start_menu:
            self.awaiting_start_menu = False
            self._configure_players(self.selected_bots)
            self._refresh_all()
            self._log(f"Selected opponents: {self.selected_bots}")
            self._start_game()
            return

        if button_id.startswith("card-"):
            idx = int(button_id.split("-")[1])
            if self.input_mode == "draw_select":
                hand_len = len(self.hands[0]) if self.hands else 0
                if idx < hand_len:
                    if idx in self.selected_discards:
                        self.selected_discards.remove(idx)
                    else:
                        self.selected_discards.add(idx)
                    self._refresh_hand_buttons()
            elif self.input_mode == "trick_select":
                if idx in self.current_legal_indices:
                    self._resolve_input(idx)
                else:
                    self._log("That card is not legal for this trick.")
            return

        if button_id == "clear" and self.input_mode == "draw_select":
            self.selected_discards.clear()
            self._refresh_hand_buttons()
            return

        if button_id == "confirm" and self.input_mode == "draw_select":
            discard_mask = 0
            for idx in self.selected_discards:
                discard_mask |= 1 << idx

            n_discards = len(self.selected_discards)
            needed_cards = n_discards
            if self.current_phase == 1 and n_discards == 1:
                needed_cards = 2

            if self.deck is None:
                self._log("Internal error: deck unavailable.")
                return

            if needed_cards > len(self.deck.cards):
                self._log(
                    f"Not enough cards left for that discard choice "
                    f"(need {needed_cards}, left {len(self.deck.cards)})."
                )
                return

            self._resolve_input(discard_mask)
            return

        if button_id == "keep" and self.input_mode == "draw_reveal":
            self._resolve_input(True)
            return

        if button_id == "reject" and self.input_mode == "draw_reveal":
            if self.deck is None or len(self.deck.cards) == 0:
                self._log("No replacement card available; must keep.")
                return
            self._resolve_input(False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play 2-draw showdown poker in a Textual TUI.")
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit model path. Default: latest checkpoint in --out-dir.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="rl_runs",
        help="Training output directory used to auto-pick latest checkpoint.",
    )
    p.add_argument("--hidden", type=int, default=256, help="Hidden size for random fallback model.")
    p.add_argument(
        "--bots",
        type=int,
        default=None,
        help="Number of model opponents (1-5). If omitted, pick in the TUI startup menu.",
    )
    p.add_argument("--target", type=int, default=50, help="Score target for getting a circle.")
    p.add_argument("--seed", type=int, default=None, help="Optional deterministic seed. Default: random each run.")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    if args.bots is not None and (args.bots < 1 or args.bots > 5):
        raise SystemExit("--bots must be between 1 and 5.")
    return args


if __name__ == "__main__":
    app = PokerTUI(parse_args())
    app.run()
