from __future__ import annotations

import argparse
import asyncio
import os
import random
from typing import List, Optional, Sequence, Tuple

import torch

from game_logic import (
    Card,
    Category,
    Deck,
    Suit,
    TieRevealEvent,
    best_hand_index,
    legal_indices,
    resolve_first_scoring_announcements,
)
from poker_ml import PolicyNet
from poker_cli.bot_policy import (
    choose_bot_draw_action,
    choose_bot_trick_action,
    draw_action_to_mask_and_reject,
)
from poker_cli.display import (
    announced_hand_name,
    bit_count,
    card_str,
    category_name,
    remove_by_discard_mask,
)
from poker_cli.model_io import load_or_init_model, resolve_model_path
from poker_cli.styles import APP_CSS

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, RichLog, Static
except ImportError as exc:
    raise SystemExit(
        "Textual is required for this interface. Install it with: python3 -m pip install textual"
    ) from exc


class PokerTUI(App[None]):
    """Textual app that orchestrates an interactive poker game against model bots."""

    CSS = APP_CSS

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

        # Runtime/model initialization.
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

        # Persistent game state (updated across rounds).
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

        # Round-local and UI interaction state.
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
        """Reinitialize all mutable state when opponent count changes."""
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
        for button in self.query(Button):
            button.can_focus = False
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
        """Redraw every visible panel from current in-memory state."""
        self._refresh_startup_menu()
        self._refresh_status()
        self._refresh_scores()
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
        """Populate helper suggestion for draw selection / keep-reject."""
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
        """Populate helper suggestion for trick play."""
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
        """Return UI input state to neutral mode after a prompt resolves."""
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
        """Resolve the currently awaited user prompt future."""
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
        """
        Apply one player's draw action and emit matching public logs.

        This mirrors game rules for draw-2 single-card reveal/replace behavior.
        """
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

    def _run_first_scoring_announcements(self) -> None:
        """Resolve and log phase-1 announcements, tie-break, and score award."""
        self._log("--- Scoring Phase 1: Announcements ---")
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
                self._log(f"{self.names[p]}: announces {announced_hand_name(announced)}")

        if result.tie_reveal_events:
            self._log("Tie-break reveal sequence:")
            for ev in result.tie_reveal_events:
                if ev.outcome == "pass":
                    self._log(
                        f"  {self.names[ev.player]}: pass (tie step {ev.component_idx + 1})."
                    )
                elif ev.outcome == "higher":
                    self._log(
                        f"  {self.names[ev.player]}: Higher (tie step {ev.component_idx + 1})."
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
        """Execute one full draw phase and its immediate scoring consequences."""
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

        # Scoring Phase 2
        self._log("--- Scoring Phase 2 ---")

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
        """Play the 5-trick showdown and return the final trick winner."""
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
                self._refresh_table_cards()
                await asyncio.sleep(0.12)

            eligible = [(p, c) for (p, c) in self.current_trick_cards if c.suit == self.current_led_suit]
            winner, _ = max(eligible, key=lambda pc: (int(pc[1].rank), int(pc[1].suit)))
            leader = winner
            last_winner = winner
            summary = f"Trick {trick_no + 1} winner: {self.names[winner]}"
            self.trick_history.append(summary)
            self._log(summary)

        self.current_trick_no = -1
        self.current_led_suit = None
        self.current_trick_cards = []
        self._refresh_table_cards()
        return last_winner

    async def _game_loop(self) -> None:
        """Top-level round loop until the game reaches terminal condition."""
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
        """Route all button events according to current input mode."""
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
    """Parse CLI arguments for TUI gameplay session."""
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
