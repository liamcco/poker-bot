from __future__ import annotations

"""Compatibility entrypoint for the Textual poker CLI.

Implementation now lives in the `poker_cli` package to keep concerns separated
and make the codebase easier to navigate.
"""

from poker_cli.app import PokerTUI, parse_args


def main() -> None:
    app = PokerTUI(parse_args())
    app.run()


if __name__ == "__main__":
    main()
