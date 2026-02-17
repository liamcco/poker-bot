from poker_cli.app import PokerTUI, parse_args


if __name__ == "__main__":
    app = PokerTUI(parse_args())
    app.run()
