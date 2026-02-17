from __future__ import annotations

# Centralized Textual stylesheet for the poker TUI. Keeping this separate from
# app control logic makes UI adjustments easier and safer.
APP_CSS = """
Screen {
    background: #0f1b2d;
    color: #f4f4f4;
}

Button {
    text-style: none;
}

Button:hover {
    text-style: none;
}

Button:focus {
    text-style: none;
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
