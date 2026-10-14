"""Copy the report's numbers into _variables.yml for the slides.

The report keeps every result as a macro in report_v2/overleaf/numbers.tex,
which report_v2/code/make_all.py writes from the item scripts. This script
reads that file and writes each macro as a Quarto variable with the same name.
A slide then shows a number with {{< var EsixMainA >}} and never types a result
by hand, so the slides and the report cannot drift apart.

Run it from anywhere, with any Python 3.9 or later. It needs no packages:

    python presentation/make_numbers.py

Run it again whenever make_all.py changes numbers.tex, then render the deck.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "report_v2" / "overleaf" / "numbers.tex"
TARGET = HERE / "_variables.yml"

# One macro per line: \newcommand{\Name}{value}% source
MACRO = re.compile(r"^\\newcommand\{\\(?P<name>[A-Za-z]+)\}\{(?P<value>.*)\}%\s*(?P<source>.*)$")

SUPERSCRIPT_POWER = re.compile(r"10\^\{(?P<power>-?\d+)\}")


def plain(value):
    """Turn the LaTeX of one number into text that Quarto renders as Markdown.

    A thin space between thousands becomes a comma, a minus sign becomes the
    Unicode minus, and a power of ten becomes a Markdown superscript, so that
    3\\times10^{-4} reads as 3 × 10^−4^.
    """
    text = value.strip()
    wrapped = re.fullmatch(r"\\ensuremath\{(.*)\}", text)
    if wrapped:
        text = wrapped.group(1)
    text = text.replace("\\,", ",")
    text = text.replace("\\%", "%")
    text = SUPERSCRIPT_POWER.sub(lambda m: "10^" + m.group("power").replace("-", "\u2212") + "^", text)
    text = text.replace("\\times", " \u00d7 ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("-"):
        text = "\u2212" + text[1:]
    if "\\" in text:
        raise ValueError(f"cannot turn {value!r} into plain text")
    return text


def main():
    if not SOURCE.exists():
        sys.exit(f"{SOURCE} does not exist. Run report_v2/code/make_all.py first.")
    lines = [
        "# Written by presentation/make_numbers.py from report_v2/overleaf/numbers.tex.",
        "# Do not edit by hand. Run the script again after report_v2/code/make_all.py.",
        "# Each name is the macro name in numbers.tex, and the comment gives its source.",
    ]
    count = 0
    for raw in SOURCE.read_text(encoding="utf-8").splitlines():
        match = MACRO.match(raw.strip())
        if not match:
            continue
        text = plain(match.group("value"))
        quoted = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{match.group("name")}: "{quoted}"  # {match.group("source")}')
        count += 1
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {count} numbers to {TARGET.relative_to(HERE.parent)}.")


if __name__ == "__main__":
    main()
