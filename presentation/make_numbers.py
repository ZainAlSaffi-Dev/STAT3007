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

def markdown(value):
    """Turn the LaTeX of one number into the Markdown that the slides show.

    A value that the report sets in math mode, inside \\ensuremath, becomes
    inline math, so that MathJax renders a sign, a power of ten or a comparison
    as LaTeX does in the report: \\ensuremath{3\\times10^{-4}} becomes
    $3\\times10^{-4}$. Any other value is a plain number. A thin space between
    thousands becomes a comma and \\% becomes %. No Unicode stands in for math.
    """
    text = value.strip()
    wrapped = re.fullmatch(r"\\ensuremath\{(.*)\}", text)
    if wrapped:
        return "$" + wrapped.group(1).strip() + "$"
    text = text.replace("\\,", ",").replace("\\%", "%")
    if re.search(r"[\\$^_]", text) or text.startswith(("-", "+")):
        raise ValueError(f"{value!r} holds math outside \\ensuremath, so it cannot be written as a plain number")
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
        text = markdown(match.group("value"))
        quoted = text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{match.group("name")}: "{quoted}"  # {match.group("source")}')
        count += 1
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {count} numbers to {TARGET.relative_to(HERE.parent)}.")


if __name__ == "__main__":
    main()
