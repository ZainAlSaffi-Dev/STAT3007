"""Collect the speaker notes of slides.qmd into script.md, and check the length of every slide.

The deck is the source of the script. Each slide's notes start with "Turn N." and then hold the
parts "Message:", "Say:", "Point at:" and "If asked:", each a list of full sentences. This script
reads the notes of every slide before the backup, fills each var shortcode from _variables.yml,
groups the slides by turn, and writes script.md. It then appends questions.md, the answers to the
questions we expect, with its numbers filled in the same way.

Every "TODO:" in the notes is printed in bold. It also prints two checks for every slide. The first compares the words under "Say:" with the
time budget of the slide at 150 words a minute. The second counts the words a viewer reads on the
slide, leaving out formulas, tables, figure labels, clip storyboards, citations and source lines.
A slide is flagged when either count is over its limit.

Run it from the repository root:

    python presentation/make_script.py
"""
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDES = HERE / 'slides.qmd'
VARIABLES = HERE / '_variables.yml'
QUESTIONS = HERE / 'questions.md'
SCRIPT = HERE / 'script.md'

WORDS_PER_MINUTE = 150
SLIDE_WORD_CAP = 40
# Divs whose text a viewer does not read as prose: clip storyboards, bar charts, source lines and
# hidden macros. The notes are the script, not the slide.
SKIP_CLASSES = ('anim', 'g2-bars', 'g2-source', 'notes', 'hidden')


def load_variables():
    values = {}
    for line in VARIABLES.read_text().splitlines():
        m = re.match(r'^(\w+): "(.*)"', line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def fill(text, values, spoken=True):
    """Put each var shortcode's value in place. Spoken text drops the dollar signs of inline math."""
    def value(m):
        v = values[m.group(1)]
        return v.replace('$', '') if spoken else v
    return re.sub(r'\{\{<\s*var\s+(\w+)\s*>\}\}', value, text)


def seconds(budget):
    m, s = budget.split(':')
    return 60 * int(m) + int(s)


def clock(total):
    return f'{total // 60}:{total % 60:02d}'


def parse(text):
    """The slides before the backup, each with its kind, title, budget, body lines and notes."""
    body = text.split('\n---\n', 1)[1] if text.startswith('---') else text
    slides = []
    for line in body.splitlines():
        heading = re.match(r'^(#{1,2}) ?(.*?)\s*(\{.*\})?\s*$', line)
        if heading and not line.startswith('###'):
            level, title, attrs = heading.group(1), heading.group(2), heading.group(3) or ''
            if level == '#' and '.backup' in attrs:
                break
            budget = re.search(r'time="([0-9:]+)"', attrs)
            kind = 'act' if level == '#' else ('title' if 'g2-title-slide' in attrs else 'slide')
            slides.append({'kind': kind, 'title': title or 'Title: Scale or Shape?', 'budget': budget.group(1)
                           if budget else '0:00', 'lines': []})
            continue
        if slides:
            slides[-1]['lines'].append(line)
    for slide in slides:
        slide['notes'], slide['body'] = split_notes(slide['lines'])
    return slides


def split_notes(lines):
    """The lines of the notes div, and the lines a viewer reads on the slide."""
    notes, body, stack = [], [], []
    fence_raw = False
    for line in lines:
        if line.startswith('```'):
            fence_raw = not fence_raw
            continue
        if fence_raw:
            continue
        fence = re.match(r'^:{3,}\s*(\{.*\})?\s*$', line)
        if fence:
            if fence.group(1):
                stack.append(fence.group(1))
            elif stack:
                stack.pop()
            continue
        if any('.notes' in attrs for attrs in stack):
            notes.append(line)
        elif not any(f'.{c}' in attrs for attrs in stack for c in SKIP_CLASSES):
            body.append(line)
    return notes, body


def note_parts(notes):
    """The turn, the message and the three lists of a slide's notes."""
    text = '\n'.join(notes)
    turn = re.search(r'Turn (\d+)\.', text)
    parts = {'Message': '', 'Say': [], 'Point at': [], 'If asked': []}
    current = None
    for line in notes:
        head = re.match(r'^(Message|Say|Point at|If asked):\s*(.*)$', line)
        if head:
            current = head.group(1)
            if current == 'Message':
                parts['Message'] = head.group(2)
            continue
        if current == 'Message' and line.strip():
            parts['Message'] += ' ' + line.strip()
        elif current in ('Say', 'Point at', 'If asked'):
            if line.startswith('- '):
                parts[current].append(line[2:].strip())
            elif line.strip() and parts[current]:
                parts[current][-1] += ' ' + line.strip()
    return (int(turn.group(1)) if turn else None), parts


def on_slide_words(body, values):
    """The words a viewer reads on a slide."""
    text = '\n'.join(line for line in body if not line.lstrip().startswith('|'))
    text = fill(text, values, spoken=False)
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.S)
    text = re.sub(r'\$[^$]*\$', ' x ', text)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?', ' ', text)
    text = re.sub(r'\[@[^\]]*\]', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\{[^}]*\}', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return len(re.findall(r"[A-Za-z0-9][\w'’.,-]*", text))


def main():
    values = load_variables()
    slides = parse(SLIDES.read_text())
    out = [
        '# Script for the G2 seminar',
        '',
        f'This script is written by `make_script.py` from the speaker notes in `slides.qmd`, on {date.today():%d %B %Y}.'
        ' Do not edit it by hand. Change the notes in the deck and run the script again.',
        '',
        'Each slide gives its number in the browser, its budget and the running total. "Say" is what to speak,'
        ' at about 150 words a minute. "Point at" says how to read the figure. "If asked" holds detail for'
        ' questions, which is not part of the timed talk. The answers to the questions we expect are at the end.',
        '',
    ]
    turns = {}
    for number, slide in enumerate(slides, start=1):
        slide['number'] = number
        slide['turn'], slide['parts'] = note_parts(slide['notes'])
        turns.setdefault(slide['turn'], []).append(slide)

    total = 0
    report = []
    out.append('| Turn | Slides | Time |')
    out.append('|---|---|---|')
    for turn, group in sorted(turns.items(), key=lambda kv: kv[0] or 0):
        span = sum(seconds(s['budget']) for s in group)
        out.append(f"| {turn} | {group[0]['number']} to {group[-1]['number']} | {clock(span)} |")
    out.append('')

    for turn, group in sorted(turns.items(), key=lambda kv: kv[0] or 0):
        span = sum(seconds(s['budget']) for s in group)
        out.append(f'## Turn {turn} ({clock(span)})')
        out.append('')
        for slide in group:
            budget = seconds(slide['budget'])
            total += budget
            kind = 'Act' if slide['kind'] == 'act' else 'Slide'
            out.append(f"### {kind} {slide['number']}. {fill(slide['title'], values)} ({slide['budget']}, total {clock(total)})")
            out.append('')
            parts = slide['parts']
            if parts['Message']:
                out.append(f"*Message:* {fill(parts['Message'], values)}")
                out.append('')
            for name in ('Say', 'Point at', 'If asked'):
                if parts[name]:
                    out.append(f'**{name}**')
                    out.append('')
                    out.extend(f'- {fill(point, values)}' for point in parts[name])
                    out.append('')
            said = sum(len(fill(p, values).split()) for p in parts['Say'])
            allowed = round(budget * WORDS_PER_MINUTE / 60)
            shown = on_slide_words(slide['body'], values) if slide['kind'] == 'slide' else 0
            flags = []
            if said > 1.1 * allowed:
                flags.append('script long')
            if shown > SLIDE_WORD_CAP:
                flags.append('slide text long')
            report.append((slide['number'], slide['title'][:58], slide['budget'], said, allowed, shown, flags))

    out.append('## Answers to the questions we expect')
    out.append('')
    questions = QUESTIONS.read_text().split('\n', 1)[1].lstrip('\n')
    out.append(fill(questions, values).rstrip())
    out.append('')
    # A task left in the notes is printed in bold, so that nobody reads past it.
    SCRIPT.write_text(re.sub(r'(?<!\*)TODO:', '**TODO:**', '\n'.join(out)))

    print(f'wrote {SCRIPT.relative_to(HERE.parent)}, {clock(total)} in all')
    print(f"{'#':>3}  {'slide':58}  {'time':>5}  {'said':>4}/{'max':<4}  {'shown':>5}  flags")
    for number, title, budget, said, allowed, shown, flags in report:
        print(f"{number:>3}  {title:58}  {budget:>5}  {said:>4}/{allowed:<4}  {shown:>5}  {', '.join(flags)}")


if __name__ == '__main__':
    main()
