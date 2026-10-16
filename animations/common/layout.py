"""Titles, captions and heat maps drawn the same way in every scene.

A caption is one short sentence on one line. caption() raises when the text
would not fit on one line of the frame, so a long caption fails the render
instead of reaching the deck.
"""

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, RESAMPLING_ALGORITHMS, Axes, Circle, ImageMobject, Text, VGroup,
                   VMobject, config)
from manim.utils.images import change_to_rgba_array

from . import palette

MARGIN = 0.5

# Every scene imports this module, so this one line gives every Text the font of the deck.
Text.set_default(font=palette.FONT)


def tag(text, size=palette.TAG):
    """A small grey note in the bottom right corner, such as "schematic"."""
    return Text(text, font_size=size, color=palette.GRAY).to_corner(DOWN + RIGHT, buff=0.3)


def step_axes(x_length, y_length, y_range=(0, 1, 1), y_labels=(0, 1), last_step=200_000,
              labelled=(1, 100, 10_000), size=palette.LABEL_COLUMN):
    """Axes with the training step on a log scale, and their tick labels.

    The x coordinate is log10 of the step, so step 1 sits at 0 and a tick marks
    every power of ten. Plot a series with axes.plot_line_graph(np.log10(steps),
    values) after leaving out step 0. The steps in labelled carry a number, and
    so do the values in y_labels. Returns the axes and a group of the labels;
    move the two together.
    """
    axes = Axes(x_range=(0, np.log10(last_step), 1), y_range=y_range, x_length=x_length, y_length=y_length,
                tips=False, axis_config=dict(color=palette.INK, stroke_width=3, tick_size=0.1))
    labels = VGroup()
    for step in labelled:
        number = Text(f"{step:,}", font_size=size, color=palette.INK)
        labels.add(number.next_to(axes.c2p(np.log10(step), y_range[0]), DOWN, buff=0.25))
    for value in y_labels:
        number = Text(f"{value:g}", font_size=size, color=palette.INK)
        labels.add(number.next_to(axes.c2p(0, value), LEFT, buff=0.25))
    return axes, labels


def event_marker(point, colour, filled):
    """A round marker for an event on a curve: hollow for memorisation, filled for grokking."""
    return Circle(radius=0.13, stroke_color=colour, stroke_width=6, fill_color=colour if filled else palette.PAPER,
                  fill_opacity=1).move_to(point)


def curve_until(axes, x, y, x_end, colour, width=7):
    """The part of a measured curve left of x_end, ending exactly at x_end.

    Call it inside always_redraw with x_end read from a ValueTracker, so that
    several curves sweep across the axes together.
    """
    keep = x <= x_end
    xs, ys = x[keep], y[keep]
    if keep.any() and not keep.all():
        xs, ys = np.append(xs, x_end), np.append(ys, np.interp(x_end, x, y))
    points = [axes.c2p(a, b) for a, b in zip(xs, ys)] or [axes.c2p(x[0], y[0])]
    line = VMobject(stroke_color=colour, stroke_width=width)
    line.set_points_as_corners(points if len(points) > 1 else points * 2)
    return line


def _check_width(mob, what, text):
    room = config.frame_width - 2 * MARGIN
    if mob.width > room:
        raise ValueError(f"The {what} '{text}' is {mob.width:.2f} units wide and the frame allows {room:.2f}.")
    return mob


def title(text):
    """A title in the top left corner."""
    mob = Text(text, font_size=palette.TITLE, color=palette.INK)
    _check_width(mob, "title", text)
    return mob.to_corner(UP + LEFT, buff=MARGIN)


def caption(text):
    """A one-line caption at the bottom of the frame."""
    mob = Text(text, font_size=palette.CAPTION, color=palette.INK)
    _check_width(mob, "caption", text)
    return mob.to_edge(DOWN, buff=MARGIN)


def label(text, color=palette.INK):
    """A short plain-text label."""
    return Text(text, font_size=palette.LABEL, color=color)


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def diverging_pixels(values, vmax):
    """Map a 2D array to RGB pixels, NEGATIVE below zero, PAPER at zero, POSITIVE above.

    Values beyond plus or minus vmax are drawn at full colour.
    """
    x = np.clip(np.asarray(values, float) / vmax, -1.0, 1.0)[..., None]
    paper, neg, pos = _rgb(palette.PAPER), _rgb(palette.NEGATIVE), _rgb(palette.POSITIVE)
    rgb = np.where(x >= 0, paper + x * (pos - paper), paper - x * (neg - paper))
    return rgb.round().astype(np.uint8)


def heat_map(values, vmax, side):
    """A square image of a 2D array with one sharp block per entry."""
    image = ImageMobject(diverging_pixels(values, vmax))
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.height = side
    return image


def grey_map(values, vmax, side):
    """A square image of a non-negative 2D array, PAPER at zero and INK at vmax or above."""
    x = np.clip(np.asarray(values, float) / vmax, 0.0, 1.0)[..., None]
    pixels = (_rgb(palette.PAPER) + x * (_rgb(palette.INK) - _rgb(palette.PAPER))).round().astype(np.uint8)
    image = ImageMobject(pixels)
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.height = side
    return image


def set_pixels(image, values, vmax):
    """Redraw a heat map in place, for use inside an updater."""
    image.pixel_array = change_to_rgba_array(diverging_pixels(values, vmax))
