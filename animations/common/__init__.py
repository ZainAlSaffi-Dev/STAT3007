"""Shared building blocks for every scene in slides/ and explainer/.

A scene imports what it needs from here and keeps nothing of its own that
another scene could reuse. Each module has one job.

data       Loads saved runs and probe kernels from repoduced-code/results.
kernels    Numpy versions of the centred kernel terms, the aim and the torus average.
layout     Titles, captions, labels and heat maps, drawn the same way in every scene.
notation   The LaTeX strings for every symbol, matching the macros in docs/report.tex.
palette    Colours and type sizes.

A scene file adds the animations folder to sys.path and then imports these
modules, because Manim loads a scene file without a package around it. See
scene_template.py in the animations folder for the exact lines.
"""
