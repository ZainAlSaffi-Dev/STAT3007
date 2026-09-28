"""Concept animations of eigenvectors, eigenbases and the infinite-width kernel.

Each scene draws a worked example from a located source, with its settings in
``config/animations.yaml`` under the scene's own name. Axler (2024), Linear
Algebra Done Right, gives the definitions: ``lambda`` is an eigenvalue of ``T``
when ``Tv = lambda v`` for some ``v != 0``, so that ``span(v)`` is a
one-dimensional invariant subspace (5.5, 5.8). ``T`` has a diagonal matrix in
some basis exactly when that basis consists of eigenvectors (5.55), and a
self-adjoint ``T`` has such a basis that is orthonormal (7.29).

The vector-space scenes follow Manim's documented ``LinearTransformationScene``
template, which moves the plane, its vectors and their labels together. Styling
comes from that template and the ColorBrewer ``Dark2`` palette of the report's
figures, taken in order of appearance, and each eigenvector keeps its colour
across scenes (de Koning et al. 2009, p. 123). Transitions keep Manim's default
slow-in/slow-out pacing, which Dragicevic et al. (2011) found most accurate for
tracking.
"""

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from manim import (
    DOWN,
    UR,
    ApplyMatrix,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    Group,
    Indicate,
    Integer,
    LinearTransformationScene,
    LogBase,
    ManimColor,
    MathTex,
    NumberPlane,
    Scene,
    ValueTracker,
    Variable,
    VGroup,
    VMobject,
    always_redraw,
    config,
)
from matplotlib import color_sequences
from matplotlib.colors import to_hex

from deeplearning import kernels

SETTINGS = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "config" / "animations.yaml").read_text(
        encoding="utf-8"
    )
)["animations"]
PALETTE = tuple(map(ManimColor, map(to_hex, color_sequences["Dark2"])))


class Eigenvectors(LinearTransformationScene):
    """Vectors that stay on their span, Radmehr (2024) and Beltran-Meneu et al. (2016).

    The matrix of the 3Blue1Brown video that Radmehr (2024, pp. 10-11)
    transcribes stretches ``(1, 0)`` by 3 and ``(-1, 1)`` by 2, and each stays
    on its dashed span. Their sum is ``(0, 1)``, which the matrix knocks off its
    span, the point of question 1.b of Beltran-Meneu et al. (2016): a sum of
    eigenvectors with different eigenvalues is not an eigenvector. Each
    eigenvalue is the Rayleigh quotient ``v^T A v / v^T v`` of its eigenvector.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Hide the basis vectors, since the eigenvectors and their sum replace them.

        Args:
            kwargs: Passed on to ``LinearTransformationScene``.
        """
        super().__init__(show_basis_vectors=False, **kwargs)

    def construct(self) -> None:
        """Draw both eigenvectors, their sum and their spans, then apply the matrix."""
        settings = SETTINGS[type(self).__name__]
        matrix = np.array(settings["matrix"])
        vectors = np.array(settings["eigenvectors"])
        eigenvalues = np.einsum("ki,ij,kj->k", vectors, matrix, vectors) / np.einsum(
            "ki,ki->k", vectors, vectors
        )
        vectors = np.vstack([vectors, vectors.sum(0)])
        extent = np.hypot(config.frame_width, config.frame_height)
        self.play(
            *(
                Create(
                    DashedLine(
                        -np.append(vector, 0), np.append(vector, 0), color=colour
                    ).set_length(extent)
                )
                for vector, colour in zip(vectors, PALETTE, strict=False)
            )
        )
        arrows = [
            self.add_vector(vector, color=colour)
            for vector, colour in zip(vectors, PALETTE, strict=False)
        ]
        for arrow, label, eigenvalue in zip(
            arrows,
            ("v_1", "v_2", "v_1 + v_2"),
            (*eigenvalues, None),
            strict=True,
        ):
            self.add_transformable_label(
                arrow,
                label,
                transformation_name="A",
                new_label=None if eigenvalue is None else f"{eigenvalue:g} {label}",
                at_tip=True,
            )
        self.apply_matrix(matrix)
        self.wait()


class Rotation(LinearTransformationScene):
    """A map with no real eigenvectors, Axler (2024), Example 5.9(a).

    ``T(w, z) = (-z, w)`` rotates the plane by 90 degrees, so every nonzero
    vector leaves its span and ``T`` has no real eigenvalues.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Keep ghosts of the basis vectors to show where they started.

        Args:
            kwargs: Passed on to ``LinearTransformationScene``.
        """
        super().__init__(leave_ghost_vectors=True, **kwargs)

    def construct(self) -> None:
        """Title the map with Axler's formula and rotate the plane."""
        self.add_title(MathTex("T(w, z) = (-z, w)"))
        self.apply_matrix(SETTINGS[type(self).__name__]["matrix"])
        self.wait()


class Eigenbasis(LinearTransformationScene):
    """The same matrix as a scaling in its eigenbasis, Radmehr (2024), p. 14.

    With the eigenvectors as the columns of ``C``, ``D = C^{-1} A C`` is diagonal
    (Axler 5.55), so ``A = C D C^{-1}``: change to eigenvector coordinates,
    scale each axis by its eigenvalue, and change back. The title's factor is
    indicated just before each step plays (de Koning et al. 2009, guideline 2).
    In Manim 0.21 every ``play`` overwrites the list of mobjects that
    ``apply_matrix`` moves, so a transformation after another animation fails;
    ManimCommunity/manim#4726 fixes this, and the scene renders from the first
    release that includes it.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Hide the basis vectors, since the eigenvectors become the new basis.

        Args:
            kwargs: Passed on to ``LinearTransformationScene``.
        """
        super().__init__(show_basis_vectors=False, **kwargs)

    def construct(self) -> None:
        """Apply ``C^{-1}``, then ``D``, then ``C``, indicating each factor."""
        settings = SETTINGS[type(self).__name__]
        matrix = np.array(settings["matrix"])
        basis = np.array(settings["eigenvectors"]).T
        diagonal = np.linalg.solve(basis, matrix @ basis)
        title = MathTex("A", "=", "C", "D", "C^{-1}")
        product, _, change, scaling, inverse = title
        self.add_title(title)
        for vector, colour in zip(basis.T, PALETTE, strict=False):
            self.add_vector(vector, color=colour)
        self.play(Indicate(inverse))
        self.apply_inverse(basis)
        self.play(Indicate(scaling))
        self.apply_matrix(diagonal)
        self.play(Indicate(change))
        self.apply_matrix(basis)
        self.play(Indicate(product))
        self.wait()


class Ellipse(Scene):
    """A symmetric matrix maps the unit circle onto an ellipse, Beltran-Meneu et al. (2016).

    Their Figure 2 shows the symmetric inertia matrix taking the unit circle to
    an ellipse whose axes are its eigenvectors, where ``Av`` keeps the direction
    of ``v`` (their Equation 3.2). By the real spectral theorem (Axler 7.29)
    those axes are orthogonal, as they are for every kernel Gram matrix. ``v``
    sweeps the circle and stops at each eigenvector, which is indicated there.
    """

    def construct(self) -> None:
        """Map the circle, then sweep ``v`` and ``Av`` round it."""
        matrix = np.array(
            [
                [Fraction(entry) for entry in row]
                for row in SETTINGS[type(self).__name__]["matrix"]
            ],
            dtype=float,
        )
        eigenvectors = np.linalg.eigh(matrix).eigenvectors
        radius = np.ceil(np.linalg.norm(matrix, axis=1).max())
        plane = NumberPlane(x_range=[-radius, radius], y_range=[-radius, radius])
        plane.scale_to_fit_height(config.frame_height)
        origin = plane.c2p(0, 0)
        circle = plane.plot_parametric_curve(
            lambda t: np.array([np.cos(t), np.sin(t), 0]), t_range=[0, 2 * np.pi]
        )
        axes = [
            DashedLine(
                plane.c2p(*(-radius * axis)), plane.c2p(*(radius * axis))
            ).set_color(colour)
            for axis, colour in zip(eigenvectors.T, PALETTE, strict=False)
        ]
        directions = np.hstack([eigenvectors, -eigenvectors])
        angles = np.mod(np.arctan2(directions[1], directions[0]), 2 * np.pi)
        order = np.argsort(angles)
        angle = ValueTracker(0)
        self.add(plane)
        self.play(Create(circle))
        self.play(ApplyMatrix(matrix, circle.copy(), about_point=origin))
        self.play(*map(Create, axes))
        self.add(
            always_redraw(
                lambda: Arrow(
                    origin,
                    plane.c2p(
                        *(
                            matrix
                            @ [np.cos(angle.get_value()), np.sin(angle.get_value())]
                        )
                    ),
                    buff=0,
                    color=PALETTE[3],
                )
            ),
            always_redraw(
                lambda: Arrow(
                    origin,
                    plane.c2p(np.cos(angle.get_value()), np.sin(angle.get_value())),
                    buff=0,
                    color=PALETTE[2],
                )
            ),
        )
        for index in order:
            self.play(angle.animate.set_value(angles[index]))
            self.play(Indicate(axes[index % len(axes)]))
        self.play(angle.animate.set_value(2 * np.pi))
        self.wait()


class SpectralBias(Scene):
    """Gradient descent on ``H^inf`` fitting two sines, Basri et al. (2019), Figures 2 and 6.

    Arora et al. (2019), Section 4, write the dynamics as
    ``u(k+1) = u(k) - eta H^inf (u(k) - y)`` from ``u(0) = 0``, so that
    ``u(k) - y = -sum_i (1 - eta lambda_i)^k (v_i^T y) v_i``. The target of
    Figure 2 is a sum of sines, each an eigenvector of ``H^inf``, so each is
    learned on its own at ratio ``1 - eta lambda_k``. Figure 6 declares
    convergence at 5% of the initial error, which
    ``(1 - eta lambda_k)^t < delta`` reaches at ``t = log delta / log(1 - eta lambda_k)``.
    ``H^inf`` on points uniformly spaced on the circle is circulant (Basri et
    al. 2019, Section 4.1), so its eigenvalues are the discrete Fourier
    transform of one row, the transform ``scipy.linalg.solve_circulant`` divides
    by. Time runs on a log scale, one segment per frequency, each ending where
    that frequency converges, and each frequency keeps one colour for its
    residual, its label and its marker (de Koning et al. 2009, p. 133).
    """

    def construct(self) -> None:
        """Run gradient descent in log time and trace each frequency's residual."""
        settings = SETTINGS[type(self).__name__]
        points = settings["points"]
        theta = 2 * np.pi * np.arange(points) / points
        circle = torch.from_numpy(np.column_stack([np.cos(theta), np.sin(theta)]))
        frequencies = np.array(settings["frequencies"])
        eigenvalues = np.real(
            np.fft.fft(kernels.infinite_width(circle[:1], circle)[0].numpy())
        )[frequencies]
        contraction = 1 - settings["learning_rate"] * eigenvalues
        times = np.log(settings["tolerance"]) / np.log(np.abs(contraction))
        horizon = np.ceil(np.log10(times.max()))
        modes = np.sin(np.outer(theta, frequencies))
        function = Axes(
            x_range=[0, 2 * np.pi], y_range=[-frequencies.size, frequencies.size]
        )
        residual = Axes(
            x_range=[0, horizon],
            y_range=[0, 1],
            tips=False,
            axis_config={"include_numbers": True},
            x_axis_config={"scaling": LogBase(custom_labels=True)},
        )
        counter = Variable(1, "t", var_type=Integer)
        colours = PALETTE[2 : 2 + frequencies.size]
        graphs = [
            residual.plot(
                lambda t, ratio=ratio: ratio**t, use_vectorized=True, color=colour
            )
            for ratio, colour in zip(contraction, colours, strict=True)
        ]
        panels = Group(
            VGroup(
                function,
                function.get_axis_labels(MathTex(r"\theta"), MathTex("u")),
                function.plot_line_graph(
                    theta, modes.sum(1), line_color=PALETTE[0], add_vertex_dots=False
                ),
            ),
            Group(
                residual,
                residual.get_axis_labels(counter, MathTex(r"(1 - \eta \lambda_k)^t")),
                residual.get_horizontal_line(
                    residual.c2p(10**horizon, settings["tolerance"])
                ),
                *graphs,
                *(
                    residual.get_graph_label(
                        graph, f"k = {frequency}", x_val=time, direction=UR, dot=True
                    )
                    for graph, frequency, time in zip(
                        graphs, frequencies, times, strict=True
                    )
                ),
            ),
        )
        panels.arrange(DOWN).scale_to_fit_height(config.frame_height)
        steps = ValueTracker(0)
        counter.add_updater(
            lambda variable: variable.tracker.set_value(10 ** steps.get_value())
        )
        self.add(
            panels,
            VMobject(stroke_color=PALETTE[1]).add_updater(
                lambda curve: curve.set_points_as_corners(
                    function.c2p(
                        np.column_stack(
                            [theta, modes @ (1 - contraction**10 ** steps.get_value())]
                        )
                    )
                ),
                call_updater=True,
            ),
            *(
                Dot(color=colour).add_updater(
                    lambda dot, ratio=ratio: dot.move_to(
                        residual.c2p(
                            10 ** steps.get_value(), ratio**10 ** steps.get_value()
                        )
                    ),
                    call_updater=True,
                )
                for ratio, colour in zip(contraction, colours, strict=True)
            ),
        )
        for time in np.sort(times):
            self.play(steps.animate.set_value(np.log10(time)))
        self.wait()
