"""The LaTeX strings for the quantities in the report and the deck.

Every scene takes its symbols from here. The deck follows the restructured
report, report_v2/overleaf/main.tex, and so do the strings below, so a viewer
who has read the report meets the same notation in the clips. The statistics
S_t, R_t and D_t are also those of the golden report, Equation 6. The kernel is
the tangent kernel of the summed logits over sqrt(p), on the 53 test pairs of a
run (report_v2 Section 3.3).

Do not write these symbols as unicode in a Text object. They go through
MathTex so that they render the way the report renders them.
"""

# The tangent kernel, as slide 5 writes it.
K = r"K"
NTK = r"K(x, x') = \langle \nabla_\theta f(x;\theta),\, \nabla_\theta f(x';\theta) \rangle"

# The three statistics, report_v2 Equation eq:statistics. S_t is a plain ratio,
# so 1 means no change. R_t is one minus the uncentred cosine to step 0. A_t is
# the centred alignment with the label kernel, with C = I - (1/53) 1 1^T.
SCALE = r"S_t = \frac{\lVert K_t \rVert_F}{\lVert K_0 \rVert_F}"
ROTATION = r"R_t = 1 - \frac{\langle K_t, K_0 \rangle_F}{\lVert K_t \rVert_F\,\lVert K_0 \rVert_F}"
ALIGNMENT = (
    r"A_t = \frac{\langle C K_t C,\, C Y Y^\top C \rangle_F}"
    r"{\lVert C K_t C \rVert_F\,\lVert C Y Y^\top C \rVert_F}"
)
LABEL_KERNEL = r"Y Y^\top"

# The relative variation and the identity that ties it to S_t and R_t,
# report_v2 Equation eq:identity and its split form on slide 12.
MOVEMENT = r"D_t = \frac{\lVert K_t - K_0 \rVert_F}{\lVert K_0 \rVert_F}"
IDENTITY = r"D_t^2 = S_t^2 - 2 S_t (1 - R_t) + 1"
IDENTITY_SPLIT = r"D_t^2 = (S_t - 1)^2 + 2 S_t R_t"
SCALE_PART = r"(S_t - 1)^2"
ROTATION_PART = r"2 S_t R_t"
PURE_SCALE_LIMIT = r"R_t = 0 \;\Rightarrow\; D_t = \lvert S_t - 1 \rvert"
PURE_ROTATION_LIMIT = r"S_t = 1 \;\Rightarrow\; D_t^2 = 2 R_t"

# The weight-decay rate, as the product the report writes. ntk_lib calls it
# eta_kappa.
DECAY = r"\eta\lambda"

# Offsets between two pairs (a, b) and (a', b'), both mod p.
OFFSET_A = r"a - a'"
OFFSET_B = r"b - b'"
SUM_LINE = r"a + b \equiv a' + b'"
DIFFERENCE_LINE = r"a - b \equiv a' - b'"

# The laziness knob, as slide 6 writes it. Kumar et al. (2024), Appendix 8.1,
# Equation 7, arXiv v3.
PREDICTOR = r"\tilde f(x;\theta) = \alpha\,[\,f(x;\theta) - f(x;\theta_0)\,]"
LEARNING_RATE = r"\eta = \eta_0 / \alpha^2"
STEP_PARAMETERS = r"\Delta\theta = -\frac{\eta_0}{\alpha}\, r\, \nabla_\theta f"
STEP_OUTPUT = r"\Delta \tilde f = \eta_0\, r\, \lVert \nabla_\theta f \rVert^2"

# Timing. T is the grokking step: the first checkpoint at which the test
# accuracy reaches 0.95, counted from step 0 (report_v2 Section 3.4).
GROK_TIME = r"T"

# The Tier 2 fit, report_v2 Equation eq:aft, without its noise term. Both
# logarithms are natural and centred at the middle of the grid.
FIT = r"\log T = c + a \log N + b \log(\eta\lambda) + \gamma \log N \log(\eta\lambda)"
SHAPE_CLOCK = r"a > 0,\quad b \approx 0"
DECAY_CLOCK = r"a \approx 0,\quad b \approx -1"

# The strings below belong to the scenes TurnAndAim and AimOverTraining,
# which the deck does not show. They work on the centred probe
# kernel of the dense runs, where docs/report.tex of September wrote S_t as a
# log ratio. The deck does not use them.
K_CENTRED = r"\widetilde{K}"
K_NORMALISED = r"\widehat{K}"
CENTRING = r"H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top"
TARGET_GRAM = r"\widetilde{G} = H Y Y^\top H"

# The aim and the identity that ties alignment to rotation. This is the
# group's own algebra, written out in repoduced-code/term_dependence.py. It is
# not from a paper. k_t is the centred kernel over its norm, and u_t is the
# unit vector along the part of k_t that is orthogonal to k_0.
K_UNIT = r"\widehat{\widetilde{K}}"
TARGET_UNIT = r"\widehat{\widetilde{G}}"
AIM = r"\gamma_t"
SPLIT = (
    r"\widehat{\widetilde{K}}_t = (1 - R_t)\,\widehat{\widetilde{K}}_0"
    r" + \sqrt{R_t(2 - R_t)}\;u_t"
)
AIM_DEFINITION = (
    r"\gamma_t = \frac{\langle u_t, \widehat{\widetilde{G}} \rangle}{\sqrt{1 - A_0^2}}"
    r",\qquad u_t \perp \widehat{\widetilde{K}}_0"
)
ALIGNMENT_IDENTITY = (
    r"A_t = A_0\,(1 - R_t) + \gamma_t \sqrt{1 - A_0^2}\,\sqrt{R_t(2 - R_t)}"
)
NORM_RATIO = r"e^{S_t}"
KERNEL_CHANGE = r"\widehat{\widetilde{K}}_t - \widehat{\widetilde{K}}_0"
