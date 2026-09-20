"""The LaTeX strings for the quantities in the report.

Every scene takes its symbols from here. The strings match the macros in
docs/report.tex, so a viewer who has read the report meets the same notation
in the deck. The report defines \\Kh as \\widehat{K}, \\Kt as \\widetilde{K},
\\tgrok as t_{\\mathrm{grok}} and \\Astar as A_{*}.

Do not write these symbols as unicode in a Text object. They go through
MathTex so that they render the way the report renders them.
"""

# Kernels. K is the empirical NTK on the probe set, K-tilde is its centred
# version H K H, K-hat is the normalised version K / ||K||_F.
K = r"K"
K_CENTRED = r"\widetilde{K}"
K_NORMALISED = r"\widehat{K}"
NTK = r"\Theta_t(x, x') = \langle \nabla_\theta f(x; \theta_t), \nabla_\theta f(x'; \theta_t) \rangle"

# The three timing variables, as defined in the Kernel metrics section of
# docs/report.tex.
SCALE = r"S_t = \log\frac{\lVert \widetilde{K}_t \rVert_F}{\lVert \widetilde{K}_0 \rVert_F}"
ROTATION = r"R_t = 1 - \langle \widehat{\widetilde{K}}_t, \widehat{\widetilde{K}}_0 \rangle_F"
ALIGNMENT = (
    r"A_t = \frac{\langle \widetilde{K}_t, \widetilde{G} \rangle}"
    r"{\lVert \widetilde{K}_t \rVert_F \lVert \widetilde{G} \rVert_F}"
)
CENTRING = r"H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top"
TARGET_GRAM = r"\widetilde{G} = H Y Y^\top H"

# The movement statistic and its decomposition, Equation eq:decomp.
MOVEMENT = r"D = \frac{\lVert K_t - K_0 \rVert_F}{\lVert K_0 \rVert_F}"
DECOMPOSITION = (
    r"\frac{\lVert K_t - K_0 \rVert_F^2}{\lVert K_0 \rVert_F^2}"
    r" = e^{2S_t} + 1 - 2e^{S_t}(1 - R_t)"
)
PURE_SCALE_LIMIT = r"R_t = 0 \;\Rightarrow\; D^2 = (1 - e^{S_t})^2"
PURE_ROTATION_LIMIT = r"S_t = 0 \;\Rightarrow\; D^2 = 2R_t"

# The laziness knob. Kumar et al. (2024), Appendix 8.1, Equation 7.
PREDICTOR = r"\tilde f(x, \theta) = \alpha\,[\,f(x, \theta) - f(x, \theta_0)\,]"
LEARNING_RATE = r"\eta = \eta_0 / \alpha^2"
STEP_PARAMETERS = r"\Delta\theta = -\frac{\eta_0}{\alpha}\, r\, \nabla_\theta f"
STEP_OUTPUT = r"\Delta \tilde f = \eta_0\, r\, \lVert \nabla_\theta f \rVert^2"

# Timing.
GROK_TIME = r"t_{\mathrm{grok}}"
THRESHOLD = r"A_{*}"
TAU_TRAIN = r"\tau_{\mathrm{train}}"
TAU_TEST = r"\tau_{\mathrm{test}}"

# The two clocks, from the discriminating prediction in the report.
FIT = r"\log t_{\mathrm{grok}} = c + a \log N + b \log(\eta\lambda)"
SHAPE_CLOCK = r"a > 0,\quad b \approx 0"
DECAY_CLOCK = r"a \approx 0,\quad b \approx -1"
